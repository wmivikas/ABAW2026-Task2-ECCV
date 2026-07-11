"""
Audio Classifier for A/H Detection.

Uses HuBERT-Base on extracted audio waveforms.
HuBERT learns speech representations via offline k-means cluster pseudo-labels,
capturing prosodic features: speech rate changes, pauses, pitch variation,
filler words — all strong A/H indicators.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import HubertModel


class AudioClassifier(nn.Module):
    """
    Audio-based A/H classifier using HuBERT-Base.
    
    Architecture:
        HuBERT-Base (frozen) → temporal features → attention pooling
        → Question embedding → MLP head → binary classification
    """
    
    def __init__(
        self,
        backbone: str = "facebook/hubert-large-ll60k",
        num_classes: int = 2,
        feature_dim: int = 1024,
        dropout: float = 0.3,
        freeze_backbone: bool = True,
        use_question_embedding: bool = True,
        question_embed_dim: int = 32,
    ):
        super().__init__()
        
        self.freeze_backbone = freeze_backbone
        self.use_question_embedding = use_question_embedding
        
        # Load HuBERT
        self.backbone = HubertModel.from_pretrained(backbone)
        backbone_dim = self.backbone.config.hidden_size  # 1024
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Project backbone features
        self.projection = nn.Linear(backbone_dim, feature_dim)
        
        # Attention pooling over time
        self.attention_pool = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        
        # Question embedding
        if use_question_embedding:
            self.question_embedding = nn.Embedding(8, question_embed_dim)
            classifier_input_dim = feature_dim + question_embed_dim
        else:
            classifier_input_dim = feature_dim
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input_dim),
            nn.Dropout(dropout),
            nn.Linear(classifier_input_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
    
    def forward(
        self,
        waveform: torch.Tensor,
        question_num: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            waveform: (batch, num_samples) raw audio
            question_num: (batch,)
        
        Returns:
            logits: (batch, 2)
        """
        # Extract audio features
        if self.freeze_backbone:
            with torch.no_grad():
                outputs = self.backbone(waveform)
        else:
            outputs = self.backbone(waveform)
        
        # Hidden states: (batch, time_steps, backbone_dim)
        hidden = outputs.last_hidden_state
        
        # Project
        features = self.projection(hidden)  # (batch, time_steps, feature_dim)
        
        # Attention pooling
        attn_weights = self.attention_pool(features)  # (batch, time_steps, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        pooled = (features * attn_weights).sum(dim=1)  # (batch, feature_dim)
        
        # Question embedding
        if self.use_question_embedding and question_num is not None:
            q_embed = self.question_embedding(question_num)
            pooled = torch.cat([pooled, q_embed], dim=-1)
        
        # Classify
        logits = self.classifier(pooled)
        return logits
    
    def get_features(self, waveform: torch.Tensor, **kwargs) -> torch.Tensor:
        """Extract audio features (for multimodal fusion)."""
        if self.freeze_backbone:
            with torch.no_grad():
                outputs = self.backbone(waveform)
        else:
            outputs = self.backbone(waveform)
        
        hidden = outputs.last_hidden_state
        features = self.projection(hidden)
        
        attn_weights = self.attention_pool(features)
        attn_weights = F.softmax(attn_weights, dim=1)
        pooled = (features * attn_weights).sum(dim=1)
        
        return pooled

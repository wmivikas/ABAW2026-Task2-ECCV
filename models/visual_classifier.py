"""
Visual Classifier for A/H Detection.

Uses VideoMAE-Base on cropped/aligned face frame sequences.
Unlike CLIP (single-frame), VideoMAE processes 16 frames as a spatiotemporal
sequence, capturing temporal hesitancy cues like micro-expression shifts,
gaze aversion, and facial freezing.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualClassifier(nn.Module):
    """
    Visual A/H classifier using VideoMAE-Base backbone on face frames.
    
    Architecture:
        VideoMAE-Base (frozen) → spatiotemporal token features → mean pooling
        → Question embedding → MLP head → binary classification
    """
    
    def __init__(
        self,
        backbone: str = "MCG-NJU/videomae-base",
        num_classes: int = 2,
        feature_dim: int = 768,
        dropout: float = 0.3,
        freeze_backbone: bool = True,
        use_question_embedding: bool = True,
        question_embed_dim: int = 32,
        **kwargs,
    ):
        super().__init__()
        
        self.use_question_embedding = use_question_embedding
        self.freeze_backbone = freeze_backbone
        
        # Load VideoMAE model
        from transformers import VideoMAEModel
        self.backbone = VideoMAEModel.from_pretrained(backbone)
        backbone_dim = self.backbone.config.hidden_size  # 768 for base
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Project backbone features to feature_dim
        self.projection = nn.Linear(backbone_dim, feature_dim)
        
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
        frames: torch.Tensor,
        question_num: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            frames: (batch, num_frames, 3, H, W)
            question_num: (batch,)
        
        Returns:
            logits: (batch, 2)
        """
        # VideoMAE expects pixel_values of shape (batch, num_frames, C, H, W)
        if self.freeze_backbone:
            with torch.no_grad():
                outputs = self.backbone(pixel_values=frames)
        else:
            outputs = self.backbone(pixel_values=frames)
        
        # VideoMAE outputs: last_hidden_state (batch, num_patches, backbone_dim)
        # Mean pool over all patch tokens
        features = outputs.last_hidden_state  # (batch, num_patches, 768)
        pooled = features.mean(dim=1)  # (batch, 768)
        
        # Project
        pooled = self.projection(pooled)  # (batch, feature_dim)
        
        # Concatenate question embedding
        if self.use_question_embedding and question_num is not None:
            q_embed = self.question_embedding(question_num)
            pooled = torch.cat([pooled, q_embed], dim=-1)
        
        # Classify
        logits = self.classifier(pooled)
        return logits
    
    def get_features(self, frames: torch.Tensor, **kwargs) -> torch.Tensor:
        """Extract visual features (for multimodal fusion)."""
        if self.freeze_backbone:
            with torch.no_grad():
                outputs = self.backbone(pixel_values=frames)
        else:
            outputs = self.backbone(pixel_values=frames)
        
        features = outputs.last_hidden_state
        pooled = features.mean(dim=1)
        return self.projection(pooled)

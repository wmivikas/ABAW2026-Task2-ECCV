"""
Text Classifier for A/H Detection.

Fine-tunes RoBERTa-GoEmotions on transcript text with question-aware prompting.
GoEmotions covers 27 emotion categories including confusion, doubt, and uncertainty,
making it ideal for detecting hesitancy language.
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class TextClassifier(nn.Module):
    """
    Text-based A/H classifier using RoBERTa-GoEmotions.
    
    Architecture:
        RoBERTa encoder → [CLS] pooling → Question embedding (optional) 
        → MLP head → binary classification
    
    The question-aware prompt is handled in the dataset (dataset.py).
    Here we just optionally concatenate a question number embedding.
    """
    
    def __init__(
        self,
        model_name: str = "SamLowe/roberta-base-go_emotions",
        num_classes: int = 2,
        dropout: float = 0.1,
        use_question_embedding: bool = True,
        question_embed_dim: int = 32,
        freeze_layers: int = 0,
    ):
        super().__init__()
        
        self.use_question_embedding = use_question_embedding
        
        # Load pretrained encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size  # 1024 for large
        
        # Optionally freeze bottom layers for faster training
        if freeze_layers > 0:
            for name, param in self.encoder.named_parameters():
                if "embeddings" in name:
                    param.requires_grad = False
                for i in range(freeze_layers):
                    if f"layer.{i}." in name:
                        param.requires_grad = False
        
        # Question number embedding (1-7)
        if use_question_embedding:
            self.question_embedding = nn.Embedding(8, question_embed_dim)  # 0=pad, 1-7
            classifier_input_dim = hidden_size + question_embed_dim
        else:
            classifier_input_dim = hidden_size
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(classifier_input_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        question_num: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            question_num: (batch,) question numbers 1-7
            token_type_ids: (batch, seq_len) optional
        
        Returns:
            logits: (batch, 2) classification logits
        """
        # Encode text
        encoder_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids
        
        outputs = self.encoder(**encoder_kwargs)
        
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]  # (batch, hidden_size)
        
        # Optionally concatenate question embedding
        if self.use_question_embedding and question_num is not None:
            q_embed = self.question_embedding(question_num)  # (batch, q_dim)
            cls_output = torch.cat([cls_output, q_embed], dim=-1)
        
        # Classify
        logits = self.classifier(cls_output)
        return logits
    
    def get_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Extract text features (for multimodal fusion)."""
        encoder_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids
        
        outputs = self.encoder(**encoder_kwargs)
        return outputs.last_hidden_state[:, 0, :]  # (batch, hidden_size)


def get_tokenizer(model_name: str = "SamLowe/roberta-base-go_emotions"):
    """Load the tokenizer for the text model."""
    return AutoTokenizer.from_pretrained(model_name)

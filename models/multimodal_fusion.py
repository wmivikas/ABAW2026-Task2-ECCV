"""
Orthogonal Conflict & Correlation Fusion (OCCF) Model for A/H Detection.

This novel architecture replaces standard absolute difference or cross-attention with
Orthogonal Decomposition. For any two modalities (e.g., Visual and Text), it computes:
1. Agreement (Parallel Projection): What the face and text agree on.
2. Conflict (Orthogonal Rejection): The pure facial cues that contradict the text.

This mathematically isolates the cross-modal conflicts that define Ambivalence/Hesitancy.
"""
import torch
import torch.nn as nn


def get_orthogonal_components(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8):
    """
    Computes the orthogonal decomposition of x with respect to y.
    
    Args:
        x: (batch, dim) tensor to be decomposed
        y: (batch, dim) tensor representing the basis
        eps: small epsilon to avoid division by zero
        
    Returns:
        proj_x_on_y: (batch, dim) The agreement (parallel) component
        conflict_x: (batch, dim) The conflict (orthogonal) component
    """
    # x dot y
    dot_xy = (x * y).sum(dim=-1, keepdim=True)
    # ||y||^2
    norm_y_sq = (y * y).sum(dim=-1, keepdim=True)
    
    # Agreement: projection of x onto y
    proj_x_on_y = (dot_xy / (norm_y_sq + eps)) * y
    
    # Conflict: rejection of x from y
    conflict_x = x - proj_x_on_y
    
    return proj_x_on_y, conflict_x


class MultimodalFusionModel(nn.Module):
    """
    Orthogonal Conflict & Correlation Fusion (OCCF) Model.
    
    Architecture:
        1. Project each modality to a shared 512-dim space.
        2. Compute Orthogonal Conflict and Parallel Agreement for:
           - Visual relative to Text
           - Audio relative to Text
           - Visual relative to Audio
        3. Concatenate base features, agreements, and conflicts (9 * 512 = 4608 dims).
        4. Fuse via a deep MLP bottleneck.
        5. Append Question Embeddings.
        6. Classification head.
    """
    
    def __init__(
        self,
        text_dim: int = 768,
        visual_dim: int = 768,
        audio_dim: int = 1024,
        hidden_dim: int = 512,
        num_classes: int = 2,
        dropout: float = 0.3,
        use_question_embedding: bool = True,
        question_embed_dim: int = 32,
        conflict_type: str = "orthogonal",
        **kwargs,
    ):
        super().__init__()

        self.use_question_embedding = use_question_embedding
        # conflict_type controls the cross-modal interaction terms:
        #   "orthogonal": agreement (projection) + conflict (rejection) per pair (OCCF, ours)
        #   "absdiff":    |x - y| per pair  (the ConflictAwareAH operator, for ablation)
        #   "none":       no interaction terms; base modalities only
        assert conflict_type in ("orthogonal", "absdiff", "none")
        self.conflict_type = conflict_type
        
        # Project each modality to shared dimension
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.visual_projection = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.audio_projection = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Number of concatenated hidden_dim-blocks depends on conflict_type:
        #   orthogonal: T,V,A + (proj,conf) x 3 pairs             = 9 blocks
        #   absdiff:    T,V,A + |diff| x 3 pairs                  = 6 blocks
        #   none:       T,V,A                                     = 3 blocks
        n_blocks = {"orthogonal": 9, "absdiff": 6, "none": 3}[conflict_type]
        fusion_input_dim = n_blocks * hidden_dim
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        # Question embedding
        if use_question_embedding:
            self.question_embedding = nn.Embedding(8, question_embed_dim)
            classifier_input_dim = hidden_dim + question_embed_dim
        else:
            classifier_input_dim = hidden_dim
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(classifier_input_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes),
        )
    
    def forward(
        self,
        text_features: torch.Tensor,
        visual_features: torch.Tensor,
        audio_features: torch.Tensor,
        question_num: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            text_features: (batch, text_dim) pre-extracted text features
            visual_features: (batch, visual_dim) pre-extracted visual features
            audio_features: (batch, audio_dim) pre-extracted audio features
            question_num: (batch,) question numbers 1-7
        
        Returns:
            logits: (batch, 2)
        """
        # Project to shared dimension
        t = self.text_projection(text_features)      # (batch, hidden_dim)
        v = self.visual_projection(visual_features)  # (batch, hidden_dim)
        a = self.audio_projection(audio_features)    # (batch, hidden_dim)
        
        # Cross-modal interaction terms (text is the anchor modality)
        if self.conflict_type == "orthogonal":
            # Agreement (parallel projection) + conflict (orthogonal rejection)
            a_vt, c_vt = get_orthogonal_components(v, t)
            a_at, c_at = get_orthogonal_components(a, t)
            a_va, c_va = get_orthogonal_components(v, a)
            fused_raw = torch.cat([t, v, a, a_vt, c_vt, a_at, c_at, a_va, c_va], dim=-1)
        elif self.conflict_type == "absdiff":
            # ConflictAwareAH operator: element-wise absolute difference per pair
            fused_raw = torch.cat(
                [t, v, a, (v - t).abs(), (a - t).abs(), (v - a).abs()], dim=-1)
        else:  # "none": base modalities only
            fused_raw = torch.cat([t, v, a], dim=-1)

        # Compress the concatenated features
        fused = self.fusion_mlp(fused_raw)
        
        # Add question embedding
        if self.use_question_embedding and question_num is not None:
            q_embed = self.question_embedding(question_num)
            fused = torch.cat([fused, q_embed], dim=-1)
        
        # Classify
        logits = self.classifier(fused)
        return logits

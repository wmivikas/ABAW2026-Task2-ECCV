"""
Enhanced Multimodal Fusion v2 — Cross-Modal Attention + Conflict Gates

Instead of static orthogonal decomposition (OCCF), use:
 1. Per-modality self-attention (query own modality)
 2. Cross-modal attention (each modality attends to others)
 3. Conflict gates (learn when to suppress agreement, amplify conflict)
 4. Learnable fusion weights per gate

This learns which conflicts matter, rather than using fixed math.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    """Compute attention of query modality to key/value modality."""

    def __init__(self, dim: int, n_heads: int = 8):
        super().__init__()
        self.n_heads = n_heads
        self.scale = (dim // n_heads) ** -0.5

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, query, key_value):
        """query, key_value: (B, dim)"""
        q = self.q(query).view(query.shape[0], self.n_heads, -1)  # (B, n_heads, d_head)
        k = self.k(key_value).view(key_value.shape[0], self.n_heads, -1)
        v = self.v(key_value).view(key_value.shape[0], self.n_heads, -1)

        # Per-head compatibility score (B, n_heads); a sigmoid gate per head
        # decides how much of the key/value modality flows through each head.
        scores = (q * k).sum(dim=-1) * self.scale
        gate = torch.sigmoid(scores)  # (B, n_heads)

        # Gate value heads, then concatenate heads back to (B, dim).
        out = (gate.unsqueeze(-1) * v).reshape(query.shape[0], -1)
        return self.proj(out)


class ConflictGate(nn.Module):
    """Learn when to use agreement vs conflict between two modalities."""

    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, agreement, conflict):
        """Blend agreement (alpha) and conflict (1-alpha)."""
        alpha = self.gate(torch.cat([agreement, conflict], dim=-1))
        return alpha * agreement + (1 - alpha) * conflict


class MultimodalFusionV2(nn.Module):
    """
    Cross-modal attention fusion with learned conflict gating.

    Per pair (A, B):
     1. Project A, B to shared dim
     2. A attends to B (and vice versa)
     3. Compute agreement = (A*B) (element-wise)
     4. Compute conflict = A - (A·B/||B||²) B (orthogonal component)
     5. Gate decides: use agreement or conflict more?
     6. Concatenate all gated pairs
     7. Final MLP + classify
    """

    def __init__(
        self,
        text_dim: int = 768,
        visual_dim: int = 768,
        audio_dim: int = 1024,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.3,
        use_question_embedding: bool = True,
        question_embed_dim: int = 32,
        **kwargs,
    ):
        super().__init__()

        def proj(in_dim):
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        self.text_projection = proj(text_dim)
        self.visual_projection = proj(visual_dim)
        self.audio_projection = proj(audio_dim)

        # Cross-modal attention for each pair
        self.attn_vt = CrossModalAttention(hidden_dim, n_heads=4)
        self.attn_at = CrossModalAttention(hidden_dim, n_heads=4)
        self.attn_va = CrossModalAttention(hidden_dim, n_heads=4)

        # Conflict gates for each pair
        self.gate_vt = ConflictGate(hidden_dim)
        self.gate_at = ConflictGate(hidden_dim)
        self.gate_va = ConflictGate(hidden_dim)

        # Fusion MLP: concatenate [t, v, a, attn_v→t, attn_a→t, attn_v→a, gated_vt, gated_at, gated_va]
        fusion_in = 9 * hidden_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_in, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        if use_question_embedding:
            self.question_embedding = nn.Embedding(8, question_embed_dim)
            classifier_in = hidden_dim + question_embed_dim
        else:
            classifier_in = hidden_dim

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(classifier_in, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        text_features: torch.Tensor,
        visual_features: torch.Tensor,
        audio_features: torch.Tensor,
        question_num: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Project, attend, gate, fuse, classify."""
        t = self.text_projection(text_features)
        v = self.visual_projection(visual_features)
        a = self.audio_projection(audio_features)

        # Cross-modal attention (each modality attends to the others)
        v_to_t = self.attn_vt(v, t)
        a_to_t = self.attn_at(a, t)
        v_to_a = self.attn_va(v, a)

        # Compute agreement and conflict for each pair
        # agreement = dot product
        agree_vt = v * t
        agree_at = a * t
        agree_va = v * a

        # conflict = rejection (orthogonal component)
        def orth(x, y, eps=1e-8):
            dot = (x * y).sum(dim=-1, keepdim=True)
            nrm = (y * y).sum(dim=-1, keepdim=True)
            return x - (dot / (nrm + eps)) * y

        conf_vt = orth(v, t)
        conf_at = orth(a, t)
        conf_va = orth(v, a)

        # Gates blend agreement and conflict
        gated_vt = self.gate_vt(agree_vt, conf_vt)
        gated_at = self.gate_at(agree_at, conf_at)
        gated_va = self.gate_va(agree_va, conf_va)

        # Concatenate all 9 components and fuse
        fused = self.fusion_mlp(torch.cat([t, v, a, v_to_t, a_to_t, v_to_a,
                                           gated_vt, gated_at, gated_va], dim=-1))

        # Add question embedding
        if hasattr(self, 'question_embedding') and question_num is not None:
            q_embed = self.question_embedding(question_num.clamp(0, 7))
            fused = torch.cat([fused, q_embed], dim=-1)

        logits = self.classifier(fused)
        return logits

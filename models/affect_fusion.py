"""
Affective Marker Fusion (AMF) — the NEW pipeline's prediction model.

Consumes affect-specialised inputs (see extract_affect_features.py):
  - text_emb       (768) : RoBERTa-GoEmotions semantic anchor (reused)
  - text_markers   (11)  : interpretable psycholinguistic hesitation cues
  - fer_stats      (14)  : facial-expression mean+std over the clip (interpretable)
  - visual_emb     (768) : FER-ViT embedding (facial affect, not Kinetics motion)
  - audio_emb      (1024): emotion-fine-tuned wav2vec2 (prosodic affect/arousal)

Design: each stream is projected and gated by a learned per-stream reliability
weight (so near-chance channels can be down-weighted rather than injected as
noise — the failure mode of prior generic-feature fusion). The interpretable
marker streams (text_markers, fer_stats) are kept as a separate low-dim branch
whose contribution is exposed for explanation.
"""
import torch
import torch.nn as nn


class AffectMarkerFusion(nn.Module):
    def __init__(self, d=256, dropout=0.3, question_embed_dim=32, n_classes=1,
                 markers_dim=11 + 14, **kw):
        super().__init__()

        def proj(i, o=d):
            return nn.Sequential(nn.Linear(i, o), nn.LayerNorm(o), nn.GELU(), nn.Dropout(dropout))

        # dense affect embeddings
        self.p_text = proj(768)
        self.p_visual = proj(768)
        self.p_audio = proj(1024)
        # interpretable low-dim markers: text(11) + FER stats(14) + TPH prosody(28)
        self.p_markers = proj(markers_dim, d // 2)

        # per-stream learned reliability gate (scalar in (0,1) per stream)
        self.gate = nn.Sequential(nn.Linear(3 * d + d // 2, 4), nn.Sigmoid())

        self.q_embed = nn.Embedding(8, question_embed_dim)
        fused_in = 3 * d + d // 2 + question_embed_dim
        self.head = nn.Sequential(
            nn.Linear(fused_in, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )
        # auxiliary interpretable head: A/H from markers alone (for explanation + regularisation)
        self.marker_head = nn.Sequential(nn.Linear(d // 2, 64), nn.GELU(),
                                         nn.Dropout(dropout), nn.Linear(64, n_classes))

    def forward(self, text_emb, visual_emb, audio_emb, markers, question_num=None, **kw):
        t = self.p_text(text_emb)
        v = self.p_visual(visual_emb)
        a = self.p_audio(audio_emb)
        m = self.p_markers(markers)

        cat = torch.cat([t, v, a, m], dim=-1)
        g = self.gate(cat)                       # (B,4) reliability per stream
        t, v, a, m = g[:, 0:1] * t, g[:, 1:2] * v, g[:, 2:3] * a, g[:, 3:4] * m

        feat = torch.cat([t, v, a, m], dim=-1)
        if question_num is not None:
            q = self.q_embed(question_num.clamp(0, 7))
            feat = torch.cat([feat, q], dim=-1)
        else:
            z = torch.zeros(t.shape[0], self.q_embed.embedding_dim, device=t.device)
            feat = torch.cat([feat, z], dim=-1)

        logit = self.head(feat).squeeze(-1)
        marker_logit = self.marker_head(m).squeeze(-1)
        return {"logit": logit, "marker_logit": marker_logit, "gates": g}

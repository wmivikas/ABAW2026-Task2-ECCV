"""
Shared inference for the Affective Marker Fusion (AMF) member.

Scores an arbitrary list of BAH sample dicts (from parse_split_file) using a
saved affect checkpoint. Standardisation uses the norm_stats stored in the
checkpoint (fit on the pooled labelled data), so private-test scoring is
reproducible without recomputing statistics.
"""
import os
import numpy as np
import torch

import config
from models.affect_fusion import AffectMarkerFusion

FEAT = config.FEATURES_DIR
AFF = os.path.join(FEAT, "affect")


def _load_video(sample):
    stem = sample["basename"].replace(".mp4", ".pt")
    text_emb = torch.load(os.path.join(FEAT, "text", stem), map_location="cpu", weights_only=True).float()
    vis = torch.load(os.path.join(AFF, "visual", stem), map_location="cpu", weights_only=False)
    aud = torch.load(os.path.join(AFF, "audio", stem), map_location="cpu", weights_only=False)
    mk = torch.load(os.path.join(AFF, "text_markers", stem), map_location="cpu", weights_only=False).float()
    return {
        "text_emb": text_emb,
        "visual_emb": vis["embedding"].float(),
        "audio_emb": aud["embedding"].float(),
        "markers": torch.cat([mk, vis["fer_stats"].float()]),  # 11 + 14 = 25
        "q": int(sample["question_num"]),
    }


@torch.no_grad()
def infer_affect(checkpoint, samples, device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    # infer marker-vector dim from the saved p_markers input layer
    md = ckpt["model_state_dict"]["p_markers.0.weight"].shape[1]
    model = AffectMarkerFusion(d=ckpt["args"].get("d_model", 256), markers_dim=md).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])
    ns = ckpt["norm_stats"]  # dict key -> (mu, sd) tensors

    data = {k: [] for k in ["text_emb", "visual_emb", "audio_emb", "markers"]}
    qs = []
    for s in samples:
        d = _load_video(s)
        for k in data:
            data[k].append(d[k])
        qs.append(d["q"])
    tensors = {k: torch.stack(v) for k, v in data.items()}
    for k, (mu, sd) in ns.items():
        tensors[k] = (tensors[k] - mu.cpu()) / sd.cpu()
    q = torch.tensor(qs, dtype=torch.long)

    probs = []
    n = len(samples)
    for i in range(0, n, 256):
        sl = slice(i, min(i + 256, n))
        r = model(tensors["text_emb"][sl].to(device), tensors["visual_emb"][sl].to(device),
                  tensors["audio_emb"][sl].to(device), tensors["markers"][sl].to(device),
                  q[sl].to(device))
        probs.append(torch.sigmoid(r["logit"]).cpu())
    return torch.cat(probs).numpy()

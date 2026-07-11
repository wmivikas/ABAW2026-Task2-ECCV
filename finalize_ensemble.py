"""
Lock final ensemble weights + threshold for the private-test submission.

All 5 final models were trained with --pool-final: train+val+test pooled,
with the SAME seeded stratified split (seed=args.seed, 8% holdout) carving
out the SAME 113 videos across every model (verified: same pool order, same
rng seed). Those 113 videos were held out of every model's own training, so
scoring them here is honest — not leakage.

Usage:
    python finalize_ensemble.py
"""
import os
import glob
import random
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from sklearn.metrics import f1_score, average_precision_score

import config
from dataset import BAHDataset, parse_split_file
from predict import load_model, predict as predict_raw


def latest_pool_final_checkpoint(model_type: str, fusion_version: str = None):
    """Find the most recent --pool-final checkpoint for a given model type
    (and fusion_version, if model_type=='fusion'). Avoids hardcoding
    timestamped run directories so this script works after a fresh run."""
    candidates = []
    for d in sorted(glob.glob(os.path.join(config.CHECKPOINT_DIR, f"{model_type}_*"))):
        ckpt = os.path.join(d, "best_model.pt")
        if not os.path.isfile(ckpt):
            continue
        meta = torch.load(ckpt, map_location="cpu", weights_only=False)
        a = meta.get("args", {})
        if not a.get("pool_final"):
            continue
        if model_type == "fusion" and a.get("fusion_version", "v1") != (fusion_version or "v1"):
            continue
        candidates.append((d, ckpt))
    if not candidates:
        raise FileNotFoundError(
            f"No --pool-final checkpoint found for model_type={model_type} "
            f"fusion_version={fusion_version}. Run train.py --pool-final first.")
    return sorted(candidates)[-1][1]  # latest by directory timestamp


def reconstruct_holdout(seed=42):
    from holdout import split_pool_final
    pool = (parse_split_file(config.TRAIN_SPLIT)
            + parse_split_file(config.VAL_SPLIT)
            + parse_split_file(config.TEST_SPLIT))
    _, holdout = split_pool_final(pool, seed=seed)
    return holdout


def infer(checkpoint, model_type, samples, device):
    model, tokenizer, _ = load_model(checkpoint, model_type, device)
    modalities = [model_type] if model_type != "fusion" else ["text", "visual", "audio"]
    ds = BAHDataset(
        split="val",  # split arg only picks file path internally; we override samples
        modalities=modalities,
        text_tokenizer=tokenizer,
        use_precomputed_features=(model_type == "fusion"),
    )
    ds.samples = samples
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4)
    probs, labels, _ = predict_raw(model, loader, device, model_type)
    return probs, labels


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = torch.device(args.device)
    holdout = reconstruct_holdout(seed=42)
    y = np.array([s["label"] for s in holdout])
    print(f"Holdout: {len(holdout)} videos, {y.sum()} pos / {len(y) - y.sum()} neg")

    members = {
        "text": (latest_pool_final_checkpoint("text"), "text"),
        "visual": (latest_pool_final_checkpoint("visual"), "visual"),
        "audio": (latest_pool_final_checkpoint("audio"), "audio"),
        "fusion_v1": (latest_pool_final_checkpoint("fusion", "v1"), "fusion"),
        "fusion_v2": (latest_pool_final_checkpoint("fusion", "v2"), "fusion"),
    }
    for name, (ckpt, _) in members.items():
        print(f"  {name}: {ckpt}")

    probs = {}
    for name, (ckpt, mtype) in members.items():
        p, lbl = infer(ckpt, mtype, holdout, device)
        assert np.array_equal(lbl, y), f"{name}: label mismatch — holdout reconstruction bug"
        probs[name] = p
        ap = average_precision_score(y, p)
        f1_05 = f1_score(y, (p >= 0.5).astype(int), average="macro")
        print(f"  {name:10s}: AP={ap:.4f}  F1@0.5={f1_05:.4f}")

    # Affect member (new pipeline): decorrelated, affect-specialised features.
    affect_ckpt = os.path.join(config.CHECKPOINT_DIR, "affect_final_seed42.pt")
    if os.path.isfile(affect_ckpt):
        from affect_infer import infer_affect
        probs["affect"] = infer_affect(affect_ckpt, holdout, device)
        members["affect"] = (affect_ckpt, "affect")
        ap = average_precision_score(y, probs["affect"])
        f1_05 = f1_score(y, (probs["affect"] >= 0.5).astype(int), average="macro")
        print(f"  {'affect':10s}: AP={ap:.4f}  F1@0.5={f1_05:.4f}")

    # Weight by holdout AP (matches eval_final.py's honest strategy)
    keys = list(probs.keys())
    w = np.array([average_precision_score(y, probs[k]) for k in keys])
    w = w / w.sum()
    ens = sum(wi * probs[k] for wi, k in zip(w, keys))
    ens_ap = average_precision_score(y, ens)
    ens_f1 = f1_score(y, (ens >= 0.5).astype(int), average="macro")
    print(f"\nEnsemble weights: " + ", ".join(f"{k}={wi:.3f}" for k, wi in zip(keys, w)))
    print(f"Ensemble @ 0.5: AP={ens_ap:.4f}  Macro-F1={ens_f1:.4f}  (n={len(y)}, held-out)")

    cfg = {
        "members": {name: {"checkpoint": ckpt, "model_type": mtype}
                    for name, (ckpt, mtype) in members.items()},
        "weights": {k: float(wi) for k, wi in zip(keys, w)},
        "threshold": 0.5,
        "holdout_ap": float(ens_ap),
        "holdout_macro_f1": float(ens_f1),
        "holdout_n": len(y),
        "seed": 42,
        "note": "Weights are AP-weighted on a 113-video internal holdout, held out "
                "of every member's own --pool-final training. Threshold fixed at 0.5 "
                "(not tuned on any split) to avoid the small-val overfitting we "
                "observed earlier (val-tuned threshold did not transfer to test).",
    }
    out_path = os.path.join(config.OUTPUT_ROOT, "final_ensemble_config.json")
    os.makedirs(config.OUTPUT_ROOT, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()

"""
Train the Affective Marker Fusion (AMF) model — the NEW pipeline's predictor.

Uses affect-specialised features (extract_affect_features.py) + the reused
RoBERTa-GoEmotions text embedding. Same calibration philosophy as the rest of
the project: fixed decision threshold, evaluate ranking with AP.

Modes mirror train.py:
  default      : train on train, early-stop on val, write val/test probs.
  --pool-final : train on all labelled data with a seeded 8% internal holdout.

Usage:
    python train_affect.py --device cuda:0
    python train_affect.py --pool-final --device cuda:0
"""
import os
import copy
import argparse
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, average_precision_score

import config
from dataset import parse_split_file
from holdout import split_pool_final
from models.affect_fusion import AffectMarkerFusion

FEAT = config.FEATURES_DIR
AFF = os.path.join(FEAT, "affect")


def load_split(split_path):
    d = defaultdict(list)
    for s in parse_split_file(split_path):
        stem = s["basename"].replace(".mp4", ".pt")
        text_emb = torch.load(os.path.join(FEAT, "text", stem), map_location="cpu", weights_only=True).float()
        vis = torch.load(os.path.join(AFF, "visual", stem), map_location="cpu", weights_only=False)
        aud = torch.load(os.path.join(AFF, "audio", stem), map_location="cpu", weights_only=False)
        mk = torch.load(os.path.join(AFF, "text_markers", stem), map_location="cpu", weights_only=False).float()
        d["text_emb"].append(text_emb)
        d["visual_emb"].append(vis["embedding"].float())
        d["fer"].append(vis["fer_stats"].float())
        d["audio_emb"].append(aud["embedding"].float())
        d["markers"].append(torch.cat([mk, vis["fer_stats"].float()]))  # 11 + 14 = 25
        d["label"].append(float(s["label"]))
        d["q"].append(s["question_num"]); d["path"].append(s["video_path"])
    out = {k: torch.stack(d[k]) for k in ["text_emb", "visual_emb", "fer", "audio_emb", "markers"]}
    out["label"] = torch.tensor(d["label"]); out["q"] = torch.tensor(d["q"], dtype=torch.long)
    out["path"] = d["path"]
    return out


def cat(a, b):
    return {k: (torch.cat([a[k], b[k]]) if isinstance(a[k], torch.Tensor) else a[k] + b[k]) for k in a}


def standardize(train, *others):
    stats = {}
    for k in ["text_emb", "visual_emb", "audio_emb", "markers"]:
        mu, sd = train[k].mean(0), train[k].std(0).clamp_min(1e-6)
        stats[k] = (mu, sd); train[k] = (train[k] - mu) / sd
        for o in others:
            o[k] = (o[k] - mu) / sd
    return stats


def take(d, idx, device):
    return {k: (v[idx].to(device) if isinstance(v, torch.Tensor) else v) for k, v in d.items()}


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def predict(model, data, device, bs=256):
    model.eval(); n = len(data["label"]); out = []
    for i in range(0, n, bs):
        idx = torch.arange(i, min(i + bs, n))
        b = take(data, idx, device)
        r = model(b["text_emb"], b["visual_emb"], b["audio_emb"], b["markers"], b["q"])
        out.append(torch.sigmoid(r["logit"]).cpu())
    return torch.cat(out).numpy()


def best_threshold(y, p):
    ts = np.arange(0.05, 0.95, 0.01)
    f = [f1_score(y, (p >= t).astype(int), average="macro") for t in ts]
    i = int(np.argmax(f)); return float(ts[i]), float(f[i])


def train_model(train, val, args, device):
    set_seed(args.seed)
    model = AffectMarkerFusion(d=args.d_model, dropout=args.dropout,
                               markers_dim=train["markers"].shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n = len(train["label"]); steps = (n + args.batch_size - 1) // args.batch_size
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=steps * args.epochs, pct_start=0.1)
    pos = train["label"].sum().item()
    pw = torch.tensor((n - pos) / max(pos, 1.0), device=device)
    y_val = val["label"].numpy()
    best_f1, best_state, best_ep, wait = 0.0, None, 0, 0
    for ep in range(args.epochs):
        model.train(); perm = torch.randperm(n)
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            b = take(train, idx, device); y = b["label"]
            ys = y * (1 - args.label_smoothing) + 0.5 * args.label_smoothing
            r = model(b["text_emb"], b["visual_emb"], b["audio_emb"], b["markers"], b["q"])
            loss = F.binary_cross_entropy_with_logits(r["logit"], ys, pos_weight=pw)
            loss = loss + args.marker_weight * F.binary_cross_entropy_with_logits(
                r["marker_logit"], y, pos_weight=pw)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
        _, f1 = best_threshold(y_val, predict(model, val, device))
        if f1 > best_f1:
            best_f1, wait, best_ep = f1, 0, ep + 1
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= args.patience:
                break
    model.load_state_dict(best_state)
    return model, best_f1, best_ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--marker-weight", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--pool-final", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)
    print("Loading affect features ...")
    train = load_split(config.TRAIN_SPLIT); val = load_split(config.VAL_SPLIT); test = load_split(config.TEST_SPLIT)

    if args.pool_final:
        # Must use the SAME holdout as train.py --pool-final and
        # finalize_ensemble.py, otherwise AMF trains on the videos its ensemble
        # weight is later fitted on. See holdout.py.
        pool = cat(cat(train, val), test)
        pool_samples = (parse_split_file(config.TRAIN_SPLIT)
                        + parse_split_file(config.VAL_SPLIT)
                        + parse_split_file(config.TEST_SPLIT))
        assert [s["video_path"] for s in pool_samples] == list(pool["path"]), \
            "pooled tensor order does not match pooled split-file order"
        _, hold_samples = split_pool_final(pool_samples, seed=args.seed)
        hold_paths = set(s["video_path"] for s in hold_samples)
        hold = np.array([i for i, p in enumerate(pool["path"]) if p in hold_paths])
        tr = np.array([i for i, p in enumerate(pool["path"]) if p not in hold_paths])
        train = take(pool, torch.from_numpy(tr), "cpu"); val = take(pool, torch.from_numpy(hold), "cpu")
        print(f"[pool-final] {len(pool['path'])} pooled -> {len(tr)} train / {len(hold)} internal holdout "
              f"(shared holdout from holdout.py)")
        # NOTE: `test` must NOT alias `val` here. standardize() mutates each dict
        # it is handed, so passing the same object twice standardises it twice.
        norm_stats = standardize(train, val)
        test = val
    else:
        norm_stats = standardize(train, val, test)
    y_val, y_test = val["label"].numpy(), test["label"].numpy()
    model, best_f1, best_ep = train_model(train, val, args, device)
    val_p, test_p = predict(model, val, device), predict(model, test, device)
    thr, vf1 = best_threshold(y_val, val_p)
    tag = "internal holdout (pool-final)" if args.pool_final else "PUBLIC TEST"
    print(f"  best val/holdout F1={best_f1:.4f} @ep{best_ep}")
    print(f"VAL F1={vf1:.4f}@{thr:.2f} | {tag}: F1@0.5="
          f"{f1_score(y_test,(test_p>=0.5).astype(int),average='macro'):.4f} "
          f"AP={average_precision_score(y_test,test_p):.4f}")

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    ckpt = f"affect_final_seed{args.seed}.pt" if args.pool_final else f"affect_seed{args.seed}.pt"
    torch.save({"model_state_dict": model.state_dict(), "norm_stats": norm_stats,
                "args": vars(args)},
               os.path.join(config.CHECKPOINT_DIR, ckpt))
    if not args.pool_final:
        os.makedirs(config.PREDICTION_DIR, exist_ok=True)
        np.save(f"{config.PREDICTION_DIR}/affect_val_predictions_probs.npy", val_p)
        np.save(f"{config.PREDICTION_DIR}/affect_test_predictions_probs.npy", test_p)
        print(f"Saved affect probs to {config.PREDICTION_DIR}")


if __name__ == "__main__":
    main()

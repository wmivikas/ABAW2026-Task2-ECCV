"""
Detailed results report for the FINAL (pool-final) models + ensemble.

Prints, for each member and for the ensemble, the full breakdown:
  Macro-F1, per-class F1 (No A/H / A/H), AP, Accuracy, Confusion matrix,
  and a Per-Question performance table.

IMPORTANT — which data is honest here:
  The final models are trained with --pool-final, i.e. on train+val+test
  pooled. Reporting them on the public val/test splits would therefore be
  LEAKAGE (they saw those videos). The only held-out data for these models is
  the 113-video seeded internal holdout that was excluded from every model's
  training. We report on THAT — it is the honest analogue of the val/test
  breakdown produced by the train-only pipeline.

Reported at the fixed submission threshold (0.5) and, for reference, at the
holdout-optimal threshold.

Usage:
    python report_results.py --device cuda:0
"""
import os
import json
import argparse

import numpy as np
import torch

import config
from evaluate import compute_metrics, print_metrics, per_question_analysis, optimize_threshold
from finalize_ensemble import reconstruct_holdout, latest_pool_final_checkpoint, infer


def q_nums(samples):
    return np.array([s["question_num"] for s in samples])


def report_block(title, y, probs, qn, thr):
    preds = (probs >= thr).astype(int)
    m = compute_metrics(y, preds, probs)
    print(f"\n{'='*60}\n  {title}  (threshold={thr:.3f})\n{'='*60}")
    print(f"  Positive predicted: {preds.sum()}/{len(preds)} ({preds.mean()*100:.1f}%)")
    print_metrics(m, prefix="  ")
    per_question_analysis(y, preds, qn)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--ensemble-config", type=str, default="outputs/final_ensemble_config.json")
    args = ap.parse_args()

    device = torch.device(args.device)
    holdout = reconstruct_holdout(seed=42)
    y = np.array([s["label"] for s in holdout])
    qn = q_nums(holdout)

    print(f"\n{'#'*60}")
    print(f"# FINAL MODELS — honest report on the {len(holdout)}-video internal holdout")
    print(f"# (held out of every pool-final model's training)")
    print(f"# Class 0 (No A/H): {int((y==0).sum())}  |  Class 1 (A/H): {int((y==1).sum())}")
    print(f"{'#'*60}")

    members = {
        "text": ("text", None),
        "visual": ("visual", None),
        "audio": ("audio", None),
        "fusion_v1": ("fusion", "v1"),
        "fusion_v2": ("fusion", "v2"),
    }

    probs = {}
    for name, (mtype, fver) in members.items():
        ckpt = latest_pool_final_checkpoint(mtype, fver)
        p, lbl = infer(ckpt, mtype, holdout, device)
        assert np.array_equal(lbl, y), f"{name}: holdout label mismatch"
        probs[name] = p
        thr_opt, _ = optimize_threshold(y, p)
        report_block(f"MEMBER: {name}", y, p, qn, 0.5)
        preds_opt = (p >= thr_opt).astype(int)
        from sklearn.metrics import f1_score as _f1
        print(f"  [ref] holdout-optimal thr={thr_opt:.3f} -> Macro-F1={_f1(y, preds_opt, average='macro'):.4f}")

    # Affect member (new pipeline)
    affect_ckpt = os.path.join(config.CHECKPOINT_DIR, "affect_final_seed42.pt")
    if os.path.isfile(affect_ckpt):
        from affect_infer import infer_affect
        probs["affect"] = infer_affect(affect_ckpt, holdout, device)
        report_block("MEMBER: affect (new pipeline)", y, probs["affect"], qn, 0.5)

    # Ensemble (weights + threshold from locked config)
    with open(args.ensemble_config) as f:
        cfg = json.load(f)
    w = cfg["weights"]
    ens = sum(w[k] * probs[k] for k in w)
    report_block("ENSEMBLE (AP-weighted, locked config)", y, ens, qn, cfg["threshold"])

    print(f"\n{'#'*60}")
    print("# SUMMARY (holdout, threshold 0.5)")
    print(f"{'#'*60}")
    from sklearn.metrics import f1_score, average_precision_score
    print(f"{'model':14s} {'MacroF1':>8s} {'F1(NoAH)':>9s} {'F1(AH)':>8s} {'AP':>7s}")
    for name in list(probs.keys()) + ["ENSEMBLE"]:
        p = ens if name == "ENSEMBLE" else probs[name]
        pr = (p >= 0.5).astype(int)
        print(f"{name:14s} {f1_score(y,pr,average='macro'):8.4f} "
              f"{f1_score(y,pr,pos_label=0):9.4f} {f1_score(y,pr,pos_label=1):8.4f} "
              f"{average_precision_score(y,p):7.4f}")


if __name__ == "__main__":
    main()

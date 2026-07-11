"""
Build the five official trial files from the saved per-member probabilities.

Reproduces submissions/trial-0.txt .. trial-4.txt from
outputs/predictions/private_test_submission_member_probs.npz
(written by predict_private_test.py).

  trial-0  locked 6-member AP-weighted ensemble, tau = 0.5   <- pre-registered
  trial-1  same probabilities, threshold matched to the labelled prior (54.5%)
  trial-2  AMF (affect) member alone, tau = 0.5
  trial-3  text member alone, tau = 0.5
  trial-4  AP-weighted rank-average of the 6 members, tau = 0.5

Usage:
  python make_all_trials.py --reference <organizers' with_probabilities/trial-0.txt>
"""
import os
import json
import argparse
import subprocess

import numpy as np
from scipy.stats import rankdata

PRIOR = 778 / 1427  # positive rate over all labelled BAH videos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", default="outputs/predictions/private_test_submission_member_probs.npz")
    ap.add_argument("--ensemble-probs", default="outputs/predictions/private_test_submission_probs.npy")
    ap.add_argument("--ensemble-config", default="outputs/final_ensemble_config.json")
    ap.add_argument("--split-file", default="../data/private/private_test.txt")
    ap.add_argument("--reference", required=True)
    ap.add_argument("--outdir", default="submissions")
    args = ap.parse_args()

    d = np.load(args.members, allow_pickle=True)
    cfg = json.load(open(args.ensemble_config))
    names = list(cfg["members"].keys())
    w = cfg["weights"]
    P = {n: d[n] for n in names}
    ens = np.load(args.ensemble_probs)
    n = len(ens)

    rank = np.stack([rankdata(P[k]) / n for k in names])
    rank_ens = (rank * np.array([w[k] for k in names])[:, None]).sum(0) / sum(w.values())

    thr_prior = float(np.quantile(ens, 1 - PRIOR))

    tmp = os.path.join(args.outdir, "_tmp_probs.npy")
    os.makedirs(args.outdir, exist_ok=True)

    trials = [
        (0, ens,           0.5),
        (1, ens,           thr_prior),
        (2, P["affect"],   0.5),
        (3, P["text"],     0.5),
        (4, rank_ens,      0.5),
    ]
    for i, probs, thr in trials:
        np.save(tmp, probs)
        subprocess.run([
            "python3", "make_submission.py",
            "--probs", tmp,
            "--split-file", args.split_file,
            "--reference", args.reference,
            "--threshold", str(thr),
            "--output", os.path.join(args.outdir, f"trial-{i}.txt"),
        ], check=True)
    os.remove(tmp)
    print(f"\nprior-matched threshold used for trial-1: {thr_prior:.4f}")


if __name__ == "__main__":
    main()

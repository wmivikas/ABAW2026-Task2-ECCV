"""
Generate an official challenge submission file (trial-<i>.txt).

Format (AH Challenge 3rd, ABAW11 / ECCV 2026), with probabilities:
    video_id,probability_of_class_0,probability_of_class_1,label_prediction
without probabilities:
    video_id,label_prediction

Hard requirements enforced by the organizers' validate_submission.py:
  * exactly the same number of lines as their reference trial-0.txt
  * the SAME video order as the reference file
  * p0, p1 in [0,1] and float(p0) + float(p1) == 1.0  (EXACT float equality)
  * label_prediction in {0,1}; no spaces anywhere in a line

We format probabilities to 4 decimals and derive p0 = 1 - p1 from the rounded
p1, which is verified to always satisfy the exact-sum check.

Usage:
  python make_submission.py \
      --probs outputs/predictions/private_test_submission_probs.npy \
      --split-file ../data/private/private_test.txt \
      --reference <path to organizers' with_probabilities/trial-0.txt> \
      --threshold 0.5 --output submissions/trial-0.txt
"""
import os
import argparse

import numpy as np

from dataset import parse_split_file


def fmt_probs(p1: float):
    """4-decimal p1 and p0 = 1 - p1 such that float(p0) + float(p1) == 1.0."""
    p1 = min(max(float(p1), 0.0), 1.0)
    s1 = f"{p1:.4f}"
    s0 = f"{1.0 - float(s1):.4f}"
    assert float(s0) + float(s1) == 1.0, (s0, s1)
    return s0, s1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", required=True, help=".npy of P(class 1) aligned with --split-file")
    ap.add_argument("--split-file", required=True, help="private test split used to produce --probs")
    ap.add_argument("--reference", default=None,
                    help="organizers' reference trial-0.txt (enforces video order)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--no-probabilities", action="store_true",
                    help="emit 'video_id,label' instead of 'video_id,p0,p1,label'")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    probs = np.load(args.probs)
    samples = parse_split_file(args.split_file)
    assert len(probs) == len(samples), f"probs {len(probs)} != samples {len(samples)}"
    by_id = {s["video_path"]: float(p) for s, p in zip(samples, probs)}

    # Determine output order: reference file order if given, else split order
    if args.reference:
        with open(args.reference) as f:
            order = [ln.strip().split(",")[0] for ln in f if ln.strip()]
        missing = [v for v in order if v not in by_id]
        assert not missing, f"{len(missing)} reference videos missing from predictions, e.g. {missing[:3]}"
        extra = [v for v in by_id if v not in set(order)]
        if extra:
            print(f"  note: {len(extra)} predicted videos not in reference (ignored)")
    else:
        order = [s["video_path"] for s in samples]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    n_pos = 0
    with open(args.output, "w") as f:
        for vid in order:
            p1 = by_id[vid]
            pred = int(p1 >= args.threshold)
            n_pos += pred
            if args.no_probabilities:
                f.write(f"{vid},{pred}\n")
            else:
                s0, s1 = fmt_probs(p1)
                f.write(f"{vid},{s0},{s1},{pred}\n")

    print(f"Wrote {args.output}: {len(order)} lines, "
          f"positives {n_pos} ({n_pos/len(order)*100:.1f}%), threshold {args.threshold}")


if __name__ == "__main__":
    main()

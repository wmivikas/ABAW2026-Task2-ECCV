"""
Single source of truth for the pool-final internal holdout.

Every --pool-final model must early-stop on the SAME holdout, and
finalize_ensemble.py must score members on that same holdout, otherwise a
member trained on those videos gets an inflated AP and an inflated ensemble
weight.

This previously drifted: train.py used random.Random on lists while
train_affect.py used np.random.RandomState on index arrays, producing two
113-video holdouts overlapping on only 7 videos. AMF therefore trained on 106
of the 113 videos it was later scored on.

Use split_pool_final() everywhere.
"""
import random
from typing import Dict, List, Tuple


def split_pool_final(pool: List[Dict], seed: int = 42, frac: float = 0.08
                     ) -> Tuple[List[Dict], List[Dict]]:
    """Stratified train/holdout split of the pooled labelled videos.

    Returns (train_samples, holdout_samples). Deterministic given `seed` and the
    order of `pool`, which must be train + val + test concatenated in that order.
    """
    rng = random.Random(seed)
    pos = [s for s in pool if s["label"] == 1]
    neg = [s for s in pool if s["label"] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    n_hold_pos = max(1, int(len(pos) * frac))
    n_hold_neg = max(1, int(len(neg) * frac))
    holdout = pos[:n_hold_pos] + neg[:n_hold_neg]
    train = pos[n_hold_pos:] + neg[n_hold_neg:]
    return train, holdout

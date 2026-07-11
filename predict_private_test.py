"""
Generate the final private-test submission.

End-to-end: takes a private-test split file, extracts whatever features are
missing, runs all 5 locked final models, and writes a submission CSV using
the ensemble weights fixed in outputs/final_ensemble_config.json.

The 5 models (text, visual, audio, fusion-OCCF, fusion-attn-gate) were each
trained on ALL 1427 labeled BAH videos (train+val+test pooled), with an
automatic 8% stratified internal holdout used only for early stopping — see
train.py --pool-final. Ensemble weights were fit on that same 113-video
holdout (finalize_ensemble.py) and are fixed here, not re-tuned.

Private test split file format (one line per video), matching the existing
train/val/test split files:
    video_path,label,transcript
The label column is required by the file parser but unused for prediction —
put 0 as a placeholder if the private test is unlabeled.

Usage:
    python predict_private_test.py --split-file /path/to/private_test.txt \
        --device cuda:0 --output outputs/predictions/private_test_submission.csv
"""
import os
import json
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from dataset import BAHDataset, parse_split_file
from predict import load_model, predict as predict_raw
import extract_audio
import extract_features


def ensure_features(split_file: str, device: str):
    """Extract audio + text/visual/audio features for any videos not yet cached."""
    samples = parse_split_file(split_file)

    missing_audio = any(
        not os.path.isfile(os.path.join(
            config.AUDIO_DIR, s["participant_id"], "Visite_1",
            s["basename"].replace(".mp4", ".wav")))
        for s in samples
    )
    if missing_audio:
        print("[ensure_features] Extracting audio (.wav) for private test videos ...")
        import sys
        sys.argv = ["extract_audio.py", "--split-file", split_file]
        extract_audio.main()

    missing_feat = {
        mod: any(
            not os.path.isfile(os.path.join(config.FEATURES_DIR, mod,
                                            s["basename"].replace(".mp4", ".pt")))
            for s in samples
        )
        for mod in ["text", "visual", "audio"]
    }
    for mod, missing in missing_feat.items():
        if missing:
            print(f"[ensure_features] Extracting {mod} features for private test videos ...")
            fn = {"text": extract_features.extract_text_features,
                  "visual": extract_features.extract_visual_features,
                  "audio": extract_features.extract_audio_features}[mod]
            kwargs = {"device": device, "split_file": split_file}
            if mod == "visual":
                kwargs["batch_size"] = 1
            elif mod == "audio":
                kwargs["batch_size"] = 1
            fn(**kwargs)

    # Affect-specialised features for the new pipeline member
    import extract_affect_features
    aff_missing = any(
        not os.path.isfile(os.path.join(config.FEATURES_DIR, "affect", "audio",
                                        s["basename"].replace(".mp4", ".pt")))
        for s in samples
    )
    if aff_missing:
        print("[ensure_features] Extracting affect features (FER + emotion-audio + markers) ...")
        extract_affect_features.extract_text_markers(split_file)
        extract_affect_features.extract_visual(device, split_file)
        extract_affect_features.extract_audio(device, split_file)

    # The AMF marker vector is 11 text markers + 14 FER statistics = 25 dims.


def infer_member(checkpoint: str, model_type: str, split_file: str, device):
    if model_type == "affect":
        from affect_infer import infer_affect
        samples = parse_split_file(split_file)
        return infer_affect(checkpoint, samples, device), samples
    model, tokenizer, _ = load_model(checkpoint, model_type, device)
    modalities = [model_type] if model_type != "fusion" else ["text", "visual", "audio"]
    ds = BAHDataset(
        split="test", split_file=split_file,
        modalities=modalities,
        text_tokenizer=tokenizer,
        use_precomputed_features=(model_type == "fusion"),
    )
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4)
    probs, _, _ = predict_raw(model, loader, device, model_type)
    return probs, ds.samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-file", type=str, required=True,
                    help="Private test split file (video_path,label,transcript)")
    ap.add_argument("--ensemble-config", type=str,
                    default="outputs/final_ensemble_config.json")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--output", type=str,
                    default="outputs/predictions/private_test_submission.csv")
    args = ap.parse_args()

    device = torch.device(args.device)

    with open(args.ensemble_config) as f:
        cfg = json.load(f)
    print(f"Loaded ensemble config: {len(cfg['members'])} members, "
          f"threshold={cfg['threshold']}, held-out F1 was {cfg['holdout_macro_f1']:.4f}")

    print("\n=== Step 1: ensure features are extracted ===")
    ensure_features(args.split_file, args.device)

    print("\n=== Step 2: run each final model ===")
    all_probs = {}
    samples_ref = None
    for name, m in cfg["members"].items():
        print(f"  Running {name} ({m['model_type']}) ...")
        probs, samples = infer_member(m["checkpoint"], m["model_type"], args.split_file, device)
        all_probs[name] = probs
        samples_ref = samples  # same order for every member (same split file)

    print("\n=== Step 3: weighted ensemble ===")
    weights = cfg["weights"]
    ens = sum(weights[k] * all_probs[k] for k in weights)
    thr = cfg["threshold"]
    preds = (ens >= thr).astype(int)
    print(f"  Positive rate: {preds.mean()*100:.1f}% ({preds.sum()}/{len(preds)})")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        for s, p in zip(samples_ref, preds):
            f.write(f"{s['video_path']},{int(p)}\n")
    np.save(args.output.replace(".csv", "_probs.npy"), ens)

    # Per-member probabilities, so alternative trials can be composed without
    # re-running inference.
    np.savez(args.output.replace(".csv", "_member_probs.npz"),
             video_path=np.array([s["video_path"] for s in samples_ref]),
             **all_probs)
    print(f"\nSaved submission to {args.output}")


if __name__ == "__main__":
    main()

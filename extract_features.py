"""
Pre-extract features from all modalities for fast fusion model training.

Encoder choices (matching ConflictAwareAH SOTA):
- Text: SamLowe/roberta-base-go_emotions (768-dim, emotion-specific)
- Visual: MCG-NJU/videomae-base (768-dim, temporal 16-frame)
- Audio: facebook/hubert-base-ls960 (768-dim, speech-tuned)

Usage:
    python extract_features.py --modality text --device cuda:0
    python extract_features.py --modality visual --device cuda:0
    python extract_features.py --modality audio --device cuda:0
"""
import os
import argparse
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from dataset import BAHDataset


def extract_text_features(device: str = "cuda:0", batch_size: int = 32, split_file: str = None):
    """Extract RoBERTa-GoEmotions features for all transcripts."""
    from transformers import AutoModel, AutoTokenizer

    model_name = "SamLowe/roberta-base-go_emotions"
    print(f"Loading text model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    output_dir = os.path.join(config.FEATURES_DIR, "text")
    os.makedirs(output_dir, exist_ok=True)

    splits = ["custom"] if split_file else ["train", "val", "test"]
    for split in splits:
        print(f"\nProcessing {split} split...")
        dataset = BAHDataset(
            split=split,
            split_file=split_file,
            modalities=["text"],
            text_tokenizer=tokenizer,
            text_max_length=512,
            use_question_prompt=True,
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        all_features = []

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"  {split}"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                cls_features = outputs.last_hidden_state[:, 0, :]  # (batch, 768)
                all_features.append(cls_features.cpu())

        all_features = torch.cat(all_features, dim=0)

        # Save per-sample
        for i, sample in enumerate(dataset.samples):
            video_name = sample["basename"].replace(".mp4", ".pt")
            torch.save(all_features[i], os.path.join(output_dir, video_name))

        print(f"  Saved {len(dataset)} features (dim={all_features.shape[1]}) to {output_dir}")

    del model
    torch.cuda.empty_cache()


def extract_visual_features(device: str = "cuda:0", batch_size: int = 4, num_frames: int = 16,
                            split_file: str = None):
    """Extract VideoMAE-Base features from face frame sequences (temporal)."""
    from transformers import VideoMAEModel, VideoMAEImageProcessor

    model_name = "MCG-NJU/videomae-base"
    print(f"Loading visual model: {model_name}")

    processor = VideoMAEImageProcessor.from_pretrained(model_name)
    model = VideoMAEModel.from_pretrained(model_name).to(device).eval()

    output_dir = os.path.join(config.FEATURES_DIR, "visual")
    os.makedirs(output_dir, exist_ok=True)

    splits = ["custom"] if split_file else ["train", "val", "test"]
    for split in splits:
        print(f"\nProcessing {split} split...")
        dataset = BAHDataset(
            split=split,
            split_file=split_file,
            modalities=["visual"],
            num_frames=num_frames,
        )
        # Process one at a time for visual (large memory footprint)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"  {split}"):
                frames = batch["frames"]  # (1, num_frames, 3, H, W)
                idx = batch["idx"].item()

                video_name = dataset.samples[idx]["basename"].replace(".mp4", ".pt")
                out_path = os.path.join(output_dir, video_name)

                if os.path.isfile(out_path):
                    continue

                # VideoMAE expects (batch, num_frames, channels, height, width)
                # Our frames are already in that shape
                frames_np = frames.squeeze(0).numpy()  # (num_frames, 3, H, W)

                # VideoMAE processor expects list of frames as (H, W, C) numpy arrays
                frames_list = [f.transpose(1, 2, 0) for f in frames_np]  # list of (H, W, 3)

                # Process with VideoMAE processor
                inputs = processor(
                    list(frames_list),
                    return_tensors="pt",
                )
                pixel_values = inputs["pixel_values"].to(device)  # (1, num_frames, C, H, W)

                outputs = model(pixel_values=pixel_values)
                # VideoMAE outputs last_hidden_state: (batch, num_patches, 768)
                # Mean pool over all patch tokens to get video-level feature
                pooled = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu()  # (768,)

                torch.save(pooled, out_path)

        print(f"  Saved features to {output_dir}")

    del model
    torch.cuda.empty_cache()


def extract_audio_features(device: str = "cuda:0", batch_size: int = 2, split_file: str = None):
    """Extract HuBERT-Large features from audio."""
    from transformers import HubertModel

    model_name = "facebook/hubert-large-ll60k"
    print(f"Loading audio model: {model_name}")

    model = HubertModel.from_pretrained(model_name).to(device).eval()

    output_dir = os.path.join(config.FEATURES_DIR, "audio")
    os.makedirs(output_dir, exist_ok=True)

    splits = ["custom"] if split_file else ["train", "val", "test"]
    for split in splits:
        print(f"\nProcessing {split} split...")
        dataset = BAHDataset(
            split=split,
            split_file=split_file,
            modalities=["audio"],
            max_audio_duration=30.0,
            audio_sample_rate=16000,
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"  {split}"):
                idx = batch["idx"].item()
                video_name = dataset.samples[idx]["basename"].replace(".mp4", ".pt")
                out_path = os.path.join(output_dir, video_name)

                if os.path.isfile(out_path):
                    continue

                waveform = batch["waveform"].to(device)  # (1, num_samples)

                try:
                    outputs = model(waveform)
                    hidden = outputs.last_hidden_state  # (1, time_steps, 1024)
                    pooled = hidden.mean(dim=1).squeeze(0).cpu()  # (1024,)
                    torch.save(pooled, out_path)
                except Exception as e:
                    # Save zeros for failed extractions
                    print(f"  Warning: Failed on {video_name}: {e}")
                    torch.save(torch.zeros(1024), out_path)

        print(f"  Saved features to {output_dir}")

    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", type=str, required=True,
                        choices=["text", "visual", "audio", "all"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--split-file", type=str, default=None,
                        help="Path to a split file (video_path,label,transcript) to "
                             "process instead of the default train/val/test splits — "
                             "e.g. the private test split.")
    args = parser.parse_args()

    if args.modality == "text" or args.modality == "all":
        extract_text_features(args.device, args.batch_size, split_file=args.split_file)

    if args.modality == "visual" or args.modality == "all":
        extract_visual_features(args.device, batch_size=1, num_frames=args.num_frames,
                                split_file=args.split_file)

    if args.modality == "audio" or args.modality == "all":
        extract_audio_features(args.device, batch_size=1, split_file=args.split_file)

    print("\n✓ Feature extraction complete!")


if __name__ == "__main__":
    main()

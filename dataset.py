"""
Dataset and DataLoader for BAH A/H Video Recognition.

Handles loading of all 3 modalities (visual, audio, text) from the BAH dataset.
Split files format: video_path,label,transcript
"""
import os
import re
import csv
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import config
import augmentation


def parse_split_file(split_path: str) -> List[Dict]:
    """
    Parse a BAH split file.
    Format: video_path,label,transcript_text
    
    The transcript is the 3rd field onwards (may contain commas).
    """
    samples = []
    with open(split_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Split only on first two commas — transcript may contain commas
            parts = line.split(",", 2)
            if len(parts) < 2:
                continue
            
            video_path = parts[0].strip()
            label = int(parts[1].strip())
            transcript = parts[2].strip() if len(parts) > 2 else ""
            
            # Extract participant ID and question number from path
            # Format: Videos/82694/Visite_1/82694_Question_1_2024-11-15_21-05-54_Video.mp4
            basename = os.path.basename(video_path)
            match = re.search(r"(\d+)_Question_(\d+)_", basename)
            if match:
                participant_id = match.group(1)
                question_num = int(match.group(2))
            else:
                participant_id = "unknown"
                question_num = 0
            
            samples.append({
                "video_path": video_path,
                "label": label,
                "transcript": transcript,
                "participant_id": participant_id,
                "question_num": question_num,
                "basename": basename,
            })
    
    return samples


class BAHDataset(Dataset):
    """
    BAH Dataset for A/H recognition.
    
    Supports loading:
    - Text transcripts (always available)
    - Visual face frames (from cropped-aligned-faces)
    - Audio (from extracted wav files)
    - Pre-extracted features (if available)
    """
    
    def __init__(
        self,
        split: str = "train",
        modalities: List[str] = None,
        num_frames: int = 16,
        image_size: int = 224,
        max_audio_duration: float = 30.0,
        audio_sample_rate: int = 16000,
        text_tokenizer=None,
        text_max_length: int = 512,
        use_question_prompt: bool = True,
        transform=None,
        use_precomputed_features: bool = False,
        split_file: str = None,
    ):
        """
        Args:
            split: "train", "val", or "test" (ignored if split_file is given)
            modalities: list of modalities to load, e.g., ["text", "visual", "audio"]
            num_frames: number of face frames to sample per video
            image_size: resize face frames to this size
            max_audio_duration: max audio duration in seconds
            audio_sample_rate: target sample rate for audio
            text_tokenizer: HuggingFace tokenizer for text
            text_max_length: max token length for text
            use_question_prompt: whether to prepend question context to text
            transform: image transforms
            use_precomputed_features: load pre-extracted features instead of raw data
            split_file: optional explicit path to a split file (e.g. the private
                test split), overriding the default train/val/test file for `split`
        """
        if modalities is None:
            modalities = ["text"]
        
        self.split = split
        self.modalities = modalities
        self.num_frames = num_frames
        self.image_size = image_size
        self.max_audio_duration = max_audio_duration
        self.audio_sample_rate = audio_sample_rate
        self.text_tokenizer = text_tokenizer
        self.text_max_length = text_max_length
        self.use_question_prompt = use_question_prompt
        self.transform = transform
        self.use_precomputed_features = use_precomputed_features

        # Load split file
        if split_file is not None:
            self.samples = parse_split_file(split_file)
        else:
            split_map = {"train": config.TRAIN_SPLIT, "val": config.VAL_SPLIT, "test": config.TEST_SPLIT}
            self.samples = parse_split_file(split_map[split])
        
        print(f"[BAHDataset] Loaded {len(self.samples)} samples from {split} split")
        
        # Count class distribution
        labels = [s["label"] for s in self.samples]
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos
        print(f"  Class 0 (No A/H): {n_neg} ({n_neg/len(labels)*100:.1f}%)")
        print(f"  Class 1 (A/H):    {n_pos} ({n_pos/len(labels)*100:.1f}%)")
    
    def __len__(self):
        return len(self.samples)
    
    def _get_text(self, sample: Dict) -> str:
        """Get text input, optionally with question context prompt."""
        transcript = sample["transcript"]
        
        if self.use_question_prompt and sample["question_num"] in config.QUESTION_INFO:
            q_info = config.QUESTION_INFO[sample["question_num"]]
            # Add question context to help the model understand the elicitation context
            prompt = (
                f"Question type: {q_info['response']}. "
                f"Question: {q_info['prompt']}. "
                f"Response: {transcript}"
            )
            text = prompt
        else:
            text = transcript
            
        if self.split == "train" and "text" in self.modalities and not self.use_precomputed_features:
            text = augmentation.apply_text_augmentations(text)
            
        return text
    
    def _tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize text using the provided tokenizer."""
        if self.text_tokenizer is None:
            raise ValueError("text_tokenizer must be provided for text modality")
        
        encoding = self.text_tokenizer(
            text,
            max_length=self.text_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}
    
    def _load_face_frames(self, sample: Dict) -> torch.Tensor:
        """Load and sample face frames from cropped-aligned-faces directory."""
        video_name = sample["basename"]
        participant_id = sample["participant_id"]
        
        face_dir = os.path.join(
            config.FACES_DIR, participant_id, "Visite_1", video_name
        )
        
        if not os.path.isdir(face_dir):
            # Return zeros if face directory not found
            return torch.zeros(self.num_frames, 3, self.image_size, self.image_size)
        
        # List all frame files and sort numerically
        frame_files = sorted(
            [f for f in os.listdir(face_dir) if f.endswith(".jpg")],
            key=lambda x: int(re.search(r"(\d+)", x).group()) if re.search(r"(\d+)", x) else 0,
        )
        
        if len(frame_files) == 0:
            return torch.zeros(self.num_frames, 3, self.image_size, self.image_size)
        
        # Uniformly sample num_frames
        if len(frame_files) >= self.num_frames:
            indices = np.linspace(0, len(frame_files) - 1, self.num_frames, dtype=int)
        else:
            # Pad by repeating last frame
            indices = list(range(len(frame_files)))
            indices += [len(frame_files) - 1] * (self.num_frames - len(frame_files))
        
        frames = []
        for idx in indices:
            frame_path = os.path.join(face_dir, frame_files[idx])
            try:
                img = Image.open(frame_path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                else:
                    # Default resize and normalize
                    img = img.resize((self.image_size, self.image_size))
                    img = torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255.0
                    # ImageNet normalization
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    img = (img - mean) / std
                frames.append(img)
            except Exception:
                frames.append(torch.zeros(3, self.image_size, self.image_size))
        
        return torch.stack(frames)  # (num_frames, 3, H, W)
    
    def _load_audio(self, sample: Dict) -> torch.Tensor:
        """Load pre-extracted audio waveform."""
        import torchaudio
        
        video_name = sample["basename"]
        audio_name = video_name.replace(".mp4", ".wav")
        participant_id = sample["participant_id"]
        
        audio_path = os.path.join(
            config.AUDIO_DIR, participant_id, "Visite_1", audio_name
        )
        
        max_samples = int(self.max_audio_duration * self.audio_sample_rate)
        
        if not os.path.isfile(audio_path):
            return torch.zeros(max_samples)
        
        try:
            waveform, sr = torchaudio.load(audio_path)
            # Resample if needed
            if sr != self.audio_sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.audio_sample_rate)
                waveform = resampler(waveform)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            waveform = waveform.squeeze(0)
            
            # Pad or truncate to max_duration
            if waveform.shape[0] > max_samples:
                waveform = waveform[:max_samples]
            elif waveform.shape[0] < max_samples:
                padding = torch.zeros(max_samples - waveform.shape[0])
                waveform = torch.cat([waveform, padding])
            
            if self.split == "train":
                waveform = augmentation.apply_audio_augmentations(waveform, self.audio_sample_rate)
                
            return waveform
        except Exception:
            return torch.zeros(max_samples)
    
    def _load_precomputed_features(self, sample: Dict, modality: str) -> torch.Tensor:
        """Load pre-extracted features for a given modality."""
        video_name = sample["basename"].replace(".mp4", ".pt")
        feat_path = os.path.join(config.FEATURES_DIR, modality, video_name)
        
        if os.path.isfile(feat_path):
            return torch.load(feat_path, weights_only=True)
        else:
            # Return zeros with expected dimension
            dim_map = {"visual": 768, "audio": 1024, "text": 768}
            return torch.zeros(dim_map.get(modality, 768))
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        output = {
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "question_num": torch.tensor(sample["question_num"], dtype=torch.long),
            "idx": torch.tensor(idx, dtype=torch.long),
        }
        
        if self.use_precomputed_features:
            # Load pre-extracted features
            for mod in self.modalities:
                output[f"{mod}_features"] = self._load_precomputed_features(sample, mod)
        else:
            # Load raw data
            if "text" in self.modalities:
                text = self._get_text(sample)
                if self.text_tokenizer is not None:
                    token_output = self._tokenize_text(text)
                    output.update(token_output)
                else:
                    output["text"] = text
            
            if "visual" in self.modalities:
                output["frames"] = self._load_face_frames(sample)
            
            if "audio" in self.modalities:
                output["waveform"] = self._load_audio(sample)
        
        return output


def get_dataloader(
    split: str,
    modalities: List[str] = None,
    batch_size: int = 16,
    num_workers: int = 4,
    text_tokenizer=None,
    **kwargs,
) -> DataLoader:
    """Create a DataLoader for the given split."""
    
    # Setup visual transforms based on split
    if "transform" not in kwargs or kwargs["transform"] is None:
        if split == "train":
            kwargs["transform"] = augmentation.get_visual_train_transforms()
        else:
            kwargs["transform"] = augmentation.get_visual_val_transforms()

    dataset = BAHDataset(
        split=split,
        modalities=modalities,
        text_tokenizer=text_tokenizer,
        **kwargs,
    )
    
    shuffle = (split == "train")
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )


# ============================================================
# Test / sanity check
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("BAH Dataset — Sanity Check")
    print("=" * 60)
    
    # Test text-only loading
    print("\n--- Text-only mode ---")
    ds = BAHDataset(split="train", modalities=["text"])
    sample = ds[0]
    print(f"  Sample keys: {list(sample.keys())}")
    print(f"  Label: {sample['label'].item()}")
    print(f"  Question: {sample['question_num'].item()}")
    print(f"  Text (first 200 chars): {sample['text'][:200]}...")
    
    # Test visual loading
    print("\n--- Visual mode ---")
    ds_vis = BAHDataset(split="train", modalities=["visual"], num_frames=4)
    sample_vis = ds_vis[0]
    print(f"  Frames shape: {sample_vis['frames'].shape}")
    
    # Count per-question distribution
    print("\n--- Question distribution (train) ---")
    from collections import Counter
    q_counts = Counter()
    q_label_counts = Counter()
    for s in ds.samples:
        q_counts[s["question_num"]] += 1
        q_label_counts[(s["question_num"], s["label"])] += 1
    
    for q in sorted(q_counts.keys()):
        n0 = q_label_counts.get((q, 0), 0)
        n1 = q_label_counts.get((q, 1), 0)
        total = q_counts[q]
        pct1 = n1 / total * 100 if total > 0 else 0
        q_name = config.QUESTION_INFO.get(q, {}).get("response", "?")
        print(f"  Q{q} ({q_name:>10s}): {total:>4d} total | 0: {n0:>3d} | 1: {n1:>3d} ({pct1:.0f}% A/H)")
    
    print("\n✓ Dataset sanity check passed!")

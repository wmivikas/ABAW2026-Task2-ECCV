"""
Extract audio (.wav) from all BAH videos.
Uses torchaudio (with ffmpeg backend) or subprocess ffmpeg.
Run this before training audio-based models.
"""
import os
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import config
from dataset import parse_split_file


def extract_audio_torchaudio(video_path: str, output_path: str, sample_rate: int = 16000) -> bool:
    """Extract audio using torchaudio (works even without ffmpeg CLI if torchaudio has backend)."""
    if os.path.isfile(output_path):
        return True
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        import torchaudio
        # torchaudio can load mp4 directly with sox/ffmpeg backend
        waveform, sr = torchaudio.load(video_path)
        
        # Resample if needed
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Save as wav
        torchaudio.save(output_path, waveform, sample_rate)
        return True
    except Exception:
        return False


def extract_audio_ffmpeg(video_path: str, output_path: str, sample_rate: int = 16000) -> bool:
    """Extract audio from a video file using ffmpeg CLI."""
    if os.path.isfile(output_path):
        return True
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        output_path,
    ]
    
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        return result.returncode == 0
    except Exception:
        return False


def check_ffmpeg_available() -> bool:
    """Check if ffmpeg CLI is available."""
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        return result.returncode == 0
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-file", type=str, default=None,
                        help="Path to a split file (video_path,label,transcript) to "
                             "process instead of the default train/val/test splits — "
                             "e.g. the private test split.")
    args = parser.parse_args()

    print("=" * 60)
    print("  Audio Extraction from BAH Videos")
    print("=" * 60)

    # Check extraction method
    use_ffmpeg_cli = check_ffmpeg_available()
    if use_ffmpeg_cli:
        print("Using: ffmpeg CLI")
        extract_fn = extract_audio_ffmpeg
    else:
        print("Using: torchaudio (ffmpeg CLI not found)")
        extract_fn = extract_audio_torchaudio

    # Collect video paths: either the given split file, or all default splits
    all_samples = []
    if args.split_file:
        all_samples.extend(parse_split_file(args.split_file))
    else:
        for split_path in [config.TRAIN_SPLIT, config.VAL_SPLIT, config.TEST_SPLIT]:
            all_samples.extend(parse_split_file(split_path))
    
    print(f"Total videos to process: {len(all_samples)}")
    
    # Build extraction tasks
    tasks = []
    for sample in all_samples:
        video_rel = sample["video_path"]  # e.g. Videos/82694/Visite_1/82694_Question_1_...mp4
        video_abs = os.path.join(config.DATA_ROOT, video_rel)
        
        # Output: data/audio/<participant_id>/Visite_1/<basename>.wav
        participant_id = sample["participant_id"]
        audio_name = sample["basename"].replace(".mp4", ".wav")
        audio_abs = os.path.join(config.AUDIO_DIR, participant_id, "Visite_1", audio_name)
        
        if not os.path.isfile(video_abs):
            continue
        
        tasks.append((video_abs, audio_abs))
    
    # Check how many already exist
    existing = sum(1 for _, audio_path in tasks if os.path.isfile(audio_path))
    remaining = [(v, a) for v, a in tasks if not os.path.isfile(a)]
    print(f"Found video files: {len(tasks)}")
    print(f"Already extracted: {existing}")
    print(f"Remaining: {len(remaining)}")
    
    if not remaining:
        print("All audio files already extracted!")
        return
    
    # Extract sequentially (torchaudio is not always thread-safe)
    success = 0
    failed = 0
    failed_list = []
    
    for video_path, audio_path in tqdm(remaining, desc="Extracting audio"):
        if extract_fn(video_path, audio_path):
            success += 1
        else:
            failed += 1
            if failed <= 5:
                failed_list.append(video_path)
    
    print(f"\n✓ Extraction complete: {success} success, {failed} failed")
    if failed_list:
        print("  First failed files:")
        for f in failed_list:
            print(f"    {f}")


if __name__ == "__main__":
    main()

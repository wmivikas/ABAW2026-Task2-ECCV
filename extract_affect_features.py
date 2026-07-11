"""
Affect-specialised feature extraction (the NEW pipeline's front-end).

Unlike prior BAH work — which uses generic backbones (VideoMAE = Kinetics
actions, HuBERT = generic speech) that leave the visual/audio channels near
chance — we extract features that are *specialised for affect*, plus explicit
interpretable hesitation markers:

  VISUAL : per-frame facial-expression distribution from a FER ViT
           (angry/disgust/fear/happy/neutral/sad/surprise), aggregated over
           the clip into mean+std (14 interpretable dims) AND a pooled ViT
           embedding (768). Captures facial affect, not body motion.

  AUDIO  : emotion-fine-tuned wav2vec2 (audeering MSP-dim) pooled hidden
           state (1024). Encodes prosodic emotion/arousal — the vocal
           correlates of hesitancy (tremor, pauses, flat affect).

  TEXT   : interpretable hesitation-marker profile from the transcript
           (hedges, filled pauses, contrastive/negation markers, first-person
           uncertainty, repetition, lexical polarity oscillation) -> a compact
           vector of psycholinguistic hesitancy cues. (The dense RoBERTa-
           GoEmotions text embedding is reused from the existing text features.)

Outputs go to data/features/affect/{visual,audio,text_markers}/<basename>.pt
Each file also stores the interpretable sub-vector for later explanation.

Usage:
    python extract_affect_features.py --modality all --device cuda:0
    python extract_affect_features.py --modality all --split-file <private_test.txt>
"""
import os
import re
import argparse

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import config
from dataset import parse_split_file

FER_MODEL = "trpakov/vit-face-expression"
AUDIO_MODEL = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
FER_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

AFFECT_DIR = os.path.join(config.FEATURES_DIR, "affect")


# ---------------------------------------------------------------- helpers
def _samples(split_file):
    if split_file:
        return parse_split_file(split_file)
    out = []
    for p in [config.TRAIN_SPLIT, config.VAL_SPLIT, config.TEST_SPLIT]:
        out.extend(parse_split_file(p))
    return out


def _face_paths(sample, num_frames=16):
    face_dir = os.path.join(config.FACES_DIR, sample["participant_id"], "Visite_1", sample["basename"])
    if not os.path.isdir(face_dir):
        return []
    frames = sorted([f for f in os.listdir(face_dir) if f.endswith(".jpg")],
                    key=lambda x: int(re.search(r"(\d+)", x).group()) if re.search(r"(\d+)", x) else 0)
    if not frames:
        return []
    idx = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
    return [os.path.join(face_dir, frames[i]) for i in idx]


# ---------------------------------------------------------------- visual (FER)
def extract_visual(device, split_file=None, num_frames=16):
    from transformers import AutoModelForImageClassification, AutoImageProcessor
    proc = AutoImageProcessor.from_pretrained(FER_MODEL)
    model = AutoModelForImageClassification.from_pretrained(FER_MODEL).to(device).eval()
    out_dir = os.path.join(AFFECT_DIR, "visual"); os.makedirs(out_dir, exist_ok=True)

    for s in tqdm(_samples(split_file), desc="visual-FER"):
        out_path = os.path.join(out_dir, s["basename"].replace(".mp4", ".pt"))
        if os.path.isfile(out_path):
            continue
        paths = _face_paths(s, num_frames)
        if not paths:
            torch.save({"fer_stats": torch.zeros(2 * len(FER_CLASSES)),
                        "embedding": torch.zeros(768)}, out_path)
            continue
        imgs = [Image.open(p).convert("RGB") for p in paths]
        with torch.no_grad():
            inp = proc(images=imgs, return_tensors="pt").to(device)
            out = model(**inp, output_hidden_states=True)
            probs = torch.softmax(out.logits, dim=-1)                 # (F,7)
            cls_emb = out.hidden_states[-1][:, 0, :]                  # (F,768) ViT CLS
        fer_stats = torch.cat([probs.mean(0), probs.std(0)]).cpu()    # (14,) interpretable
        emb = cls_emb.mean(0).cpu()                                   # (768,)
        torch.save({"fer_stats": fer_stats, "embedding": emb}, out_path)
    del model; torch.cuda.empty_cache()


# ---------------------------------------------------------------- audio (emotion)
def extract_audio(device, split_file=None, sr=16000, max_s=30.0):
    import soundfile as sf
    from transformers import AutoModel, AutoFeatureExtractor
    fe = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL)
    model = AutoModel.from_pretrained(AUDIO_MODEL).to(device).eval()
    out_dir = os.path.join(AFFECT_DIR, "audio"); os.makedirs(out_dir, exist_ok=True)

    for s in tqdm(_samples(split_file), desc="audio-emotion"):
        out_path = os.path.join(out_dir, s["basename"].replace(".mp4", ".pt"))
        if os.path.isfile(out_path):
            continue
        wav_path = os.path.join(config.AUDIO_DIR, s["participant_id"], "Visite_1",
                                s["basename"].replace(".mp4", ".wav"))
        if not os.path.isfile(wav_path):
            torch.save({"embedding": torch.zeros(1024)}, out_path)
            continue
        wav, in_sr = sf.read(wav_path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(1)
        if in_sr != sr and len(wav) > 0:
            # linear resample (avoid torchaudio; good enough for pooled features)
            n = int(round(len(wav) * sr / in_sr))
            wav = np.interp(np.linspace(0, len(wav) - 1, n), np.arange(len(wav)), wav).astype(np.float32)
        wav = wav[: int(max_s * sr)] if len(wav) else np.zeros(sr, dtype=np.float32)
        with torch.no_grad():
            inp = fe(wav, sampling_rate=sr, return_tensors="pt").to(device)
            hidden = model(**inp).last_hidden_state           # (1,T,1024)
            emb = hidden.mean(1).squeeze(0).cpu()             # (1024,)
        torch.save({"embedding": emb}, out_path)
    del model; torch.cuda.empty_cache()


# ---------------------------------------------------------------- text markers
HEDGES = ["maybe", "perhaps", "kind of", "sort of", "i think", "i guess", "probably",
          "might", "could", "possibly", "not sure", "i don't know", "i dont know",
          "i suppose", "somewhat", "a bit", "i mean"]
FILLERS = ["um", "uh", "er", "erm", "hmm", "uhh", "umm", "eh", "like"]
CONTRAST = ["but", "however", "although", "though", "on the other hand", "yet",
            "whereas", "still", "even though"]
NEGATION = ["not", "no", "never", "don't", "dont", "can't", "cant", "won't", "wouldn't",
            "shouldn't", "nothing", "neither", "nor"]
UNCERTAIN_1P = ["i think", "i feel", "i'm not sure", "im not sure", "i guess", "i don't know",
                "i believe", "i suppose", "for me"]
POS_LEX = ["good", "great", "love", "like", "happy", "enjoy", "nice", "want", "willing",
           "yes", "definitely", "sure", "positive", "fun"]
NEG_LEX = ["bad", "hate", "dislike", "difficult", "hard", "afraid", "worry", "worried",
           "stress", "anxious", "no", "problem", "guilt", "guilty", "avoid", "reluctant"]


def _count(text, phrases):
    return sum(len(re.findall(r"\b" + re.escape(p) + r"\b", text)) for p in phrases)


def text_marker_vector(transcript: str) -> torch.Tensor:
    t = (transcript or "").lower()
    words = re.findall(r"[a-z']+", t)
    nw = max(len(words), 1)
    sents = [s for s in re.split(r"[.!?]+", t) if s.strip()]
    ns = max(len(sents), 1)

    # repetition proxy: adjacent duplicate words
    reps = sum(1 for i in range(1, len(words)) if words[i] == words[i - 1])

    # lexical polarity oscillation across sentences (sign changes)
    signs = []
    for s in sents:
        pos, neg = _count(s, POS_LEX), _count(s, NEG_LEX)
        signs.append(1 if pos > neg else (-1 if neg > pos else 0))
    nz = [s for s in signs if s != 0]
    flips = sum(1 for i in range(1, len(nz)) if nz[i] != nz[i - 1])

    feats = [
        _count(t, HEDGES) / nw,
        _count(t, FILLERS) / nw,
        _count(t, CONTRAST) / nw,
        _count(t, NEGATION) / nw,
        _count(t, UNCERTAIN_1P) / nw,
        reps / nw,
        flips / ns,
        _count(t, POS_LEX) / nw,
        _count(t, NEG_LEX) / nw,
        np.log1p(nw) / 10.0,      # verbosity (log #words)
        ns / nw,                   # sentence density (short/broken speech)
    ]
    return torch.tensor(feats, dtype=torch.float32)


def extract_text_markers(split_file=None):
    out_dir = os.path.join(AFFECT_DIR, "text_markers"); os.makedirs(out_dir, exist_ok=True)
    for s in tqdm(_samples(split_file), desc="text-markers"):
        out_path = os.path.join(out_dir, s["basename"].replace(".mp4", ".pt"))
        torch.save(text_marker_vector(s["transcript"]), out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["visual", "audio", "text", "all"], default="all")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--split-file", default=None)
    args = ap.parse_args()
    if args.modality in ("text", "all"):
        extract_text_markers(args.split_file)
    if args.modality in ("visual", "all"):
        extract_visual(args.device, args.split_file)
    if args.modality in ("audio", "all"):
        extract_audio(args.device, args.split_file)
    print("Affect feature extraction complete ->", AFFECT_DIR)


if __name__ == "__main__":
    main()

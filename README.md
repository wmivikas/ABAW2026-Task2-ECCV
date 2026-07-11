# Ambivalence / Hesitancy Recognition in Video
### ABAW 2026 · 3rd A/H Challenge · BAH dataset

> Given a short interview video, decide: does this person show **ambivalence or
> hesitancy**? Six models, one honest ensemble, one command.

**Public test: 0.731 macro-F1** 

---

## TL;DR — run it

```bash
# 1. environment
conda create -n bah python=3.10 -y && conda activate bah
pip install -r requirements.txt          # + ffmpeg on PATH, 1 GPU

# 2. put the BAH data next to this folder (see "Data layout" below)

# 3. everything: features → training → ensemble → report → submission
bash run_all.sh
```

That is the whole pipeline. Seed 42, no hand-tuning, no hidden steps.
Resume anywhere: `bash run_all.sh --step 8`.

---

## How it works

```
                 ┌──────────────┐
  transcript ──► │ text model   │──┐
                 ├──────────────┤  │
  face frames ─► │ video model  │──┤
                 ├──────────────┤  │     AP-weighted        fixed
  waveform ────► │ audio model  │──┼──►  average of 6  ──►  threshold ──► A/H?
                 ├──────────────┤  │     probabilities        0.5
  all three ───► │ fusion ×2    │──┤
                 ├──────────────┤  │
  affect feats ► │ AMF (ours)   │──┘
                 └──────────────┘
```

- **Text** RoBERTa fine-tuned on GoEmotions · **Video** VideoMAE on cropped faces
  · **Audio** HuBERT · **Fusion** orthogonal + gated-attention variants
- **AMF** — our member: FER-ViT face statistics + emotion-wav2vec2 audio +
  **11 readable hesitation markers** (hedges, filled pauses, "but", repeated
  words, sentiment flips). A learned gate turns weak channels down.
- **The key decision is calibration, not architecture.** Tuning ensemble weights
  and threshold on the 124-video validation split overfits: 0.741 val → **0.690
  test**. Weighting members by AP and fixing the threshold at 0.5 gives
  **0.731** — that one change is worth more than any model change we tried.

### Results

| Setup | Macro-F1 | AP |
|---|:---:|:---:|
| Ensemble, public test (models trained on train split) | **0.731** | 0.875 |
| Ensemble, internal 113-video holdout (all-data models) | 0.814 | 0.907 |


Verified end-to-end: a from-scratch rerun of `run_all.sh` reproduces the
ensemble to ±0.0001 F1 (individual members may drift ~a point across
GPUs/sessions; the near-uniform AP weighting absorbs it).

---

## Data layout

BAH is EULA-protected — request it from the organisers, then place it **next to**
this folder:

```
<anywhere>/
├── code/                        ← this folder (any name works)
│   └── run_all.sh, *.py, models/
└── data/
    ├── Videos/<id>/Visite_1/*.mp4
    ├── cropped-aligned-faces/Videos/<id>/Visite_1/<video>.mp4/*.jpg
    ├── transcription/Videos/...
    ├── split/train.txt  val.txt  test.txt     # video_path,label,transcript
    └── private/private_test.txt               # private split (labels = 0)
```

**Private test:** the private zip already contains its split file in this exact
format (`split/test.txt`). Copy it to `data/private/private_test.txt` and
copy/symlink the 30 private participants into `data/Videos/` and
`data/cropped-aligned-faces/Videos/` — their IDs don't clash with the 300
labelled ones.

---

## What run_all.sh does, step by step

| Step | What happens | Output |
|:---:|---|---|
| 1 | extract 16 kHz audio from every video | `data/audio/` |
| 2 | text / video / audio encoder features | `data/features/` |
| 3–7 | train the five encoder members on **all** 1427 labelled videos | `outputs/checkpoints/` |
| 8 | affect features: FER-ViT, emotion-wav2vec2, 11 markers | `data/features/affect/` |
| 9 | train AMF | `outputs/checkpoints/` |
| 10 | score all 6 on the shared 113-video holdout, lock AP weights, τ = 0.5 | `outputs/final_ensemble_config.json` |
| 11 | full report: per-class F1, AP, confusion matrix, per-question table | console |
| 12 | predict the 152 private-test videos | `outputs/predictions/` |
| 13 | write `trial-0.txt … trial-4.txt` + zip them | `predictions.zip` |

Steps 12–13 need two environment variables and skip cleanly without them:

```bash
DEVICE=cuda:0 \
PRIVATE_SPLIT=../data/private/private_test.txt \
REFERENCE=/path/to/organisers/with_probabilities/trial-0.txt \
bash run_all.sh
```

`REFERENCE` is the organisers' template file — it fixes the required video
order. The writer guarantees the official format: `video_id,p0,p1,label`,
same 152-video order, and `p0 + p1 == 1.0` under exact float equality.

### Why training on all data is still honest

There is no validation split left when you train on everything, so every
trainer carves out the **same** seeded 8% holdout (113 videos) for early
stopping — from **one shared function**, [`holdout.py`](holdout.py). No model is
ever scored or weighted on a video it trained on, and the training and
inference paths are asserted to produce identical probabilities.

---

## The five submission trials

| Trial | Contents |
|:---:|---|
| **0** | **6-member AP-weighted ensemble, τ = 0.5 — the main submission** |
| 1 | same probabilities, threshold matched to the labelled positive rate |
| 2 | AMF alone |
| 3 | text member alone |
| 4 | rank-average of the 6 members |

---

## File map

| File | Role |
|---|---|
| `run_all.sh` | the one script |
| `config.py` | all paths (self-locating) + question metadata |
| `dataset.py`, `augmentation.py` | split parsing, multimodal loading |
| `holdout.py` | **single source of truth** for the 8% internal holdout |
| `extract_audio.py` | mp4 → 16 kHz wav (ffmpeg) |
| `extract_features.py` | text / video / audio encoder features |
| `extract_affect_features.py` | FER-ViT, emotion-wav2vec2, 11 hesitation markers |
| `train.py` | trains text · visual · audio · fusion (`--pool-final`) |
| `train_affect.py` | trains AMF |
| `finalize_ensemble.py` | locks AP weights on the shared holdout |
| `report_results.py` | per-class / per-question report |
| `predict_private_test.py` | end-to-end private-test inference (auto-extracts missing features) |
| `make_submission.py`, `make_all_trials.py` | official trial files |
| `models/` | the six model definitions |

---

## Ethics & data use

BAH is EULA-protected and not redistributed here. Participants **83249** and
**83277** did not consent to publication use of their data: they are used for
challenge predictions only and appear in no figure, table, or example.

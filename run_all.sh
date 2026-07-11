#!/bin/bash
# =====================================================================
# ABAW 2026 -- 3rd A/H Video Recognition Challenge (BAH dataset)
# ONE script: features -> train 6 models on ALL labelled data ->
# lock the ensemble -> report -> predict the private test -> write
# the official submission files (trial-0..4).
#
# Usage:
#   bash run_all.sh                # everything from scratch
#   bash run_all.sh --step 8       # resume from step 8
#
# Optional environment variables:
#   DEVICE=cuda:0                  # GPU (default cuda:0)
#   PRIVATE_SPLIT=../data/private/private_test.txt
#                                  # the private test split file
#                                  # (format: video_path,label,transcript;
#                                  #  label is an unused 0 placeholder)
#   REFERENCE=trial-template.txt   # organisers' with_probabilities/trial-0.txt
#                                  # (fixes the required video order)
#
# Data layout expected (see README.md):
#   ../data/Videos/...   ../data/cropped-aligned-faces/Videos/...
#   ../data/transcription/Videos/...   ../data/split/{train,val,test}.txt
#
# Everything is deterministic: seed 42, repeated runs give the same
# numbers. Steps 12-13 are skipped automatically if PRIVATE_SPLIT or
# REFERENCE is not set / not found.
# =====================================================================
set -e

DEVICE=${DEVICE:-cuda:0}
PRIVATE_SPLIT=${PRIVATE_SPLIT:-../data/private/private_test.txt}
REFERENCE=${REFERENCE:-}

START_STEP=1
if [ "$1" == "--step" ]; then START_STEP=$2; fi

GREEN='\033[0;32m'; BLUE='\033[0;34m'; NC='\033[0m'
CODE_DIR="$(cd "$(dirname "$0")" && pwd)"; cd "$CODE_DIR"
hdr()  { echo ""; echo -e "${BLUE}=== STEP $1: $2 ===${NC}"; echo ""; }
done_() { echo -e "${GREEN}  done: step $1${NC}"; }

if [ "$START_STEP" -le 1 ]; then
    hdr 1 "Extract 16 kHz audio from every video"
    python3 extract_audio.py
    done_ 1
fi

if [ "$START_STEP" -le 2 ]; then
    hdr 2 "Extract text / video / audio encoder features"
    python3 extract_features.py --modality text   --device "$DEVICE" --batch-size 32
    python3 extract_features.py --modality visual --device "$DEVICE" --num-frames 16
    python3 extract_features.py --modality audio  --device "$DEVICE"
    done_ 2
fi

if [ "$START_STEP" -le 3 ]; then
    hdr 3 "Train text member (all data, seed 42)"
    python3 train.py --model text --text-model SamLowe/roberta-base-go_emotions \
        --pool-final --epochs 50 --batch-size 16 --lr 2e-5 --dropout 0.3 \
        --freeze-layers 8 --patience 15 --rdrop-alpha 0.7 --seed 42 \
        --device "$DEVICE" --no-focal-loss --label-smoothing 0.1 --bf16
    done_ 3
fi

if [ "$START_STEP" -le 4 ]; then
    hdr 4 "Train video member"
    python3 train.py --model visual --visual-model MCG-NJU/videomae-base \
        --pool-final --epochs 40 --batch-size 4 --lr 1e-4 --dropout 0.3 \
        --num-frames 16 --patience 15 --seed 42 \
        --device "$DEVICE" --no-focal-loss --label-smoothing 0.1 --bf16
    done_ 4
fi

if [ "$START_STEP" -le 5 ]; then
    hdr 5 "Train audio member"
    python3 train.py --model audio --audio-model facebook/hubert-large-ll60k \
        --pool-final --epochs 40 --batch-size 8 --lr 1e-4 --dropout 0.3 \
        --patience 15 --seed 42 \
        --device "$DEVICE" --no-focal-loss --label-smoothing 0.1 --bf16
    done_ 5
fi

if [ "$START_STEP" -le 6 ]; then
    hdr 6 "Train fusion member (orthogonal)"
    python3 train.py --model fusion --fusion-version v1 \
        --pool-final --hidden-dim 512 --num-fusion-layers 2 \
        --epochs 40 --batch-size 16 --lr 1e-4 --dropout 0.3 \
        --patience 15 --rdrop-alpha 0.5 --seed 42 \
        --device "$DEVICE" --no-focal-loss --label-smoothing 0.1 --bf16
    done_ 6
fi

if [ "$START_STEP" -le 7 ]; then
    hdr 7 "Train fusion member (gated attention)"
    python3 train.py --model fusion --fusion-version v2 \
        --pool-final --hidden-dim 256 \
        --epochs 60 --batch-size 16 --lr 1e-4 --dropout 0.3 \
        --patience 15 --rdrop-alpha 0.5 --seed 42 \
        --device "$DEVICE" --no-focal-loss --label-smoothing 0.1 --bf16
    done_ 7
fi

if [ "$START_STEP" -le 8 ]; then
    hdr 8 "Extract affect features (FER-ViT + emotion audio + 11 text markers)"
    python3 extract_affect_features.py --modality all --device "$DEVICE"
    done_ 8
fi

if [ "$START_STEP" -le 9 ]; then
    hdr 9 "Train AMF member"
    python3 train_affect.py --pool-final --seed 42 --device "$DEVICE"
    done_ 9
fi

if [ "$START_STEP" -le 10 ]; then
    hdr 10 "Lock the AP-weighted ensemble (fixed threshold 0.5)"
    python3 finalize_ensemble.py --device "$DEVICE"
    done_ 10
fi

if [ "$START_STEP" -le 11 ]; then
    hdr 11 "Report: per-class F1, AP, confusion, per-question (113-video holdout)"
    python3 report_results.py --device "$DEVICE"
    done_ 11
fi

if [ "$START_STEP" -le 12 ]; then
    if [ -f "$PRIVATE_SPLIT" ]; then
        hdr 12 "Predict the private test set"
        python3 predict_private_test.py --split-file "$PRIVATE_SPLIT" \
            --device "$DEVICE" \
            --output outputs/predictions/private_test_submission.csv
        done_ 12
    else
        echo "  (step 12 skipped: PRIVATE_SPLIT not found at $PRIVATE_SPLIT)"
    fi
fi

if [ "$START_STEP" -le 13 ]; then
    if [ -f "$PRIVATE_SPLIT" ] && [ -n "$REFERENCE" ] && [ -f "$REFERENCE" ]; then
        hdr 13 "Write the official submission files (trial-0..4)"
        python3 make_all_trials.py --split-file "$PRIVATE_SPLIT" --reference "$REFERENCE"
        ( cd submissions && zip -q ../predictions.zip trial-0.txt trial-1.txt trial-2.txt trial-3.txt trial-4.txt )
        echo "  wrote predictions.zip"
        done_ 13
    else
        echo "  (step 13 skipped: set REFERENCE=<organisers' with_probabilities/trial-0.txt>)"
    fi
fi

echo ""
echo -e "${GREEN}=== ALL DONE ===${NC}"
echo "Ensemble config : outputs/final_ensemble_config.json"
echo "Submission zip  : predictions.zip (if steps 12-13 ran)"

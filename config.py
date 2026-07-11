"""
Configuration for ABAW 2026 Task 2 - A/H Video Recognition Challenge.
Central config file for all hyperparameters, paths, and model settings.
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional


# ============================================================
# Path Configuration
# ============================================================
# Self-locating: this code folder can have any name; data/ sits next to it,
# outputs/ (checkpoints, predictions, logs) go inside it.
CODE_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CODE_ROOT)
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
OUTPUT_ROOT = os.path.join(CODE_ROOT, "outputs")

# Data paths
VIDEOS_DIR = os.path.join(DATA_ROOT, "Videos")
FACES_DIR = os.path.join(DATA_ROOT, "cropped-aligned-faces", "Videos")
AUDIO_DIR = os.path.join(DATA_ROOT, "audio")
TRANSCRIPTION_DIR = os.path.join(DATA_ROOT, "transcription", "Videos")
FEATURES_DIR = os.path.join(DATA_ROOT, "features")

# Split files
TRAIN_SPLIT = os.path.join(DATA_ROOT, "split", "train.txt")
VAL_SPLIT = os.path.join(DATA_ROOT, "split", "val.txt")
TEST_SPLIT = os.path.join(DATA_ROOT, "split", "test.txt")

# Output paths
CHECKPOINT_DIR = os.path.join(OUTPUT_ROOT, "checkpoints")
LOG_DIR = os.path.join(OUTPUT_ROOT, "logs")
PREDICTION_DIR = os.path.join(OUTPUT_ROOT, "predictions")

# ============================================================
# Question Info (strong prior for A/H detection)
# ============================================================
QUESTION_INFO = {
    1: {"response": "Neutral", "prompt": "Activity after waking up"},
    2: {"response": "Positive", "prompt": "Activity that brings joy"},
    3: {"response": "Negative", "prompt": "Activity you dislike"},
    4: {"response": "Ambivalent", "prompt": "Guilty pleasure / want to stop or start"},
    5: {"response": "Willing", "prompt": "Activity always willing to do"},
    6: {"response": "Resistant", "prompt": "Something others do you would not"},
    7: {"response": "Hesitant", "prompt": "Something you could have done but haven't"},
}

# Questions designed to elicit A/H (higher prior for label=1)
AH_ELICITING_QUESTIONS = [4, 7]


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # General
    seed: int = 42
    num_epochs: int = 30
    batch_size: int = 16
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # Optimizer
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    
    # Early stopping
    patience: int = 7
    min_delta: float = 0.001
    
    # Mixed precision
    fp16: bool = False
    bf16: bool = True
    
    # Data
    num_workers: int = 4
    pin_memory: bool = True
    
    # Loss
    focal_loss: bool = True
    focal_gamma: float = 2.0
    focal_alpha: float = 0.4  # weight for class 0 (minority in test)
    
    # GPU
    device: str = "cuda:0"
    multi_gpu: bool = False


@dataclass
class TextModelConfig:
    """Text classifier configuration."""
    model_name: str = "SamLowe/roberta-base-go_emotions"
    max_length: int = 512
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 20
    dropout: float = 0.1
    use_question_prompt: bool = True
    freeze_layers: int = 0  # number of bottom layers to freeze


@dataclass
class VisualModelConfig:
    """Visual classifier configuration."""
    backbone: str = "MCG-NJU/videomae-base"
    num_frames: int = 16  # frames to sample per video
    image_size: int = 224
    feature_dim: int = 768  # VideoMAE-Base output dim
    temporal_pool: str = "attention"  # "mean", "max", "attention"
    learning_rate: float = 1e-4
    batch_size: int = 8
    num_epochs: int = 25
    dropout: float = 0.3
    freeze_backbone: bool = True  # freeze visual backbone, train head only


@dataclass
class AudioModelConfig:
    """Audio classifier configuration."""
    backbone: str = "facebook/hubert-large-ll60k"
    sample_rate: int = 16000
    max_duration: float = 30.0  # seconds
    feature_dim: int = 1024
    temporal_pool: str = "attention"
    learning_rate: float = 1e-4
    batch_size: int = 8
    num_epochs: int = 25
    dropout: float = 0.3
    freeze_backbone: bool = True


@dataclass
class FusionModelConfig:
    """Multimodal fusion configuration."""
    text_dim: int = 768
    visual_dim: int = 768
    audio_dim: int = 1024
    hidden_dim: int = 512
    num_attention_heads: int = 8
    num_fusion_layers: int = 2
    dropout: float = 0.3
    use_question_embedding: bool = True
    question_embed_dim: int = 32
    gate_fusion: bool = True
    learning_rate: float = 1e-4
    batch_size: int = 16
    num_epochs: int = 30


@dataclass
class EnsembleConfig:
    """Ensemble configuration."""
    method: str = "weighted_average"  # "weighted_average", "stacking", "voting"
    optimize_threshold: bool = True
    temperature_scaling: bool = True
    n_folds_stacking: int = 5

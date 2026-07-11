"""
Main training script for A/H Video Recognition.

Supports training individual modality models and the multimodal fusion model.
Usage:
    python train.py --model text --epochs 20 --lr 2e-5
    python train.py --model visual --epochs 25 --lr 1e-4
    python train.py --model audio --epochs 25 --lr 1e-4
    python train.py --model fusion --epochs 30 --lr 1e-4
"""
import os
import sys
import json
import time
import random
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

import config
from dataset import BAHDataset, get_dataloader
from evaluate import compute_metrics, print_metrics


# ============================================================
# Loss functions
# ============================================================
class FocalLoss(nn.Module):
    """Focal loss for imbalanced binary classification."""
    
    def __init__(self, gamma: float = 2.0, alpha: float = 0.4):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        
        # Apply alpha weighting
        alpha_t = torch.where(targets == 1, 1 - self.alpha, self.alpha)
        
        focal_loss = alpha_t * ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ============================================================
# Utilities
# ============================================================
def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model(args):
    """Initialize model based on args.model type."""
    if args.model == "text":
        from models.text_classifier import TextClassifier, get_tokenizer
        model = TextClassifier(
            model_name=args.text_model,
            dropout=args.dropout,
            use_question_embedding=True,
            freeze_layers=args.freeze_layers,
        )
        tokenizer = get_tokenizer(args.text_model)
        return model, tokenizer
    
    elif args.model == "llm":
        from models.llm_classifier import LLMClassifier, get_llm_tokenizer
        model = LLMClassifier(
            model_name=args.llm_model,
            device=args.device,
        )
        tokenizer = get_llm_tokenizer(args.llm_model)
        return model, tokenizer
    
    elif args.model == "visual":
        from models.visual_classifier import VisualClassifier
        model = VisualClassifier(
            backbone=args.visual_model,
            dropout=args.dropout,
            freeze_backbone=True,
        )
        return model, None
    
    elif args.model == "audio":
        from models.audio_classifier import AudioClassifier
        model = AudioClassifier(
            backbone=args.audio_model,
            dropout=args.dropout,
            freeze_backbone=True,
        )
        return model, None
    
    elif args.model == "fusion":
        if getattr(args, "fusion_version", "v1") == "v2":
            from models.multimodal_fusion_v2 import MultimodalFusionV2
            model = MultimodalFusionV2(
                text_dim=768,
                visual_dim=768,
                audio_dim=1024,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
            )
        else:
            from models.multimodal_fusion import MultimodalFusionModel
            model = MultimodalFusionModel(
                text_dim=768,
                visual_dim=768,
                audio_dim=1024,
                hidden_dim=args.hidden_dim,
                num_fusion_layers=args.num_fusion_layers,
                dropout=args.dropout,
                conflict_type=getattr(args, "conflict_type", "orthogonal"),
            )
        return model, None
    
    else:
        raise ValueError(f"Unknown model type: {args.model}")


def get_dataloaders(args, tokenizer=None):
    """Create train and validation dataloaders."""
    modalities = [args.model] if args.model not in ("fusion", "llm") else (["text", "visual", "audio"] if args.model == "fusion" else ["text"])
    
    common_kwargs = dict(
        modalities=modalities,
        num_frames=args.num_frames,
        text_tokenizer=tokenizer,
        text_max_length=args.max_length,
        use_question_prompt=True,
    )
    
    if args.model == "fusion":
        common_kwargs["use_precomputed_features"] = True
        common_kwargs.pop("text_tokenizer", None)
        common_kwargs["text_tokenizer"] = None
    
    train_dataset = BAHDataset(split="train", **common_kwargs)
    val_dataset = BAHDataset(split="val", **common_kwargs)

    if getattr(args, "pool_final", False):
        # Final production model: use ALL labeled data (train+val+test), but
        # keep automatic best-checkpoint selection by carving a small
        # stratified internal holdout out of the pool for early stopping —
        # same train()/validate() loop as always, nothing hand-picked.
        test_dataset = BAHDataset(split="test", **common_kwargs)
        pool = train_dataset.samples + val_dataset.samples + test_dataset.samples

        # Shared holdout: every trainer and the ensembler MUST use this one
        # function (holdout.py), so no member is ever scored on its own
        # training videos.
        from holdout import split_pool_final
        train_dataset.samples, val_dataset.samples = split_pool_final(pool, seed=args.seed)
        print(f"[pool-final] Pooled {len(pool)} labeled videos -> "
              f"{len(train_dataset.samples)} train / {len(val_dataset.samples)} internal holdout "
              f"(early stopping still automatic)")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader


# ============================================================
# Training loop
# ============================================================
def _forward_model(model, batch, args):
    """Shared forward pass logic for all model types."""
    if args.model in ("text", "llm"):
        return model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            question_num=batch["question_num"],
            token_type_ids=batch.get("token_type_ids"),
        )
    elif args.model == "visual":
        return model(frames=batch["frames"], question_num=batch["question_num"])
    elif args.model == "audio":
        return model(waveform=batch["waveform"], question_num=batch["question_num"])
    elif args.model == "fusion":
        return model(
            text_features=batch["text_features"],
            visual_features=batch["visual_features"],
            audio_features=batch["audio_features"],
            question_num=batch["question_num"],
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")


def compute_kl_loss(p_logits, q_logits):
    """Compute symmetric KL divergence between two sets of logits."""
    p = torch.nn.functional.log_softmax(p_logits, dim=-1)
    q = torch.nn.functional.log_softmax(q_logits, dim=-1)
    
    p_loss = torch.nn.functional.kl_div(p, q.exp(), reduction='batchmean')
    q_loss = torch.nn.functional.kl_div(q, p.exp(), reduction='batchmean')
    
    return (p_loss + q_loss) / 2


def train_one_epoch(model, train_loader, optimizer, criterion, scaler, scheduler, device, args):
    """Train for one epoch with optional R-Drop regularization."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    rdrop_alpha = getattr(args, 'rdrop_alpha', 0.0)
    
    for batch_idx, batch in enumerate(train_loader):
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        labels = batch["label"]
        
        # Forward pass
        with autocast(device_type="cuda", dtype=torch.bfloat16 if args.bf16 else torch.float16, enabled=args.use_amp):
            logits = _forward_model(model, batch, args)
            loss = criterion(logits, labels)
            
            # R-Drop: forward same input again with different dropout, compute KL loss
            if rdrop_alpha > 0 and model.training:
                logits2 = _forward_model(model, batch, args)
                loss2 = criterion(logits2, labels)
                # Average the two CE losses + KL divergence regularization
                kl_loss = compute_kl_loss(logits, logits2)
                loss = (loss + loss2) / 2 + rdrop_alpha * kl_loss
            
            if torch.isnan(loss):
                print(f"NaN Loss at batch {batch_idx}! Logits: {logits}")
                break
            
            loss = loss / args.gradient_accumulation_steps
        
        # Backward pass
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * args.gradient_accumulation_steps
        
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(train_loader)
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    
    return avg_loss, metrics


@torch.no_grad()
def validate(model, val_loader, criterion, device, args):
    """Validate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    for batch in val_loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        labels = batch["label"]
        
        with autocast(device_type="cuda", dtype=torch.bfloat16 if args.bf16 else torch.float16, enabled=args.use_amp):
            if args.model in ("text", "llm"):
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    question_num=batch["question_num"],
                    token_type_ids=batch.get("token_type_ids"),
                )
            elif args.model == "visual":
                logits = model(frames=batch["frames"], question_num=batch["question_num"])
            elif args.model == "audio":
                logits = model(waveform=batch["waveform"], question_num=batch["question_num"])
            elif args.model == "fusion":
                logits = model(
                    text_features=batch["text_features"],
                    visual_features=batch["visual_features"],
                    audio_features=batch["audio_features"],
                    question_num=batch["question_num"],
                )
            
            loss = criterion(logits, labels)
            
            if torch.isnan(loss):
                print(f"NaN Loss in Validation at batch! Logits: {logits}")
                break
                
        total_loss += loss.item()
        
        probs = torch.softmax(logits, dim=-1)[:, 1]
        preds = torch.argmax(logits, dim=-1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.float().cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    metrics = compute_metrics(
        np.array(all_labels), 
        np.array(all_preds),
        np.array(all_probs),
    )
    
    return avg_loss, metrics


def train(args):
    """Main training function."""
    # Setup
    set_seed(args.seed)
    device = torch.device(args.device)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.model}_{timestamp}"
    run_dir = os.path.join(config.CHECKPOINT_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Save args
    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    print("=" * 60)
    print(f"  Training: {args.model} model")
    print(f"  Run: {run_name}")
    print(f"  Device: {device}")
    print("=" * 60)
    
    # Initialize model
    model, tokenizer = get_model(args)
    model = model.to(device)
    print(f"  Trainable parameters: {count_parameters(model):,}")
    
    # Data
    train_loader, val_loader = get_dataloaders(args, tokenizer)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        eps=1e-6,
    )
    
    # Scheduler
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    
    # Loss
    if args.focal_loss:
        criterion = FocalLoss(gamma=args.focal_gamma, alpha=args.focal_alpha)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    # AMP scaler (not needed for bfloat16)
    scaler = GradScaler("cuda", enabled=args.use_amp and not args.bf16)
    
    # Tensorboard
    writer = SummaryWriter(os.path.join(config.LOG_DIR, run_name))
    
    # Training loop
    best_macro_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, scheduler, device, args
        )

        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device, args)

        epoch_time = time.time() - epoch_start

        # Log
        print(f"\nEpoch {epoch}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Train — Loss: {train_loss:.4f} | Macro-F1: {train_metrics['macro_f1']:.4f}")
        print(f"  Val   — Loss: {val_loss:.4f} | Macro-F1: {val_metrics['macro_f1']:.4f} | AP: {val_metrics.get('ap', 0):.4f}")

        writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("macro_f1", {"train": train_metrics["macro_f1"], "val": val_metrics["macro_f1"]}, epoch)

        # Save best model
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            patience_counter = 0

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_macro_f1": best_macro_f1,
                "val_metrics": val_metrics,
                "args": vars(args),
            }
            torch.save(checkpoint, os.path.join(run_dir, "best_model.pt"))
            print(f"  ★ New best! Macro-F1: {best_macro_f1:.4f}")
        else:
            patience_counter += 1
        
        # Save latest
        torch.save(
            {"epoch": epoch, "model_state_dict": model.state_dict()},
            os.path.join(run_dir, "latest_model.pt"),
        )
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n  Early stopping after {args.patience} epochs without improvement.")
            break
    
    writer.close()
    
    print("\n" + "=" * 60)
    print(f"  Training complete!")
    print(f"  Best Val Macro-F1: {best_macro_f1:.4f}")
    print(f"  Checkpoint: {run_dir}")
    print("=" * 60)
    
    return run_dir, best_macro_f1


# ============================================================
# CLI
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train A/H recognition model")
    
    # Model selection
    parser.add_argument("--model", type=str, default="text",
                        choices=["text", "visual", "audio", "fusion", "llm"],
                        help="Model type to train")
    
    # Model-specific
    parser.add_argument("--text-model", type=str, default="SamLowe/roberta-base-go_emotions")
    parser.add_argument("--llm-model", type=str, default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--visual-model", type=str, default="MCG-NJU/videomae-base")
    parser.add_argument("--audio-model", type=str, default="facebook/hubert-large-ll60k")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-fusion-layers", type=int, default=2)
    parser.add_argument("--fusion-version", type=str, default="v1", choices=["v1", "v2"])
    parser.add_argument("--conflict-type", type=str, default="orthogonal",
                        choices=["orthogonal", "absdiff", "none"],
                        help="Cross-modal interaction for fusion v1: orthogonal (OCCF, ours), "
                             "absdiff (ConflictAwareAH |diff|), or none (ablation)")
    parser.add_argument("--freeze-layers", type=int, default=0)
    
    # Training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--rdrop-alpha", type=float, default=0.0,
                        help="R-Drop KL regularization weight (0.5-1.0 recommended)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pool-final", action="store_true",
                        help="Final production model: train on all labeled data "
                             "(train+val+test), with an internal stratified holdout "
                             "carved out automatically for early stopping")
    
    # Loss
    parser.add_argument("--focal-loss", action="store_true", default=True)
    parser.add_argument("--no-focal-loss", dest="focal_loss", action="store_false")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha", type=float, default=0.4)
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Label smoothing epsilon (0.1 recommended for small datasets)")
    
    # Hardware
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")
    parser.add_argument("--use-amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="use_amp", action="store_false")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

"""
Generate predictions on the test set.

Usage:
    python predict.py --checkpoint outputs/checkpoints/text_20260614/best_model.pt --model text
"""
import os
import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

import config
from dataset import BAHDataset
from evaluate import optimize_threshold


def load_model(checkpoint_path: str, model_type: str, device: torch.device):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_args = checkpoint.get("args", {})
    
    if model_type == "text":
        from models.text_classifier import TextClassifier, get_tokenizer
        model_name = saved_args.get("text_model", "SamLowe/roberta-base-go_emotions")
        model = TextClassifier(
            model_name=model_name,
            dropout=saved_args.get("dropout", 0.1),
        )
        tokenizer = get_tokenizer(model_name)
    elif model_type == "visual":
        from models.visual_classifier import VisualClassifier
        model = VisualClassifier(
            backbone=saved_args.get("visual_model", "MCG-NJU/videomae-base"),
            dropout=saved_args.get("dropout", 0.3),
        )
        tokenizer = None
    elif model_type == "audio":
        from models.audio_classifier import AudioClassifier
        model = AudioClassifier(
            backbone=saved_args.get("audio_model", "facebook/hubert-large-ll60k"),
            dropout=saved_args.get("dropout", 0.3),
        )
        tokenizer = None
    elif model_type == "fusion":
        if saved_args.get("fusion_version", "v1") == "v2":
            from models.multimodal_fusion_v2 import MultimodalFusionV2
            model = MultimodalFusionV2(
                hidden_dim=saved_args.get("hidden_dim", 256),
                dropout=saved_args.get("dropout", 0.3),
            )
        else:
            from models.multimodal_fusion import MultimodalFusionModel
            model = MultimodalFusionModel(
                hidden_dim=saved_args.get("hidden_dim", 512),
                num_fusion_layers=saved_args.get("num_fusion_layers", 2),
                dropout=saved_args.get("dropout", 0.3),
                conflict_type=saved_args.get("conflict_type", "orthogonal"),
            )
        tokenizer = None
    elif model_type == "llm":
        from models.llm_classifier import LLMClassifier, get_llm_tokenizer
        model_name = saved_args.get("llm_model", "meta-llama/Meta-Llama-3.1-8B")
        model = LLMClassifier(
            model_name=model_name,
            device=str(device),
        )
        tokenizer = get_llm_tokenizer(model_name)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model = model.to(device)
    model.eval()
    
    return model, tokenizer, checkpoint


@torch.no_grad()
def predict(model, dataloader, device, model_type, use_amp=True):
    """Generate predictions."""
    all_probs = []
    all_labels = []
    all_question_nums = []
    
    for batch in dataloader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        with autocast(dtype=torch.bfloat16, enabled=use_amp):
            if model_type in ("text", "llm"):
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    question_num=batch["question_num"],
                    token_type_ids=batch.get("token_type_ids"),
                )
            elif model_type == "visual":
                logits = model(frames=batch["frames"], question_num=batch["question_num"])
            elif model_type == "audio":
                logits = model(waveform=batch["waveform"], question_num=batch["question_num"])
            elif model_type == "fusion":
                logits = model(
                    text_features=batch["text_features"],
                    visual_features=batch["visual_features"],
                    audio_features=batch["audio_features"],
                    question_num=batch["question_num"],
                )
        
        probs = torch.softmax(logits, dim=-1)[:, 1].float().cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(batch["label"].cpu().numpy())
        all_question_nums.extend(batch["question_num"].cpu().numpy())
    
    return np.array(all_probs), np.array(all_labels), np.array(all_question_nums)


def generate_submission(
    dataset: BAHDataset,
    predictions: np.ndarray,
    output_path: str,
):
    """Generate submission file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        for i, sample in enumerate(dataset.samples):
            video_path = sample["video_path"]
            pred = int(predictions[i])
            f.write(f"{video_path},{pred}\n")
    
    print(f"Submission saved to: {output_path}")
    print(f"  Total predictions: {len(predictions)}")
    print(f"  Positive: {predictions.sum()} ({predictions.mean()*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=["text", "visual", "audio", "fusion", "llm"])
    parser.add_argument("--llm-model", type=str, default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=None, help="Classification threshold (auto if None)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # Load model
    print(f"Loading {args.model} model from {args.checkpoint}")
    model, tokenizer, checkpoint = load_model(args.checkpoint, args.model, device)
    
    # Determine optimal threshold from validation set
    if args.threshold is None and args.split == "test":
        print("Finding optimal threshold on validation set...")
        modalities = [args.model] if args.model not in ("fusion", "llm") else (["text", "visual", "audio"] if args.model == "fusion" else ["text"])
        val_dataset = BAHDataset(
            split="val", modalities=modalities,
            text_tokenizer=tokenizer,
            use_precomputed_features=(args.model == "fusion"),
        )
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        val_probs, val_labels, _ = predict(model, val_loader, device, args.model)
        
        threshold, best_f1 = optimize_threshold(val_labels, val_probs)
        print(f"  Optimal threshold: {threshold:.3f} (Val Macro-F1: {best_f1:.4f})")
    else:
        threshold = args.threshold or 0.5
    
    # Generate predictions on target split
    print(f"\nPredicting on {args.split} set with threshold={threshold:.3f}")
    modalities = [args.model] if args.model not in ("fusion", "llm") else (["text", "visual", "audio"] if args.model == "fusion" else ["text"])
    test_dataset = BAHDataset(
        split=args.split, modalities=modalities,
        text_tokenizer=tokenizer,
        use_precomputed_features=(args.model == "fusion"),
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    probs, labels, q_nums = predict(model, test_loader, device, args.model)
    predictions = (probs >= threshold).astype(int)
    
    # If we have labels (val set), compute metrics
    if args.split != "test" or labels.max() > 0:
        from evaluate import compute_metrics, print_metrics, per_question_analysis
        metrics = compute_metrics(labels, predictions, probs)
        print_metrics(metrics, prefix="  ")
        per_question_analysis(labels, predictions, q_nums)
    
    # Save predictions
    output_path = args.output or os.path.join(
        config.PREDICTION_DIR,
        f"{args.model}_{args.split}_predictions.csv"
    )
    generate_submission(test_dataset, predictions, output_path)
    
    # Also save probabilities
    prob_path = output_path.replace(".csv", "_probs.npy")
    np.save(prob_path, probs)
    print(f"Probabilities saved to: {prob_path}")


if __name__ == "__main__":
    main()

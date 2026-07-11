"""
Evaluation metrics for A/H recognition.

Primary metric: Macro-F1
Secondary: AP of positive class
"""
import numpy as np
from sklearn.metrics import (
    f1_score, 
    precision_score, 
    recall_score, 
    average_precision_score,
    confusion_matrix,
    accuracy_score,
    classification_report,
)


def compute_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray = None,
) -> dict:
    """
    Compute evaluation metrics for binary A/H classification.
    
    Args:
        labels: ground truth labels (0 or 1)
        predictions: predicted labels (0 or 1)
        probabilities: predicted probabilities for class 1 (optional, for AP)
    
    Returns:
        dict with all metrics
    """
    metrics = {}
    
    # Core metrics
    metrics["macro_f1"] = f1_score(labels, predictions, average="macro", zero_division=0)
    metrics["f1_class0"] = f1_score(labels, predictions, pos_label=0, zero_division=0)
    metrics["f1_class1"] = f1_score(labels, predictions, pos_label=1, zero_division=0)
    
    metrics["precision"] = precision_score(labels, predictions, zero_division=0)
    metrics["recall"] = recall_score(labels, predictions, zero_division=0)
    metrics["accuracy"] = accuracy_score(labels, predictions)
    
    # Average Precision for positive class
    if probabilities is not None:
        metrics["ap"] = average_precision_score(labels, probabilities)
    else:
        metrics["ap"] = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    metrics["tn"] = int(cm[0, 0])
    metrics["fp"] = int(cm[0, 1])
    metrics["fn"] = int(cm[1, 0])
    metrics["tp"] = int(cm[1, 1])
    
    return metrics


def optimize_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    metric: str = "macro_f1",
    thresholds: np.ndarray = None,
) -> tuple:
    """
    Find optimal classification threshold for Macro-F1.
    
    Args:
        labels: ground truth
        probabilities: predicted probabilities for class 1
        metric: metric to optimize ("macro_f1")
        thresholds: thresholds to search over
    
    Returns:
        (best_threshold, best_score)
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.9, 0.01)
    
    best_threshold = 0.5
    best_score = 0.0
    
    for thresh in thresholds:
        preds = (probabilities >= thresh).astype(int)
        score = f1_score(labels, preds, average="macro", zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = thresh
    
    return best_threshold, best_score


def print_metrics(metrics: dict, prefix: str = ""):
    """Pretty print metrics."""
    print(f"{prefix}Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"{prefix}  F1 (No A/H): {metrics['f1_class0']:.4f}")
    print(f"{prefix}  F1 (A/H):    {metrics['f1_class1']:.4f}")
    print(f"{prefix}  AP:          {metrics['ap']:.4f}")
    print(f"{prefix}  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"{prefix}  Confusion:   TP={metrics['tp']} FP={metrics['fp']} TN={metrics['tn']} FN={metrics['fn']}")


def per_question_analysis(labels, predictions, question_nums):
    """Analyze performance broken down by question number."""
    import config
    
    print("\n--- Per-Question Performance ---")
    print(f"{'Q':>3s} {'Type':>12s} {'N':>4s} {'Acc':>6s} {'F1':>6s} {'Pos%':>6s}")
    print("-" * 45)
    
    for q in sorted(set(question_nums)):
        mask = np.array(question_nums) == q
        q_labels = labels[mask]
        q_preds = predictions[mask]
        
        if len(q_labels) == 0:
            continue
        
        q_acc = accuracy_score(q_labels, q_preds)
        q_f1 = f1_score(q_labels, q_preds, average="macro", zero_division=0)
        q_pos_pct = q_labels.mean() * 100
        q_name = config.QUESTION_INFO.get(q, {}).get("response", "?")
        
        print(f"{q:>3d} {q_name:>12s} {len(q_labels):>4d} {q_acc:>6.3f} {q_f1:>6.3f} {q_pos_pct:>5.1f}%")

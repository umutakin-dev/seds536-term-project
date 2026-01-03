"""
Evaluation script for Monk Skin Tone classifier.
Provides detailed per-class metrics, confusion matrix, and fairness analysis.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from tqdm import tqdm

from .dataset import create_dataloaders, load_config, MonkSkinToneDataset, get_transforms
from .model import create_model, load_checkpoint


CLASS_NAMES = {
    "10class": [
        "Scale 1 (Lightest)",
        "Scale 2",
        "Scale 3",
        "Scale 4",
        "Scale 5",
        "Scale 6",
        "Scale 7",
        "Scale 8",
        "Scale 9",
        "Scale 10 (Darkest)"
    ],
    "3class": [
        "Light (1-3)",
        "Medium (4-7)",
        "Dark (8-10)"
    ],
    "5class": [
        "Very Light (1-2)",
        "Light (3-4)",
        "Medium (5-6)",
        "Dark (7-8)",
        "Very Dark (9-10)"
    ]
}


def get_predictions(
    model: nn.Module,
    dataloader,
    device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get predictions for entire dataset.

    Returns:
        Tuple of (all_predictions, all_labels, all_probabilities)
    """
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute comprehensive classification metrics.

    Returns:
        Dictionary with all metrics
    """
    # Overall accuracy
    overall_acc = accuracy_score(y_true, y_pred)

    # Per-class precision, recall, F1
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    # Macro and weighted averages
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Per-class accuracy
    per_class_acc = []
    for i in range(10):
        mask = y_true == i
        if mask.sum() > 0:
            class_acc = (y_pred[mask] == i).mean()
        else:
            class_acc = 0.0
        per_class_acc.append(class_acc)

    return {
        "overall_accuracy": overall_acc,
        "per_class": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1": f1.tolist(),
            "accuracy": per_class_acc,
            "support": support.tolist()
        },
        "macro": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1
        },
        "weighted": {
            "precision": weighted_precision,
            "recall": weighted_recall,
            "f1": weighted_f1
        },
        "confusion_matrix": cm.tolist()
    }


def compute_fairness_metrics(metrics: dict) -> dict:
    """
    Compute fairness-related metrics.

    Returns:
        Dictionary with fairness metrics
    """
    per_class_acc = metrics["per_class"]["accuracy"]

    # Filter out classes with no samples
    valid_acc = [a for a, s in zip(per_class_acc, metrics["per_class"]["support"]) if s > 0]

    if len(valid_acc) == 0:
        return {}

    # Performance disparity
    max_acc = max(valid_acc)
    min_acc = min(valid_acc)
    acc_range = max_acc - min_acc
    acc_std = np.std(valid_acc)

    # Worst-case ratio (min/max)
    worst_case_ratio = min_acc / max_acc if max_acc > 0 else 0

    return {
        "accuracy_range": acc_range,
        "accuracy_std": acc_std,
        "worst_case_ratio": worst_case_ratio,
        "max_accuracy": max_acc,
        "min_accuracy": min_acc,
        "target_parity": acc_range < 0.05,  # <5% difference is target
        "target_ratio": worst_case_ratio > 0.85  # Worst should be >85% of best
    }


def print_results(metrics: dict, fairness: dict, class_grouping: str = "10class"):
    """Print formatted evaluation results."""
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    class_names = CLASS_NAMES.get(class_grouping, CLASS_NAMES["10class"])
    num_classes = len(class_names)

    # Overall metrics
    print(f"\nOverall Accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro']['f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted']['f1']:.4f}")

    # Per-class breakdown
    print("\n" + "-" * 70)
    print("PER-CLASS METRICS")
    print("-" * 70)
    print(f"{'Class':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Accuracy':>10} {'Support':>10}")
    print("-" * 70)

    for i in range(num_classes):
        precision = metrics["per_class"]["precision"][i]
        recall = metrics["per_class"]["recall"][i]
        f1 = metrics["per_class"]["f1"][i]
        acc = metrics["per_class"]["accuracy"][i]
        support = metrics["per_class"]["support"][i]

        print(f"{class_names[i]:<25} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {acc:>10.4f} {support:>10}")

    # Fairness metrics
    if fairness:
        print("\n" + "-" * 70)
        print("FAIRNESS METRICS")
        print("-" * 70)
        print(f"Accuracy Range (max - min): {fairness['accuracy_range']:.4f}")
        print(f"Accuracy Std Dev: {fairness['accuracy_std']:.4f}")
        print(f"Worst-Case Ratio (min/max): {fairness['worst_case_ratio']:.4f}")
        print(f"Best Class Accuracy: {fairness['max_accuracy']:.4f}")
        print(f"Worst Class Accuracy: {fairness['min_accuracy']:.4f}")
        print(f"\nFairness Targets:")
        parity_status = "PASS" if fairness['target_parity'] else "FAIL"
        ratio_status = "PASS" if fairness['target_ratio'] else "FAIL"
        print(f"  - Accuracy parity (<5% diff): {parity_status}")
        print(f"  - Worst-case ratio (>85%): {ratio_status}")

    # Confusion matrix
    print("\n" + "-" * 70)
    print("CONFUSION MATRIX (rows=true, cols=pred)")
    print("-" * 70)
    cm = np.array(metrics["confusion_matrix"])
    print("     " + "".join([f"{i+1:>6}" for i in range(num_classes)]))
    for i in range(num_classes):
        print(f"{i+1:>3}  " + "".join([f"{cm[i,j]:>6}" for j in range(num_classes)]))

    print("\n" + "=" * 70)


def evaluate(
    config_path: str,
    checkpoint_path: str,
    split: str = "test",
    output_path: str = None
):
    """
    Main evaluation function.

    Args:
        config_path: Path to configuration file
        checkpoint_path: Path to model checkpoint
        split: Dataset split to evaluate (train/val/test)
        output_path: Optional path to save results JSON
    """
    # Load configuration
    config = load_config(config_path)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model and load weights
    print("\nLoading model...")
    model = create_model(config)
    checkpoint_info = load_checkpoint(model, checkpoint_path, device)
    model = model.to(device)

    print(f"Loaded checkpoint from epoch {checkpoint_info.get('epoch', 'unknown')}")
    if "val_acc" in checkpoint_info:
        print(f"Checkpoint val accuracy: {checkpoint_info['val_acc']:.4f}")

    # Create dataloader for specified split
    print(f"\nLoading {split} data...")
    transform = get_transforms(config, "val")  # Always use val transforms for evaluation
    class_grouping = config["data"].get("class_grouping", "10class")
    dataset = MonkSkinToneDataset(
        config["data"][f"{split}_dir"],
        transform=transform,
        class_grouping=class_grouping
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["hardware"].get("num_workers", 4),
        pin_memory=True
    )

    # Get predictions
    print(f"\nEvaluating on {len(dataset)} samples...")
    predictions, labels, probabilities = get_predictions(model, dataloader, device)

    # Compute metrics
    metrics = compute_metrics(labels, predictions)
    fairness = compute_fairness_metrics(metrics)

    # Print results
    print_results(metrics, fairness, class_grouping)

    # Save results
    if output_path:
        # Convert numpy types to Python native types for JSON serialization
        def convert_to_native(obj):
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(i) for i in obj]
            elif isinstance(obj, (np.bool_, np.integer)):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, bool):
                return obj
            return obj

        results = {
            "config": config_path,
            "checkpoint": checkpoint_path,
            "split": split,
            "class_grouping": class_grouping,
            "num_classes": dataset.num_classes,
            "num_samples": len(dataset),
            "metrics": convert_to_native(metrics),
            "fairness": convert_to_native(fairness)
        }

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_path}")

    return metrics, fairness


def main():
    parser = argparse.ArgumentParser(description="Evaluate Monk Skin Tone Classifier")
    parser.add_argument(
        "--config",
        type=str,
        default="training/configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON"
    )

    args = parser.parse_args()

    evaluate(args.config, args.checkpoint, args.split, args.output)


if __name__ == "__main__":
    main()

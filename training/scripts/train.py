"""
Main training script for Monk Skin Tone classifier.
Handles training loop, validation, checkpointing, and logging.
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from tqdm import tqdm

from .dataset import create_dataloaders, load_config
from .model import create_model, save_checkpoint


def get_device(config: dict) -> torch.device:
    """Get device based on configuration and availability."""
    device_setting = config["hardware"].get("device", "auto")

    if device_setting == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device_setting)


def create_scheduler(optimizer, config: dict, num_batches: int):
    """Create learning rate scheduler from config."""
    sched_config = config["training"]["scheduler"]
    sched_type = sched_config.get("type", "cosine")
    num_epochs = config["training"]["num_epochs"]

    if sched_type == "cosine":
        return CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif sched_type == "step":
        return StepLR(
            optimizer,
            step_size=sched_config.get("step_size", 10),
            gamma=sched_config.get("gamma", 0.1)
        )
    elif sched_type == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=sched_config.get("patience", 3),
            factor=sched_config.get("factor", 0.5)
        )
    else:
        return None


def train_one_epoch(
    model: nn.Module,
    train_loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_batches: int = None
) -> tuple[float, float]:
    """
    Train for one epoch.

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    total_batches = min(len(train_loader), max_batches) if max_batches else len(train_loader)
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False, total=total_batches)

    for batch_idx, (images, labels) in enumerate(pbar):
        if max_batches and batch_idx >= max_batches:
            break

        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100.*correct/total:.2f}%"
        })

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    max_batches: int = None
) -> tuple[float, float]:
    """
    Validate model.

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    total_batches = min(len(val_loader), max_batches) if max_batches else len(val_loader)
    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False, total=total_batches)

    for batch_idx, (images, labels) in enumerate(pbar):
        if max_batches and batch_idx >= max_batches:
            break

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def train(config_path: str, resume: str = None):
    """
    Main training function.

    Args:
        config_path: Path to configuration YAML file
        resume: Optional path to checkpoint to resume from
    """
    # Load configuration
    config = load_config(config_path)

    # Set seed for reproducibility
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Setup device
    device = get_device(config)
    print(f"Using device: {device}")

    # Create output directories
    checkpoint_dir = Path(config["output"]["checkpoint_dir"])
    log_dir = Path(config["output"]["log_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader, class_weights = create_dataloaders(config)

    # Create model
    print("\nCreating model...")
    model = create_model(config)
    model = model.to(device)

    print(f"Architecture: {model.architecture}")
    print(f"Total parameters: {model.get_num_parameters():,}")
    print(f"Trainable parameters: {model.get_num_parameters(trainable_only=True):,}")

    # Loss function with class weights
    if config["training"].get("class_weights", "none") != "none":
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"Using weighted cross-entropy loss")
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )

    # Scheduler
    scheduler = create_scheduler(optimizer, config, len(train_loader))

    # Resume from checkpoint if provided
    start_epoch = 0
    best_val_acc = 0.0

    if resume:
        print(f"\nResuming from checkpoint: {resume}")
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_acc = checkpoint.get("val_acc", 0.0)
        print(f"Resuming from epoch {start_epoch}, best val acc: {best_val_acc:.4f}")

    # Early stopping
    early_stop_config = config["training"].get("early_stopping", {})
    early_stop_enabled = early_stop_config.get("enabled", True)
    early_stop_patience = early_stop_config.get("patience", 7)
    early_stop_min_delta = early_stop_config.get("min_delta", 0.001)
    epochs_without_improvement = 0

    # Training loop
    num_epochs = config["training"]["num_epochs"]
    max_batches = config["training"].get("max_batches_per_epoch", None)
    save_frequency = config["output"].get("save_frequency", 5)

    print(f"\nStarting training for {num_epochs} epochs...")
    if max_batches:
        print(f"  (Limited to {max_batches} batches per epoch for testing)")
    print("=" * 60)

    # Log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"

    with open(log_file, "w") as f:
        f.write(f"Training started at {timestamp}\n")
        f.write(f"Config: {config_path}\n")
        f.write(f"Device: {device}\n\n")

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, max_batches
        )

        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch, max_batches
        )

        epoch_time = time.time() - epoch_start

        # Update scheduler
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        # Print epoch results
        print(f"Epoch {epoch:3d}/{num_epochs-1} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.6f} | "
              f"Time: {epoch_time:.1f}s")

        # Log to file
        with open(log_file, "a") as f:
            f.write(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                    f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, lr={current_lr:.6f}\n")

        # Save best model
        if val_acc > best_val_acc + early_stop_min_delta:
            best_val_acc = val_acc
            epochs_without_improvement = 0

            best_path = checkpoint_dir / "best_model.pth"
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_acc,
                str(best_path), config
            )
            print(f"  -> New best model saved! Val Acc: {val_acc:.4f}")
        else:
            epochs_without_improvement += 1

        # Save periodic checkpoint
        if (epoch + 1) % save_frequency == 0:
            ckpt_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_acc,
                str(ckpt_path), config
            )

        # Early stopping
        if early_stop_enabled and epochs_without_improvement >= early_stop_patience:
            print(f"\nEarly stopping triggered after {epochs_without_improvement} epochs without improvement")
            break

    print("=" * 60)
    print(f"Training completed! Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best model saved to: {checkpoint_dir / 'best_model.pth'}")
    print(f"Training log saved to: {log_file}")

    # Final evaluation on test set
    print("\nEvaluating on test set...")
    best_checkpoint = torch.load(checkpoint_dir / "best_model.pth", map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    test_loss, test_acc = validate(model, test_loader, criterion, device, -1)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    with open(log_file, "a") as f:
        f.write(f"\nFinal test results: loss={test_loss:.4f}, acc={test_acc:.4f}\n")


def main():
    parser = argparse.ArgumentParser(description="Train Monk Skin Tone Classifier")
    parser.add_argument(
        "--config",
        type=str,
        default="training/configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )

    args = parser.parse_args()

    train(args.config, args.resume)


if __name__ == "__main__":
    main()

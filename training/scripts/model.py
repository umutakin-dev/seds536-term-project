"""
Model architectures for Monk Skin Tone classification.
Supports various pretrained backbones fine-tuned for 10-class classification.
"""

import torch
import torch.nn as nn
from torchvision import models


class SkinToneClassifier(nn.Module):
    """
    Skin tone classifier using pretrained backbone.

    Supports multiple architectures with customizable dropout and head.
    """

    SUPPORTED_ARCHITECTURES = [
        "resnet18", "resnet34", "resnet50",
        "efficientnet_b0", "efficientnet_b2",
        "mobilenet_v3_small", "mobilenet_v3_large"
    ]

    def __init__(
        self,
        architecture: str = "efficientnet_b0",
        num_classes: int = 10,
        pretrained: bool = True,
        dropout: float = 0.3
    ):
        """
        Args:
            architecture: Backbone architecture name
            num_classes: Number of output classes (10 for Monk scale)
            pretrained: Whether to use pretrained weights
            dropout: Dropout rate before classifier
        """
        super().__init__()

        if architecture not in self.SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"Architecture {architecture} not supported. "
                f"Choose from: {self.SUPPORTED_ARCHITECTURES}"
            )

        self.architecture = architecture
        self.num_classes = num_classes

        # Build backbone and classifier
        self.backbone, in_features = self._create_backbone(architecture, pretrained)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(in_features, num_classes)

    def _create_backbone(self, architecture: str, pretrained: bool):
        """
        Create backbone network and return it with the number of features.

        Returns:
            Tuple of (backbone, in_features)
        """
        weights = "IMAGENET1K_V1" if pretrained else None

        if architecture == "resnet18":
            model = models.resnet18(weights=weights)
            in_features = model.fc.in_features
            model.fc = nn.Identity()

        elif architecture == "resnet34":
            model = models.resnet34(weights=weights)
            in_features = model.fc.in_features
            model.fc = nn.Identity()

        elif architecture == "resnet50":
            model = models.resnet50(weights=weights)
            in_features = model.fc.in_features
            model.fc = nn.Identity()

        elif architecture == "efficientnet_b0":
            model = models.efficientnet_b0(weights=weights)
            in_features = model.classifier[1].in_features
            model.classifier = nn.Identity()

        elif architecture == "efficientnet_b2":
            model = models.efficientnet_b2(weights=weights)
            in_features = model.classifier[1].in_features
            model.classifier = nn.Identity()

        elif architecture == "mobilenet_v3_small":
            model = models.mobilenet_v3_small(weights=weights)
            in_features = model.classifier[0].in_features
            model.classifier = nn.Identity()

        elif architecture == "mobilenet_v3_large":
            model = models.mobilenet_v3_large(weights=weights)
            in_features = model.classifier[0].in_features
            model.classifier = nn.Identity()

        return model, in_features

    def forward(self, x):
        """Forward pass through the network."""
        features = self.backbone(x)
        features = self.dropout(features)
        logits = self.classifier(features)
        return logits

    def freeze_backbone(self):
        """Freeze backbone weights for transfer learning."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone weights."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_num_parameters(self, trainable_only=False):
        """Get total number of parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


def create_model(config: dict) -> SkinToneClassifier:
    """
    Create model from configuration.

    Args:
        config: Full configuration dictionary

    Returns:
        SkinToneClassifier model
    """
    model_config = config["model"]

    model = SkinToneClassifier(
        architecture=model_config.get("architecture", "efficientnet_b0"),
        num_classes=config["data"]["num_classes"],
        pretrained=model_config.get("pretrained", True),
        dropout=model_config.get("dropout", 0.3)
    )

    return model


def load_checkpoint(model: SkinToneClassifier, checkpoint_path: str, device: str = "cpu"):
    """
    Load model weights from checkpoint.

    Args:
        model: Model instance
        checkpoint_path: Path to checkpoint file
        device: Device to load weights to

    Returns:
        Dictionary with checkpoint info (epoch, metrics, etc.)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        return {
            "epoch": checkpoint.get("epoch", 0),
            "val_loss": checkpoint.get("val_loss", None),
            "val_acc": checkpoint.get("val_acc", None)
        }
    else:
        model.load_state_dict(checkpoint)
        return {}


def save_checkpoint(
    model: SkinToneClassifier,
    optimizer,
    epoch: int,
    val_loss: float,
    val_acc: float,
    path: str,
    config: dict = None
):
    """
    Save model checkpoint.

    Args:
        model: Model instance
        optimizer: Optimizer instance
        epoch: Current epoch
        val_loss: Validation loss
        val_acc: Validation accuracy
        path: Path to save checkpoint
        config: Optional config to save
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "val_acc": val_acc,
        "architecture": model.architecture,
        "num_classes": model.num_classes,
    }

    if config:
        checkpoint["config"] = config

    torch.save(checkpoint, path)


if __name__ == "__main__":
    # Test model creation
    print("Testing model architectures...\n")

    for arch in SkinToneClassifier.SUPPORTED_ARCHITECTURES:
        model = SkinToneClassifier(architecture=arch, pretrained=False)
        total_params = model.get_num_parameters()
        trainable_params = model.get_num_parameters(trainable_only=True)

        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        out = model(x)

        print(f"{arch:20s} | Params: {total_params:,} | Output: {out.shape}")

    print("\nAll architectures working correctly!")

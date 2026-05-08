import torch
import torch.nn as nn
from torchvision import models

def get_resnet_model(num_classes=2, feature_extract=False  ):
    """
    Loads a pre-trained ResNet18 model and modifies the final layer.
    """
    # 1. Load the pre-trained weights (Industry standard method)
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    # 2. Freeze the early layers (Feature Extraction)
    # This prevents the pre-trained "vision" weights from being destroyed during training
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False

    # 3. Replace the final fully connected classification layer
    # The new layer automatically has requires_grad=True by default
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model

if __name__ == "__main__":
    model = get_resnet_model()
    print("Successfully loaded ResNet18 and replaced the classification head.")
    # Check what parameters are actually going to be trained
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}") # Should only be a few thousand, not millions!
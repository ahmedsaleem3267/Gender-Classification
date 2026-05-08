import torch
import torch.nn as nn


class CustomCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomCNN, self).__init__()

        # Block 1: Input (3, 128, 128) -> Output (16, 64, 64)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 2: Input (16, 64, 64) -> Output (32, 32, 32)
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 3: Input (32, 32, 32) -> Output (64, 16, 16)
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Fully Connected (Dense) Layers
        # After 3 MaxPools, the 128x128 image is reduced to 16x16
        # Flattened size = 64 channels * 16 height * 16 width = 16384
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.5),  # 50% probability of dropping neurons to prevent overfitting
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x


# Quick test to ensure the dimensions match up
if __name__ == "__main__":
    # Create a dummy batch of 8 images, 3 color channels, 128x128 pixels
    dummy_data = torch.randn(8, 3, 128, 128)

    # Initialize the model
    model = CustomCNN(num_classes=2)

    # Pass the data through the model
    output = model(dummy_data)

    print(f"Model architecture:\n{model}")
    print(f"\nOutput shape: {output.shape}")  # Should be [8, 2] (8 images, 2 class probabilities)
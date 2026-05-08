import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def get_data_loaders(data_dir='../data/raw', batch_size=32, img_size=(128, 128)):
    """
    Loads images from the raw data folder, applies transformations,
    and returns Training and Validation DataLoaders.
    """
    # 1. Define the transformations (Resize, Convert to Tensor, Normalize)
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Standard ImageNet normalization
                             std=[0.229, 0.224, 0.225])
    ])

    # 2. Load the dataset from the folder structure
    print(f"Loading data from: {os.path.abspath(data_dir)}")
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # 3. Split the data into 80% Training and 20% Validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 4. Create the DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Classes found: {full_dataset.classes}")
    print(f"Training images: {train_size} | Validation images: {val_size}")

    return train_loader, val_loader, full_dataset.classes


# Quick test to ensure it works when you run this file directly
if __name__ == "__main__":
    # Get the absolute path of the directory this script is in (src/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to data/raw/Training dynamically
    target_data_dir = os.path.join(current_dir, '..', 'data', 'raw', 'Training')

    train_loader, val_loader, classes = get_data_loaders(data_dir=target_data_dir)

    # Fetch one batch to verify shape
    images, labels = next(iter(train_loader))
    print(f"Batch image shape: {images.shape}")  # Should be [32, 3, 128, 128]
    print(f"Batch label shape: {labels.shape}")  # Should be [32]
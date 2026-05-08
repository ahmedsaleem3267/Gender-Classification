import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Import the specific ResNet setup function
from transfer_model import get_resnet_model


def evaluate_resnet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating ResNet18 on device: {device}")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    val_dir = os.path.join(current_dir, '..', 'data', 'raw', 'Validation')

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    classes = val_dataset.classes

    # Load the ResNet model architecture
    model = get_resnet_model(num_classes=2, feature_extract=True).to(device)

    # Target the newly saved ResNet weights
    model_path = os.path.join(current_dir, '..', 'best_resnet_model.pth')

    print(f"Loading weights from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    all_preds = []
    all_labels = []

    print("Running inference on validation data...")
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Print the Classification Report
    print("\n" + "=" * 40)
    print("RESNET18 CLASSIFICATION REPORT")
    print("=" * 40)
    print(classification_report(all_labels, all_preds, target_names=classes))

    # Generate and Save the Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',  # Changed color to Greens to distinguish from CNN
                xticklabels=classes, yticklabels=classes)

    plt.xlabel('Predicted Label', fontweight='bold')
    plt.ylabel('True Label', fontweight='bold')
    plt.title('Confusion Matrix: ResNet18 Transfer Learning', fontweight='bold', pad=20)

    cm_path = os.path.join(current_dir, '..', 'resnet_confusion_matrix.png')
    plt.savefig(cm_path, bbox_inches='tight')
    print(f"\n--> Saved ResNet confusion matrix plot to: {cm_path}")


if __name__ == "__main__":
    evaluate_resnet()
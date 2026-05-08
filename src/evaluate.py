import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import CustomCNN


def evaluate_model():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    # 2. Load the Unseen Validation Data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    val_dir = os.path.join(current_dir, '..', 'data', 'raw', 'Validation')

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # We load the Validation folder directly without splitting it
    val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    classes = val_dataset.classes

    # 3. Load the Saved Model Weights
    model = CustomCNN(num_classes=2).to(device)
    model_path = os.path.join(current_dir, '..', 'best_custom_cnn.pth')

    print(f"Loading weights from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    # CRITICAL: Put the model in evaluation mode (turns off Dropout and freezes BatchNorm)
    model.eval()

    # 4. Run Inference
    all_preds = []
    all_labels = []

    print("Running inference on validation data. This might take a minute...")
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # Save predictions and true labels for metric calculations
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 5. Print the Classification Report
    print("\n" + "=" * 40)
    print("CLASSIFICATION REPORT")
    print("=" * 40)
    print(classification_report(all_labels, all_preds, target_names=classes))

    # 6. Generate and Save the Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))

    # Create a visually appealing heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)

    plt.xlabel('Predicted Label', fontweight='bold')
    plt.ylabel('True Label', fontweight='bold')
    plt.title('Confusion Matrix: Custom CNN', fontweight='bold', pad=20)

    cm_path = os.path.join(current_dir, '..', 'confusion_matrix.png')
    plt.savefig(cm_path, bbox_inches='tight')
    print(f"\n--> Saved confusion matrix plot to: {cm_path}")


if __name__ == "__main__":
    evaluate_model()
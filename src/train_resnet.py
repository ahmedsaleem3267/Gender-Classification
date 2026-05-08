import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt

# Import data loader and the new transfer model
from data_loader import get_data_loaders
from transfer_model import get_resnet_model


def train_transfer_model(epochs=5, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training ResNet18 on device: {device}")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '..', 'data', 'raw', 'Training')

    # Portfolio Note: ResNet REQUIRES the exact ImageNet normalization [0.485, 0.456, 0.406]
    # that we already perfectly set up in data_loader.py!
    train_loader, val_loader, classes = get_data_loaders(data_dir=data_dir, batch_size=32)

    # Load ResNet
    model = get_resnet_model(num_classes=2, feature_extract=True).to(device)

    criterion = nn.CrossEntropyLoss()

    # CRITICAL: Only pass the parameters that require gradients (the new final layer) to the optimizer
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params_to_update, lr=learning_rate)

    best_val_acc = 0.0
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)
        train_acc = 100 * correct_train / total_train

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)
        val_acc = 100 * correct_val / total_val

        print(f"Epoch [{epoch + 1}/{epochs}] "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(current_dir, '..', 'best_resnet_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved new best model with Val Acc: {val_acc:.2f}%")

        # Plotting
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, epoch + 2), train_losses, label='Train Loss', color='blue', marker='o')
        plt.plot(range(1, epoch + 2), val_losses, label='Validation Loss', color='red', marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('ResNet18 Transfer Learning: Training vs Validation Loss')
        plt.legend()
        plt.grid(True)

        plot_path = os.path.join(current_dir, '..', 'resnet_loss_curve.png')
        plt.savefig(plot_path)
        plt.close()


if __name__ == "__main__":
    # Because we are only training the final layer, ResNet converges incredibly fast.
    # 3 to 5 epochs is usually plenty.
    train_transfer_model(epochs=5, learning_rate=0.0001)
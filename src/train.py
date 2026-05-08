import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt  # Added for plotting

# Import your custom modules
from data_loader import get_data_loaders
from model import CustomCNN


def train_model(epochs=10, learning_rate=0.001):
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '..', 'data', 'raw', 'Training')

    train_loader, val_loader, classes = get_data_loaders(data_dir=data_dir, batch_size=32)

    model = CustomCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_acc = 0.0

    # Initialize lists to track metrics for plotting
    train_losses = []
    val_losses = []

    # 4. The Training Loop
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

        # Calculate epoch metrics
        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)
        train_acc = 100 * correct_train / total_train

        # 5. The Validation Loop
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

        # 6. Save the Best Model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(current_dir, '..', 'best_custom_cnn.pth')
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved new best model with Val Acc: {val_acc:.2f}%")

        # 7. Dynamically Generate the Plot
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, epoch + 2), train_losses, label='Train Loss', color='blue', marker='o')
        plt.plot(range(1, epoch + 2), val_losses, label='Validation Loss', color='red', marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Custom CNN: Training vs Validation Loss')
        plt.legend()
        plt.grid(True)

        # Save the plot (overwrites the previous one each epoch)
        plot_path = os.path.join(current_dir, '..', 'loss_curve.png')
        plt.savefig(plot_path)
        plt.close()  # Close the figure to free up memory


if __name__ == "__main__":
    train_model(epochs=5)
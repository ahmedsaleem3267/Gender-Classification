import torch
import os
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

# Import the ResNet builder instead of the Custom CNN
from transfer_model import get_resnet_model


def predict_single_image_resnet(image_path):
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Define the EXACT same transformations used in training
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. Load and transform the image
    if not os.path.exists(image_path):
        print(f"Error: Could not find image at {image_path}")
        return

    # Convert to RGB to ensure 3 channels
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)

    # Add a batch dimension! [3, 128, 128] becomes [1, 3, 128, 128]
    image_batch = image_tensor.unsqueeze(0).to(device)

    # 4. Load the ResNet Model Architecture
    # Note: feature_extract doesn't matter for inference since we don't calculate gradients,
    # but we pass False here if you fine-tuned the whole model in the last step.
    model = get_resnet_model(num_classes=2, feature_extract=False).to(device)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, '..', 'best_resnet_model.pth')

    print(f"Loading weights from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    # CRITICAL: Put model in evaluation mode
    model.eval()

    # 5. Make the Prediction
    classes = ['female', 'male']

    with torch.no_grad():
        output = model(image_batch)

        # Convert raw output logits to percentages using Softmax
        probabilities = F.softmax(output, dim=1)

        confidence, predicted_idx = torch.max(probabilities, 1)

        predicted_class = classes[predicted_idx.item()]
        confidence_score = confidence.item() * 100

    print("\n" + "=" * 40)
    print("RESNET18 PREDICTION")
    print("=" * 40)
    print(f"Image File: {os.path.basename(image_path)}")
    print(f"Prediction: {predicted_class.upper()}")
    print(f"Confidence: {confidence_score:.2f}%")
    print("=" * 40)


if __name__ == "__main__":
    # Point this to the exact same image you tested with the CNN
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_image = os.path.join(current_dir, '..', 'data', 'test_images', '1.jfif')

    predict_single_image_resnet(target_image)
import torch
from PIL import Image
from torchvision import transforms
from model import LightweightFingerprintNet  # Import the model class

# Define transformations (same as during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to ResNet input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize; same as during training
])

# Function to load the model
def load_model(model_path, device):
    model = LightweightFingerprintNet(num_classes=2, transformer_layers=2, d_model=256, nhead=8)
    model.load_state_dict(torch.load(model_path, map_location=device))  # Load the saved model weights
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

# Function to predict whether the fingerprint is real or fake
def predict_fingerprint(model, img_path, device):
    # Open the image file and apply transformations
    image = Image.open(img_path).convert('RGB')  # Open the image and convert it to RGB
    image = transform(image).unsqueeze(0).to(device)  # Apply transformation and add batch dimension

    # Make a prediction
    with torch.no_grad():  # No need to track gradients during inference
        output = model(image)  # Forward pass
        _, predicted = torch.max(output, 1)  # Get the predicted class (0 or 1)

    # Return the prediction
    if predicted.item() == 1:
        return "Live"
    else:
        return "Spoof"

# Main function to run the detection
if __name__ == "__main__":
    # Set the device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model
    model = load_model('fingerprint_liveness_model.pth', device)
    
    # Path to the input fingerprint image
    img_path = "103_1.tif"  # Replace with your image path
    
    # Make the prediction
    result = predict_fingerprint(model, img_path, device)
    
    # Print the result
    print(f"The fingerprint is: {result}")

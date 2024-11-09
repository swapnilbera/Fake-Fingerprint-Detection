import torch
import torch.nn as nn
import torch.optim as optim
from model import LightweightFingerprintNet
from dataset import load_data

# Load training data
train_loader = load_data(root_dir='dataset/', batch_size=32)

# Instantiate model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LightweightFingerprintNet(num_classes=2, transformer_layers=2, d_model=256, nhead=8).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(10):  # Adjust number of epochs as needed
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}')

# Save the trained model
torch.save(model.state_dict(), "fingerprint_liveness_model.pth")

import torch
import torch.nn as nn
import torchvision.models as models

# Define MHSA Module for multi-head self-attention
class MHSA(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MHSA, self).__init__()
        self.mhsa = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        x = x.permute(1, 0, 2)  # Reshape for MultiheadAttention (seq_len, batch, embed_dim)
        x, _ = self.mhsa(x, x, x)
        return x.permute(1, 0, 2)  # Reshape back (batch, seq_len, embed_dim)

# Define the ResNet-Transformer based model
class LightweightFingerprintNet(nn.Module):
    def __init__(self, num_classes=2, transformer_layers=2, d_model=256, nhead=8):
        super(LightweightFingerprintNet, self).__init__()
        
        # Load a lightweight ResNet (ResNet-18) and modify for feature extraction
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()  # Remove the original classification layer
        self.resnet.layer4 = nn.Identity()  # Remove the last block for lightweight design
        
        # Add the MHSA module
        self.mhsa = MHSA(embed_dim=d_model, num_heads=nhead)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # Classification head
        self.fc = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        x = self.resnet(x)
        x = self.mhsa(x.unsqueeze(1))  # Add sequence dimension
        x = self.transformer_encoder(x).squeeze(1)
        x = self.fc(x)
        return x

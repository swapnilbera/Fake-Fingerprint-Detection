import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Define transformations for LivDet images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to ResNet input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize; adjust if needed
])

class LivDet2017Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        for label_type in ['live', 'spoof']:
            label = 1 if label_type == 'live' else 0
            folder = os.path.join(root_dir, label_type)
            for img_file in os.listdir(folder):
                self.image_paths.append(os.path.join(folder, img_file))
                self.labels.append(label)
                
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # Convert to RGB if needed
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def load_data(root_dir, batch_size=32):
    dataset = LivDet2017Dataset(root_dir=root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

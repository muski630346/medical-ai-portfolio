import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from PIL import Image
import os

# ─── 1. LOAD LABELS ───────────────────────────────────────────
df = pd.read_csv(r'C:\Users\shaik\OneDrive\Desktop\medical-ai-portfolio\medical-ai-portfolio\Data_Entry_2017.csv')
print(f"Total images in CSV: {len(df)}")

# ─── 2. DISEASE CLASSES ───────────────────────────────────────
CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
    'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia'
]

# ─── 3. ENCODE LABELS ─────────────────────────────────────────
def encode_labels(finding):
    label = np.zeros(14)
    for i, cls in enumerate(CLASSES):
        if cls in finding:
            label[i] = 1
    return label

df['encoded'] = df['Finding Labels'].apply(encode_labels)
print("Labels encoded successfully!")

# ─── 4. FILTER TO EXISTING IMAGES ONLY ───────────────────────
image_dir = r'C:\Users\shaik\OneDrive\Desktop\medical-ai-portfolio\images'

print("Filtering to available images...")
df['exists'] = df['Image Index'].apply(
    lambda x: os.path.exists(os.path.join(image_dir, x))
)
df = df[df['exists'] == True].reset_index(drop=True)
print(f"Available images for training: {len(df)}")

# ─── 5. DATASET CLASS ─────────────────────────────────────────
class ChestXrayDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.df.iloc[idx]['Image Index'])
        image = Image.open(img_path).convert('RGB')
        label = torch.FloatTensor(self.df.iloc[idx]['encoded'])
        if self.transform:
            image = self.transform(image)
        return image, label

# ─── 6. TRANSFORMS ────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ─── 7. SPLIT DATA ────────────────────────────────────────────
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

train_dataset = ChestXrayDataset(train_df, image_dir, transform)
val_dataset = ChestXrayDataset(val_df, image_dir, transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")

# ─── 8. LOAD RESNET50 MODEL ───────────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 14)
model = model.to(device)
print("ResNet50 model loaded!")

# ─── 9. LOSS AND OPTIMIZER ────────────────────────────────────
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ─── 10. TRAINING LOOP ────────────────────────────────────────
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=3):
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_train_loss = running_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"\n✓ Epoch {epoch+1} Complete!")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}\n")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'chest_xray_model.pth')
            print(f"  ✅ Best model saved! Val Loss: {best_val_loss:.4f}\n")

    return model

print("\n🚀 Starting training...")
print("=" * 50)
model = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=3)
print("=" * 50)
print("✅ Training complete! Model saved as chest_xray_model.pth")
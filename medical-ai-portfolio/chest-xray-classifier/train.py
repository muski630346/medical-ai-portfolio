import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import os


df = pd.read_csv(r'C:\Users\shaik\OneDrive\Desktop\medical-ai-portfolio\medical-ai-portfolio\Data_Entry_2017.csv')
print(f"Total images: {len(df)}")
print(df['Finding Labels'].value_counts().head(10))


CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
    'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia'
]


def encode_labels(finding):
    label = np.zeros(14)
    for i, cls in enumerate(CLASSES):
        if cls in finding:
            label[i] = 1
    return label

df['encoded'] = df['Finding Labels'].apply(encode_labels)
print("Labels encoded successfully!")
print(df.head())
# 4. Dataset Class
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

# 5. Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 6. Load dataset
image_dir = r'C:\Users\shaik\OneDrive\Desktop\medical-ai-portfolio\images'
dataset = ChestXrayDataset(df, image_dir, transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

print(f"Dataset ready! Total batches: {len(loader)}")
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import datasets, transforms 
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader, random_split, Dataset
from PIL import Image
from collections import Counter
import pandas as pd

"""
dataset = ImageDataset(df, data_folder, transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True) #if this used, comment out train and test loader
def calculate_mean_std(loader):
    mean = 0.0
    std = 0.0
    total_images = 0
    
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples
    
    mean /= total_images
    std /= total_images
    
    return mean.tolist(), std.tolist()

dataset_mean, dataset_std = calculate_mean_std(loader)
print(f"\nCalculated mean: {dataset_mean}")
print(f"Calculated std: {dataset_std}")
"""
class ImageDataset(Dataset):        # retrieving the class labels from the csv file 
    def __init__(self, csv_df, img_folder, transform=None):
        self.df = csv_df
        self.img_folder = img_folder
        self.transform = transform

        # Create a mapping from class names to indices - so the model can work with it
        self.class_mapping = {name: idx for idx, name in enumerate(sorted(self.df['label'].unique()))}

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]  # Get filename
        label_name = self.df.iloc[idx, 1]  # Get class label
        img_path = os.path.join(self.img_folder, img_name)
        image = Image.open(img_path).convert("RGB")  # Open image
        if self.transform:
            image = self.transform(image)

        label = self.class_mapping[label_name]  # Convert label to number
        return image, torch.tensor(label, dtype=torch.long)  # Ensure it's a tensor
    

def compute_mean_std(img_dir):
    transform = transforms.Compose([
        #transforms.Grayscale(num_output_channels=3),  # Optional: if images are grayscale
        transforms.Resize(256),  # Resize to match ResNet input
        transforms.ToTensor()
    ])
    
    dataset = ImageDataset(csv_df, img_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    mean = 0.
    std = 0.
    total_images = 0

    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples

    mean /= total_images
    std /= total_images
    return mean.tolist(), std.tolist()

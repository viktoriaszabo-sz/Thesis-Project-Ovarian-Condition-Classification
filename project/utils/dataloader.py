"""-------------------------------------------------------------------------------------------------------
- create some data visualization 

----------------------------------------------------------------------------------------------------------"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch_directml # Use DirectML backend
from torch.optim import lr_scheduler
from torchvision import transforms 
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torch.utils.data import DataLoader, random_split, Dataset
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from bayes_opt import BayesianOptimization

#-----------------------------------------------------------------------------------------------------------
#Parameters
#device = torch_directml.device() #causes memory leak on AMD
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_classes = 6
batch_size = 32     #maybe 64?

#-----------------------------------------------------------------------------------------------------------
#DATALOADING
#weights = DenseNet161_Weights   #this will help us with a pre-built image transformation 
weights=MobileNet_V2_Weights.IMAGENET1K_V1
#use transfer learning and pre-trained models
model = mobilenet_v2(weights=weights).to(device) 
print(model)

data_folder = '.\data\_filtered_ovary_diseases\images'
csv_file = '.\data\_filtered_ovary_diseases\_annotations.csv'
df = pd.read_csv(csv_file) #columns: filename, label

transform_aug = transforms.Compose([
    transforms.Resize(256),  # resize for resnet50 settings
    transforms.CenterCrop(224),
    transforms.RandomRotation(10),  # slight rotations help the model generalize
    transforms.RandomHorizontalFlip(),  # augmentation only for training
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Minor translation variation
    transforms.ToTensor(),  
    #transforms.Normalize(mean=mean, std=std)  # no norm, i think the best: test image nim/max:  tensor(0., device='cuda:0') tensor(0.9922, device='cuda:0')
    ])

transform = transforms.Compose([
    transforms.Resize(256),  
    transforms.CenterCrop(224),
    transforms.ToTensor(),  #no augmentation, bc this one will be used for testing
    #transforms.Normalize(mean=mean, std=std) # no norm
    ])

class ImageDataset(Dataset):        # retrieving the class labels from the csv file 
    def __init__(self, csv_df, img_folder, transform=None):
        self.df = csv_df
        self.img_folder = img_folder
        self.transform = transform
        # mapping from class names to indices - so the model can work with it
        self.class_mapping = {name: idx for idx, name in enumerate(sorted(self.df['label'].unique()))}

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]  # get filename
        label_name = self.df.iloc[idx, 1]  # get class label
        img_path = os.path.join(self.img_folder, img_name)
        image = Image.open(img_path).convert("RGB")  # Open image
        if self.transform:
            image = self.transform(image)

        label = self.class_mapping[label_name]  # convert label to number
        return image, torch.tensor(label, dtype=torch.long) 
    




    
dataset = ImageDataset(df, data_folder, transform=transform_aug)
#print(dataset.class_mapping)        #debug to see the classes 

train_size = int(0.8 * len(dataset))        #later to be redivided based on hyperparameter optimization
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size]) #each sets should have all classes 
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

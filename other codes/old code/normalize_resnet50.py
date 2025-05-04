import os
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import datasets, transforms 
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader, random_split, Dataset
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
import pandas as pd
from bayes_opt import BayesianOptimization

#-----------------------------------------------------------------------------------------------------------
#Hyperparameters - they are not defined inside ResNet, but set when training the model 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #if GPU available, otherwise falls back to CPU - can be removed 

batch_size = 64     #even if its 61, its beter to work with 2^n numbers  
num_classes = 9
lr = 0.05554
step_size = 7
epochs = 12
dropout = 0.3282 #not in resnet by default but helps with normalization 
#-----------------------------------------------------------------------------------------------------------
#DATALOADING
weights = ResNet50_Weights.IMAGENET1K_V2    #this will help us with a pre-built image transformation - should have 80.86% accuracy
#use transfer learning and pre-trained models
model = resnet50(weights=weights).to(device) #u can set the weight urself if needed 

data_folder = './data/ovary_diseases/images'
csv_file = './data/ovary_diseases/_annotations1.csv'
df = pd.read_csv(csv_file)  # Columns: filename, label
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
shuffled_csv_file = './data/ovary_diseases/_annotations1_shuffled.csv'   
df_shuffled.to_csv(shuffled_csv_file, index=False)

class ImageDataset(Dataset):
    def __init__(self, csv_df, img_folder, transform=None):
        self.df = csv_df
        self.img_folder = img_folder
        self.transform = transform
        self.class_mapping = {name: idx for idx, name in enumerate(sorted(self.df['label'].unique()))}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]
        label_name = self.df.iloc[idx, 1]
        img_path = os.path.join(self.img_folder, img_name)
        image = Image.open(img_path).convert("RGB")  # Ensure RGB

        if self.transform:
            image = self.transform(image)

        label = self.class_mapping[label_name]
        return image, torch.tensor(label, dtype=torch.long)

temp_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()  # No normalization yet
])

temp_dataset = ImageDataset(df_shuffled, data_folder, transform=temp_transform)
loader = DataLoader(temp_dataset, batch_size=32, shuffle=False)

# 4. Compute Mean and Std
def compute_mean_std(loader):
    mean = torch.zeros(3)
    std = torch.zeros(3)
    num_samples = 0

    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, 3, -1)  # Flatten images
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        num_samples += batch_samples

    mean /= num_samples
    std /= num_samples
    return mean, std

custom_mean, custom_std = compute_mean_std(loader)
print(f"Custom Mean: {custom_mean}, Custom Std: {custom_std}")

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=custom_mean.tolist(), std=custom_std.tolist())  # Custom normalization
])

dataset = ImageDataset(df_shuffled, data_folder, transform=transform)
print(dataset.class_mapping)  # Debug class mappings
img, label = dataset[0]
plt.imshow(img.permute(1, 2, 0).numpy())  # Convert (C, H, W) to (H, W, C)
plt.show()              # this is to see how the image processing was done before feeding it into the network 
                        # based on this, tune the transformation 

train_size = int(0.8 * len(dataset))        #later to be redivided based on hyperparameter optimization
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size]) #each sets should have all classes 
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#-----------------------------------------------------------------------------------------
#initialize model with corresponding weights - ResNet50 API 


for param in model.parameters():        #this will freeze all the layers, so i dont retrain the whole model later
    param.requires_grad = False

num_ftrs = model.fc.in_features #we modify the final layer here 
model.fc = nn.Linear(num_ftrs, num_classes)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr) 
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)  #might be useless? 

#-----------------------------------------------------------------------------------------
# Hyperparameter optimization

# Define an evaluation function for Bayesian Optimization
def obj_function(dropout, lr, epochs): #important to do it after the model is set to eval() mode 
    #batch_size = int(batch_size)  # Convert batch_size to int
    epochs = int(epochs)  # Convert epochs to int
    # Define loss function and optimizer
    #criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)    #updates model weights 

    for module in model.modules(): 
        if isinstance(module, nn.Dropout): 
            module.p = dropout    

    model.train()   #it will only train the fc layer instead of the whole structure 
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # Send to GPU if available
            
            optimizer.zero_grad()  # reset gradients 
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update model weights
            
            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}')

    model.eval()
    with torch.no_grad():   # Disable gradient computation for faster evaluation
        n_correct = 0
        n_samples = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)  # can be deleted? 
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)        #or torch.argmax / softmax
            n_samples += labels.size(0)                     #all samples
            n_correct += (predicted == labels).sum().item() #counts how many predictions were correct
    acc = 100.0 * n_correct / n_samples                 #accuracy score = correct prediciton / all samples 
    print(acc)
    return acc 

pbounds = {'dropout': (0.3, 0.499), 
           'lr': (0.04, 0.1), #learning_rate - 0.1 might be too high, 0.001 is enough) 
           #'batch_size': (4, 64), 
           #'epochs': (10, 15),
           }

bayes_optimizer = BayesianOptimization(
    f=obj_function,   #model evaluation is already provided in pytorch 
    pbounds=pbounds, 
    verbose = 2, 
)
obj_function(dropout, lr, epochs)
"""
start_time = time.time()
bayes_optimizer.maximize(init_points=2, n_iter=2)   #so it runs fasta 
time_took = time.time() - start_time

print (f"Total runtime: {time_took}")
print(bayes_optimizer.max)
"""
def show_random_predictions(model, test_loader, class_mapping, num_images=5):
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images[:num_images], labels[:num_images]
    outputs = model(images.to(device))
    _, preds = torch.max(outputs, 1)
    idx_to_class = {v: k for k, v in class_mapping.items()}
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    
    for idx in range(num_images):
        ax = axes[idx]
        img = images[idx].cpu().permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        pred_class = idx_to_class[preds[idx].item()]
        true_class = idx_to_class[labels[idx].item()]
        ax.set_title(f"Pred: {pred_class}\nGT: {true_class}")
        ax.axis('off')
    plt.show()
show_random_predictions(model, test_loader, dataset.class_mapping)

#-----------------------------------------------------------------------------------------------------------

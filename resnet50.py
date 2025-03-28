"""-------------------------------------------------------------------------------------------------------
STEPS OF THE CODE: 

2) dataloading 
4) training the model (? maybe retraining?)
5) testing the model
6) final prediction/testing with random image  

- need to figure out some hypertuning methods, bc the loss is good, but doesnt classify correctly 
- create some data visualization 
----------------------------------------------------------------------------------------------------------"""

import os
import torch
from torchvision import datasets, transforms 
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader, random_split, Dataset
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
import pandas as pd

#-----------------------------------------------------------------------------------------------------------
#Hyperparameters - they are not defined inside ResNet, but set when training the model 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #if GPU available, otherwise falls back to CPU - can be removed 

image_size = 128
batch_size = 32
num_classes = 9

""""
learning rate (lr) 
optimizer 
momentum (used only in sgd) 
weight_decay (if overfitting, set this) 
epochs
dropout (not in resnet by default) 
criterion (loss function) - typically CrossEntropyLoss()
"""

#-----------------------------------------------------------------------------------------------------------
#initialize model with corresponding weights - ResNet50 API 
weights = ResNet50_Weights.IMAGENET1K_V1    #this will help us with a pre-built image transformation 
model = resnet50(weights=weights).to(device) #u can set the weight urself if needed 
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)  # 7 is the number of your classes
model = model.to(device)

#potimizer and loss function here? 

#-----------------------------------------------------------------------------------------------------------
#DATALOADING

preprocess = weights.transforms()
data_folder = './data/ovary_diseases/images'
csv_file = './data/ovary_diseases/_annotations1.csv'
df = pd.read_csv(csv_file) #columns: filename, label

# retrieving the class labels from the csv file 
class ImageDataset(Dataset):
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
    
# Create dataset and print out class labels 
dataset = ImageDataset(df, data_folder, transform=preprocess)
#labels = [label for _, label in dataset]    # Debug: Check class distribution
#print("Class distribution:", Counter(labels))
print(dataset.class_mapping)        #debug to see the classes 

train_size = int(0.8 * len(dataset))        #later to be redivided based on hyperparameter optimization
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size]) #each sets should have all classes 
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


#-----------------------------------------------------------------------------------------

model.eval()
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)  # can be deleted? 
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)        #or torch.argmax / softmax
        n_samples += labels.size(0)                     #all samples
        n_correct += (predicted == labels).sum().item() #counts how many predictions were correct
    acc = 100.0 * n_correct / n_samples                 #accuracy score = correct prediciton / all samples 
    print(f'Accuracy on the test dataset: {acc:.2f}%') 

#-----------------------------------------------------------------------------------------------------------
# Prediction / testing - most of this part is just debugging 
img_path = './data/ovary_diseases/clean_ovaries.jpg' 
img = Image.open(img_path).convert("RGB")   #do i need RGB? 
img = preprocess(img).unsqueeze(0).to(device)   

print("test image nim/max: ", img.min(), img.max()) #resnet50 pre-defined weight allow them to be between -2 and 2


with torch.no_grad():
    pred = model(img)
    probabilities = torch.nn.functional.softmax(pred, dim=1)  # debug - Converts logits to probabilities
    pred_label = torch.argmax(probabilities, dim=1) #debug
    
    predicted_class = list(dataset.class_mapping.keys())[list(dataset.class_mapping.values()).index(pred_label.item())]
    print(f"Predicted label index: {pred_label.item()}")  #debug
    print("Predicted class name: ", predicted_class)

# De-normalize for Display (ResNet50 expects the input to be in a certain range for display)
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)  # ImageNet mean
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)  # ImageNet std
img_denorm = img.squeeze(0).cpu() * std + mean  # Reverse normalization
img_denorm = img_denorm.clamp(0, 1)  # Ensure values are within [0, 1] for display


plt.figure()    # Plotting of image at the end
plt.title(f"Predicted Label: {predicted_class}")
img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
img = (img - img.min()) / (img.max() - img.min())  # Normalize for display
plt.imshow(img)
plt.xticks([])
plt.yticks([])
plt.savefig(f"./prediction_of_image.png", dpi=300)
plt.show()



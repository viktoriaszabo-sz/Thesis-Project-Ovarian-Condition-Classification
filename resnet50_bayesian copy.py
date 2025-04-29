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
import time
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
from tqdm import tqdm

from bayes_opt import BayesianOptimization

#-----------------------------------------------------------------------------------------------------------
#Hyperparameters - they are not defined inside ResNet, but set when training the model 

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #if GPU available, otherwise falls back to CPU - can be removed 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 32     #maybe 64?
num_classes = 6
lr = 0.01055
step_size = 30 #image_num / batch size # maybe try 100 later
epochs = 1 # SET IT TO 1 FOR TESTING THE FINAL IMAGE INPUT - LETS SAVE TIME HERE 
dropout = 0.3361 #not in resnet by default but helps with normalization 
#62.59% accuracy - 60.65 - 0.3282 - 12.23 - 0.05554
#                 batch   dropout   epochs      lr

#+ data augmentation! 

#-----------------------------------------------------------------------------------------------------------
#DATALOADING
weights = ResNet50_Weights.IMAGENET1K_V2    #this will help us with a pre-built image transformation - should have 80.86% accuracy
#use transfer learning and pre-trained models
model = resnet50(weights=weights).to(device) #u can set the weight urself if needed 

#model.load_state_dict(torch.load("chexnet.pth")) - chest xray code weight file 

transform = transforms.Compose([
    #transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3-channel (ResNet requires RGB)
    #transforms.Lambda(lambda img: img if img.mode == 'RGB' else img.convert('RGB')),  # Ensure RGB mode
    transforms.Resize(256),  # Resize to match ResNet input
    transforms.CenterCrop(224),
    #transforms.RandomRotation(10),  # Slight rotations help the model generalize
    #transforms.RandomHorizontalFlip(),  # Ultrasounds may not be direction-sensitive
    #transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Minor translation variation
    transforms.ToTensor(),  # Convert to tensor
    #transforms.Normalize(mean=custom_mean.tolist(), std=custom_std.tolist())  # Custom normalization
    ])

data_folder = '.\data\_filtered_ovary_diseases\images'
csv_file = '.\data\_filtered_ovary_diseases\_annotations.csv'
df = pd.read_csv(csv_file) #columns: filename, label

df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)     # Shuffle the rows of the dataframe
shuffled_csv_file = '.\data\_filtered_ovary_diseases\_annotations_shuffled.csv'      # maybe use StratifiedShuffleSplit? 
df_shuffled.to_csv(shuffled_csv_file, index=False)

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
    
dataset = ImageDataset(df_shuffled, data_folder, transform=transform)
print(dataset.class_mapping)        #debug to see the classes 

# to get one sample in the beginning to see what the model is working with after the transformation is applied 
img, label = dataset[0]  # Get one sample
plt.imshow(img.permute(1, 2, 0).numpy())  # Convert (C, H, W) to (H, W, C) for visualization
#plt.title(f"Class: {label}")
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
# Hyperparameter optimization inside the training function

# Define an evaluation function for Bayesian Optimization
def obj_function(dropout, lr, epochs, batch_size): #important to do it after the model is set to eval() mode 
    batch_size = int(batch_size)  # Convert batch_size to int
    epochs = int(epochs)  # Convert epochs to int
    # Define loss function and optimizer
    #criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)    #updates model weights 

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) #do i need to reload the data here?

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

pbounds = {'dropout': (0.2, 0.499), 
           'lr': (0.0005, 0.1), #learning_rate - 0.1 might be too high, 0.001 is enough) 
           'batch_size': ([32, 64]), 
           'epochs': (10, 20),
           }

bayes_optimizer = BayesianOptimization(
    f=obj_function,   #model evaluation is already provided in pytorch 
    pbounds=pbounds, 
    verbose = 2, 
)
#----------------------------------------------------------------------------------------
#initializing the actual program

obj_function(dropout, lr, epochs, batch_size)
"""
#bayesian initialization: 
start_time = time.time()
bayes_optimizer.maximize(init_points=5, n_iter=10)   #so it runs fasta 
time_took = time.time() - start_time
print (f"Total runtime: {time_took}")
print(bayes_optimizer.max)
"""

def show_random_predictions(model, test_loader, class_mapping, num_images=5): 
    #maybe i need one from each class 
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

# Prediction / testing for fed in image (for final UI) - most of this part is just debugging 

img_path = '.\data\_filtered_ovary_diseases\simple_cyst.jpg'
img = Image.open(img_path).convert("RGB")   #do i need RGB? 
img = transform(img).unsqueeze(0).to(device)   

print("test image nim/max: ", img.min(), img.max()) #resnet50 pre-defined weight allow them to be between -2 and 2

with torch.no_grad():
    pred = model(img)
    probabilities = torch.nn.functional.softmax(pred, dim=1)  # debug - Converts logits to probabilities
    pred_label = torch.argmax(probabilities, dim=1) #debug
    
    predicted_class = list(dataset.class_mapping.keys())[list(dataset.class_mapping.values()).index(pred_label.item())]
    print(f"Predicted label index: {pred_label.item()}")  #debug
    print("Predicted class name: ", predicted_class)


"""
#do i need this? 
# De-normalize for Display (ResNet50 expects the input to be in a certain range for display)
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)  # ImageNet mean
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)  # ImageNet std
img_denorm = img.squeeze(0).cpu() * std + mean  # Reverse normalization
img_denorm = img_denorm.clamp(0, 1)  # Ensure values are within [0, 1] for display
"""


plt.figure()    # Plotting of image at the end
plt.title(f"Predicted Label: {predicted_class}")
img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
img = (img - img.min()) / (img.max() - img.min())  # Normalize for display
plt.imshow(img)
plt.xticks([])
plt.yticks([])
plt.savefig(f".\prediction_of_image.png", dpi=300)
plt.show()


"""
# Apply Transformations if normalization is used on the images - so it can be displayed for final test 
# the normalization done in the begining when feeding into resnet has to be reverted so plt can display 
# the og image 

transformed_image = transform(img)

# Convert Tensor back to Image for Visualization
def tensor_to_image(tensor):
    inv_normalize = transforms.Normalize(
        mean=[-m/s for m, s in zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],
        std=[1/s for s in [0.229, 0.224, 0.225]]
    )
    unnormalized_tensor = inv_normalize(tensor)  # Undo normalization
    unnormalized_tensor = unnormalized_tensor.clamp(0, 1)  # Clip values to valid range
    return unnormalized_tensor.permute(1, 2, 0).numpy()  # Convert to HWC format

# Plot Original vs. Processed Image
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(img)
axes[0].set_title("Original Image") 
axes[0].axis("off")

axes[1].imshow(tensor_to_image(transformed_image))
axes[1].set_title("Transformed Image (Preprocessed)")
axes[1].axis("off")

plt.show()
"""

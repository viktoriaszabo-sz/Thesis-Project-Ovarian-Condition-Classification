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
import sys
from pathlib import Path

# Get the absolute path of the project root
project = Path(__file__).parents[1]  # Adjust based on your actual structure
sys.path.append(str(project))

#from utils import dataloading # + other models 
from utils.dataloading import transform_aug, transform, dataset, test_dataset, train_dataset, device

#model = mobilenet_v2(num_classes=6)
#model.load_state_dict(torch.load('mobilenet_model.pth', map_location='cpu'))
#model.eval()

#-----------------------------------------------------------------------------------------------------------
#Parameters
num_classes = 6
step_size = 30 #image_num / batch size 

#hyperparameters
batch_size = 32     #maybe 64?
lr = 0.01055
epochs = 20 #og 20  
dropout = 0.3361 #not in resnet by default but helps with normalization 

#-----------------------------------------------------------------------------------------------------------
#DATALOADING
#weights = DenseNet161_Weights   #this will help us with a pre-built image transformation 
weights=MobileNet_V2_Weights.IMAGENET1K_V1
#use transfer learning and pre-trained models
model = mobilenet_v2(weights=weights).to(device) 
#print(model)

"""
# to get one sample in the beginning to see what the model is working with after the transformation is applied 
img, label = dataset[0]  # Get one sample
plt.imshow(img.permute(1, 2, 0).numpy())  # Convert (C, H, W) to (H, W, C) for visualization
#plt.title(f"Class: {label}")
plt.show()              # this is to see how the image processing was done before feeding it into the network 
                        # based on this, tune the transformation 
"""

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#-----------------------------------------------------------------------------------------
#initialize model with corresponding weights - densenet

for param in model.parameters():        #this will freeze all the layers, so i dont retrain the whole model later
    param.requires_grad = False 

#num_ftrs = model.classifier.in_features #we modify the final layer here 
model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr) 
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1) 

#-----------------------------------------------------------------------------------------
# Hyperparameter optimization inside the training function

#if __name__ == "__main__":
    #evaluation function for Bayesian Optimization
def obj_function(dropout, lr, epochs, batch_size): 
    batch_size = int(batch_size)
    epochs = int(epochs) 

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) #do i need to reload the data here?

    for module in model.modules(): 
        if isinstance(module, nn.Dropout): 
            module.p = dropout    

    start_time = time.time()
    model.train()   #it will only train the fc layer instead of the whole structure 
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device) 
            
            optimizer.zero_grad()  # reset gradients 
            outputs = model(images)  # forward pass
            loss = criterion(outputs, labels)  # calculate loss
            loss.backward()  # backward pass
            optimizer.step()  # update model weights
            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}')
    end_time = time.time()
    total_time = end_time - start_time

    print(f"\nTraining completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

    model.eval()
    with torch.no_grad():   # disable gradient computation for faster evaluation
        n_correct = 0
        n_samples = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device) 
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  #or torch.argmax / softmax
            n_samples += labels.size(0)           #all samples
            n_correct += (predicted == labels).sum().item() #counts how many predictions were correct
    acc = 100.0 * n_correct / n_samples                 #accuracy score = correct prediciton / all samples 
    #confidence score here?
    print(acc)
    torch.save(model.state_dict(), '.\models\mobilenet_model.pth')
    return acc 

#torch.save(model.state_dict(), 'mobilenet_model.pth')


pbounds = {'dropout': (0.2, 0.499), 
        'lr': (0.0005, 0.1), 
        'batch_size': ([32, 64]), 
        'epochs': (10, 20),
        }

bayes_optimizer = BayesianOptimization(
    f=obj_function, 
    pbounds=pbounds, 
    verbose = 2, 
)
#----------------------------------------------------------------------------------------
#initializing the actual program

if __name__ == "__main__":
    obj_function(dropout, lr, epochs, batch_size)





""" # bayesian method - uncomment if needed
#bayesian initialization: 
start_time = time.time()
bayes_optimizer.maximize(init_points=5, n_iter=10)   #so it runs fasta 
time_took = time.time() - start_time
print (f"Total runtime: {time_took}")
print(bayes_optimizer.max)
"""
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
        ax.set_title(f"Pred: {pred_class}\nGT: {true_class}") # , {confidence}
        ax.axis('off')
    plt.show()
show_random_predictions(model, test_loader, dataset.class_mapping)
"""







"""
#-----------------------------------------------------------------------------------------------------------
# Prediction / testing for fed in image (for final UI) - most of this part is just debugging 
# to put this part to a separate file 

img_path = '.\data\_filtered_ovary_diseases\simple_cyst.jpg'
img = Image.open(img_path).convert("RGB")   #do i need RGB? 
img_tensor = transform(img).unsqueeze(0).to(device)   

print("test image nim/max: ", img_tensor.min(), img_tensor.max()) 
#resnet50 pre-defined weight allow them to be between -2 and 2

with torch.no_grad():
    pred = model(img_tensor)
    probabilities = torch.nn.functional.softmax(pred, dim=1)  # debug - Converts logits to probabilities
    pred_label = torch.argmax(probabilities, dim=1) #debug
    predicted_class = list(dataset.class_mapping.keys())[list(dataset.class_mapping.values()).index(pred_label.item())]
    print(f"Predicted label index: {pred_label.item()}, ({predicted_class})")  #debug

plt.figure()    # plot the image in the end
plt.title(f"Predicted Label: {predicted_class}")
img_to_show = img_tensor.squeeze(0).detach().cpu()  # Shape: [3, H, W]
img_to_show = img_to_show.permute(1, 2, 0).numpy()  # Shape: [H, W, 3]
img_to_show = (img_to_show - img_to_show.min()) / (img_to_show.max() - img_to_show.min())
#img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
#img = (img - img.min()) / (img.max() - img.min())  # Normalize for display
plt.imshow(img)
plt.xticks([])
plt.yticks([])
plt.savefig(f".\prediction_of_image.png", dpi=300)
plt.show()
"""
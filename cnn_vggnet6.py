import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from PIL import Image

"""------------------------------------------------------------------------
COMMENTS FOR THE CODE: 
- try to rewrtie this CNN into different types to see the difference 
- test whether ReLU or sigmoid is better for activation function 
- test how many filters (should increase to 64, 128, 526 bc it common for CNNs) 
- num classes here is 3
- might need to seperate the training from the final funcitoning prototype (so that when we feed in the final picture, its like an app interface)
    - and also we wouldnt have to train the model all the time 
------------------------------------------------------------------------"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #if GPU available, otherwise falls back to CPU
# can be removed 

#------------------------------------------------------------------------------------------------

# Hyper-parameters
kernel_size = 3
stride = 1 #doesnt neccessarily needs to be defined, bc its 1 on default 
#hidden_size = 1024  # it might not even be needed - Number of neurons in the hidden layers
num_classes = 3  # Binary classification (normal/malignant/benign)
num_epochs = 10  # Number of epochs
batch_size = 32  # Batch size
out_channels = 0
image_size = 128


#---------------------------------------------------------------------------------------------------
#DATALOADING

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),  
    transforms.Grayscale(num_output_channels=1), 
    #transforms.RandomHorizontalFlip(),
    #transforms.RandomRotation(15),
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.5], std=[0.5])  
])

data_path = './data/BreastUltrasound3/'
dataset = datasets.ImageFolder(root=data_path, transform=transform)

# Split the dataset into training (80%) and testing (20%)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

print(dataset.class_to_idx) # prints out the mapping of class names to numbers 

#----------------------------------------------------------------------------------------------

from collections import Counter
labels = [label for _, label in dataset.samples]
print(Counter(labels))

#----------------------------------------------------------------------------------------------

# Neural Network - vggnet-16
class CNN(nn.Module): # this one is based on VGG-16 first to simplify the arcitecture at first (doubling in each layer) 
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels=32, kernel_size=3, stride=1) #126x126x64
        self.relu1 = nn.ReLU()
        self.norm1 = nn.BatchNorm2d(32)    #batch_normalization The num_features value should always match the number of output channels (filters) of the previous convolutional layer.
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 62x62x64, stride = 2 is the common chocie for pooling in CNN #stride = 1 would not reduce spatial size 
        self.reg1 = nn.Dropout(0.2)       #dropout (regularization) - 20% dropout rate (increase if overfitting, decrease if underfitting)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1) #60x60x128
        self.relu2 = nn.ReLU()
        self.norm2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 29x29x128 kernel = 2 more efficient, bc it literally halves the spatial dimensions 
        self.reg2 = nn.Dropout(0.2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1) #27x27x256
        self.relu3 = nn.ReLU()
        self.norm3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) #12x12x256 kernel = 2 more efficient, bc it literally halves the spatial dimensions 
        self.reg3 = nn.Dropout(0.2)


        #i need to understand whats going on here: 
        self.fc1 = nn.Linear(25088, 128) #(out_channel * final pooling * final pooling) (256*12*12)
        #self.dropout = nn.Dropout(p=0.5)  # Add this in __init__
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
        #self.dropout = nn.Dropout(p=0.5)  # Add this in __init__


    def forward(self, x):   # and in here: 
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.pool1(out)
        
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.pool2(out)

        out = self.conv3(out)
        out = self.relu3(out)
        out = self.pool3(out)
                
        out = out.reshape(out.size(0), -1)
        
        out = self.fc1(out)
        out = self.relu4(out)
        out = self.fc2(out)
        return out

model = CNN(num_classes).to(device)

#----------------------------------------------------------------------------------------------

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum = 0.9)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.005, weight_decay = 0.005, momentum = 0.9)

#Stochaistic Gradient Descendent, weight_decay prevents overfitting, momentum accelerates learning, smooths out the weight updates so it doesnt get stuck on a local minima 


#----------------------------------------------------------------------------------------------
# Train the model

print("Training started...")
for epoch in range(num_epochs):
    total_loss = 0
    model.train()
    
    for i, (images, labels) in enumerate(train_loader):
        # Forward pass
        images, labels = images.to(device), labels.to(device)
        #optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        #optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Final Loss: {avg_loss:.4f}")

print("Training finished!")

#----------------------------------------------------------------------------------------------
# Test the model

model.eval() #is it needed?
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the test dataset: {acc:.2f}%')

#--------------------------------------------------------------------------------------------------
# Prediction / testing
img_path = './data/BreastUltrasound2/normal (1).png' 
img = Image.open(img_path)
img = transform(img).unsqueeze(0)
img = img.to(device)
print("test image nim/max: ", img.min(), img.max()) # if between -1 and 1 - good / 0 and 1 if tensor

with torch.no_grad():
    pred = model(img)
    print("Raw model output: ", pred)   #debug
    
    probabilities = torch.nn.functional.softmax(pred, dim=1)  # Converts logits to probabilities
    pred_label = torch.argmax(probabilities)
    print(f"Predicted probabilities: {probabilities}")
    print(f"Predicted label: {pred_label.item()}")
    
    pred_label = torch.argmax(pred)     # picks the index with the highest value (0 = benign, 1 = malignant, 2 = normal)
    print("Predicted class: ", pred_label.item())   #debug 

plt.figure()
plt.title(f"Predicted Label: {'Malignant' if pred_label.item() == 1 else 'Benign' if pred_label.item() == 0 else 'Normal'}")
plt.imshow(img.cpu().squeeze(), cmap="gray")
plt.xticks([])
plt.yticks([])
plt.savefig(f"./prediction_of_image.png", dpi=300)
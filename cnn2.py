import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

"""------------------------------------------------------------------------
COMMENTS FOR THE CODE: 
- rewrite the current FFNN into CNN
- test whether ReLU or sigmoid is better for activation function 
- test how many filters (should increase to 64, 128, 526 bc it common for CNNs) 
- num classes here is 3
- might need to seperate the training from the final funcitoning prototype (so that when we feed in the final picture, its like an app interface)
    - and also we wouldnt have to train the model all the time 
- how to implement pooling? (linear helyett pooling?)
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
learning_rate = 0.001  # Learning rate
out_channels = 0


#---------------------------------------------------------------------------------------------------
#DATALOADING

data = "./data/BreastUltraSound2"
# Define image transformations
transform = transforms.Compose([transforms.Resize((32,32)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                          std=[0.2023, 0.1994, 0.2010])
                                     ])
# Load dataset using ImageFolder
dataset = datasets.ImageFolder(root=data, transform=transform)

# Extract features (X) and labels (y)
X = torch.stack([img[0] for img in dataset])  # Stack all images into a tensor batch
y = torch.tensor(dataset.targets)  # Convert labels to tensor

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Split dataset into train (80%) and test (20%)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False) 

#----------------------------------------------------------------------------------------------

# Neural Network
class CNN(nn.Module): # this one is based on VGG-16 first to simplify the arcitecture at first (doubling in each layer) 
    def __init__(self, num_classes, kernel_size=3, stride=1):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels=32, kernel_size=kernel_size, stride=stride) #convolving = 30
                                                                            #need to look into the caluclation more when writing the actual thesis!!!
        self.relu1 = nn.ReLU() #activation function
        self.pool1 = nn.MaxPool2d(kernel_size, stride=stride) # pooling = 28

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel_size, stride=stride) #convolving = 26
        self.relu2 = nn.ReLU() #activation function
        self.pool2 = nn.MaxPool2d(kernel_size, stride=stride) # pooling = 24

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=kernel_size, stride=stride) #convolving = 22
        self.relu3 = nn.ReLU() #activation function
        self.pool3 = nn.MaxPool2d(kernel_size, stride=stride) # pooling = 20

        #i need to understand whats going on here: 
        self.fc1 = nn.Linear(51200, 128) #(out_channel * final pooling * final pooling = 51200) 
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):   # and in here: 
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))

        x = torch.flatten(x, start_dim=1)  # Flatten for the fully connected layer
        x = self.relu4(self.fc1(x))  
        x = self.fc2(x)  
        return x

model = CNN(num_classes, kernel_size, stride).to(device)

#----------------------------------------------------------------------------------------------

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)
#Stochaistic Gradient Descendent, weight_decay prevents overfitting, momentum accelerates learning, smooths out the weight updates so it doesnt get stuck on a local minima 

#----------------------------------------------------------------------------------------------
# Train the model

print("Training started...")
for epoch in range(num_epochs):
    model.train()
    for i, (features, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

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

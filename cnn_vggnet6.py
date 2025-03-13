import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter

"""-------------------------------------------------------------------------------------------------------
STEPS OF THE MODEL: 

1) set up hyperparameters 
2) dataloading 
3) building up the model 
4) training the model
5) testing the model
6) final prediction/testing with random image  

- need to figure out some hypertuning methods, bc the loss is good, but doesnt classify correctly 
----------------------------------------------------------------------------------------------------------"""
# Hyper-parameters

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #if GPU available, otherwise falls back to CPU - can be removed 
kernel_size = 3
stride = 1 #doesnt neccessarily needs to be defined, bc its 1 on default 
num_classes = 3  # Binary classification (benign/malignant/normal)
num_epochs = 15  # Number of epochs
batch_size = 32  # Batch size
out_channels = 0
image_size = 128
dropout_rate = 0.2 

#-----------------------------------------------------------------------------------------------------------
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

train_size = int(0.8 * len(dataset)) # Split the dataset into training (80%) and testing (20%)
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

print(dataset.class_to_idx) # debug - prints out the mapping of class names to numbers 
labels = [label for _, label in dataset.samples] # debug
print(Counter(labels)) #debug

#-----------------------------------------------------------------------------------------------------------
# Neural Network - vggnet-16 - to simplify the arcitecture at first (doubling in each layer)

class CNN(nn.Module): 
    def __init__(self, num_classes, p):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels=32, kernel_size=3, stride=1) #126
        self.relu1 = nn.ReLU()
        self.norm1 = nn.BatchNorm2d(32, momentum=0.1)    #batch_normalization The num_features value should always match the number of output channels (filters) of the previous convolutional layer.
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 63, stride = 2 is the common chocie for pooling in CNN #stride = 1 would not reduce spatial size 
        self.drop1 = nn.Dropout(dropout_rate)       #dropout (regularization) - 20% dropout rate (increase if overfitting, decrease if underfitting)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1) #61
        self.relu2 = nn.ReLU()
        self.norm2 = nn.BatchNorm2d(64, momentum=0.1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 30
        self.drop2 = nn.Dropout(dropout_rate)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1) #28
        self.relu3 = nn.ReLU()
        self.norm3 = nn.BatchNorm2d(128, momentum=0.1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) #14
        self.drop3 = nn.Dropout(dropout_rate)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1) #12
        self.relu4 = nn.ReLU()
        self.norm4 = nn.BatchNorm2d(256, momentum=0.1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) #6
        self.drop4 = nn.Dropout(dropout_rate)

        self.fc1 = nn.Linear(256*6*6, 256) #(out_channel * final pooling * final pooling)
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):   
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.norm1(out)  
        out = self.pool1(out)
        out = self.drop1(out)  

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.norm2(out)
        out = self.pool2(out)
        out = self.drop2(out)

        out = self.conv3(out)
        out = self.relu3(out)
        out = self.norm3(out)
        out = self.pool3(out)
        out = self.drop3(out)

        out = self.conv4(out)
        out = self.relu4(out)
        out = self.norm4(out)
        out = self.pool4(out)
        out = self.drop4(out)

        out = out.reshape(out.size(0), -1)  # Flatten for fully connected layers

        out = self.fc1(out)
        out = self.relu5(out)
        out = self.fc2(out)  

        return out

model = CNN(num_classes, dropout_rate).to(device)

#-----------------------------------------------------------------------------------------------------------
# Train the model

# Loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

print("Training started...")
for epoch in range(num_epochs):
    total_loss = 0
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        # Forward pass
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
print("Training finished!")

#-----------------------------------------------------------------------------------------------------------
# Test the model

model.eval() 
with torch.no_grad(): # no need to track gradients during evaluation 
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1) #or torch.argmax
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy on the test dataset: {acc:.2f}%')

#-----------------------------------------------------------------------------------------------------------
# Prediction / testing - most of this part is just debugging 
img_path = './data/BreastUltrasound3/normal_test.png' 
img = Image.open(img_path)
img = transform(img).unsqueeze(0)
img = img.to(device)
print("test image nim/max: ", img.min(), img.max()) # debug - if between -1 and 1 - good / 0 and 1 if tensor

with torch.no_grad():
    pred = model(img)
    print("Raw model output: ", pred)   #debug 
    probabilities = torch.nn.functional.softmax(pred, dim=1)  # debug - Converts logits to probabilities
    pred_label = torch.argmax(probabilities) #debug
    print(f"Predicted probabilities: {probabilities}") #debug
    print(f"Predicted label: {pred_label.item()}")  #debug

    pred_label = torch.argmax(pred)     # picks the index with the highest value (0 = benign, 1 = malignant, 2 = normal)
    print("Predicted class: ", pred_label.item()) 

plt.figure()    #plotting of image at the end
plt.title(f"Predicted Label: {'Malignant' if pred_label.item() == 1 else 'Benign' if pred_label.item() == 0 else 'Normal'}")
plt.imshow(img.cpu().squeeze(), cmap="gray")
plt.xticks([])
plt.yticks([])
plt.savefig(f"./prediction_of_image.png", dpi=300)
"""-------------------------------------------------------------------------------------------------------
- create some data visualization 

----------------------------------------------------------------------------------------------------------"""
import time
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
from bayes_opt import BayesianOptimization
import sys
from pathlib import Path

project = Path(__file__).parents[1]  # get project root path
sys.path.append(str(project))

from utils.dataloading import test_dataset, train_dataset, device

#-----------------------------------------------------------------------------------------------------------
#Parameters
num_classes = 6
step_size = 30 #image_num / batch size 

#hyperparameters
batch_size = 32     #maybe 64?
lr = 0.01055
epochs = 20 
dropout = 0.3361 #not in resnet by default but helps with normalization 

#-----------------------------------------------------------------------------------------------------------
#DATALOADING
weights = ResNet50_Weights.IMAGENET1K_V2    #this will help us with a pre-built image transformation 
#use transfer learning and pre-trained models
model = resnet50(weights=weights).to(device) 

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
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1) 

#-----------------------------------------------------------------------------------------
# Hyperparameter optimization inside the training function

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

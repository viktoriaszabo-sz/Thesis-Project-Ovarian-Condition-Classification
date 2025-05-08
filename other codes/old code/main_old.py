"""-------------------------------------------------------------------------------------------------------
- create some data visualization 
REMEMBER: to run it in the project folder 
----------------------------------------------------------------------------------------------------------"""
from sklearn.metrics import confusion_matrix
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.models import mobilenet_v2
import sys
from pathlib import Path

# Get the absolute path of the project root
project = Path(__file__).parent  # Adjust based on your actual structure
sys.path.append(str(project))

from models.mobilenet import mobilenet_v2 #change according to name
from utils.dataloading import dataset, transform, device, num_class, label, labels

model = mobilenet_v2(num_classes=6) # change according to name
model.load_state_dict(torch.load('.\models\mobilenet_model.pth', map_location='cpu', weights_only=True))
model.eval()
#---------------------------------------------------------------------------

#input testing image
img_path = '.\simple_cyst.jpg'
#here would come an actual implementation of a UI 

#-----------------------------------------------------------------------------------------------------------
# Prediction / testing for fed in image (for final UI) - most of this part is just debugging 
# to put this part to a separate file 

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
#plt.show()





"""
# function to plot confusion matrix
import itertools

def plot_confusion_matrix(actual, predicted):

    cm = confusion_matrix(actual, predicted)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(7,7))
    cmap=plt.cm.Blues
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix', fontsize=25)
  
    tick_marks = np.arange(len(num_class))
    plt.xticks(tick_marks, num_class, rotation=90, fontsize=15)
    plt.yticks(tick_marks, num_class, fontsize=15)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f'),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black", fontsize = 14)

    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)
    plt.show()
# plot confusion matrix
plot_confusion_matrix(label, predicted_class)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, mobilenet_v2, googlenet
import sys
from pathlib import Path
from sklearn.metrics import confusion_matrix
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from models.mobilenet import mobilenet_v2, test_loader
from models.resnet50 import resnet50, test_loader
from models.googlenet import googlenet, test_loader
from utils.dataloading import device
from torch.func import stack_module_state
from utils.dataloading import dataset, transform, device


# Get the absolute path of the project root
project = Path(__file__).parent  # Adjust based on your actual structure
sys.path.append(str(project))

resnet = resnet50(num_classes=6) # change according to name
resnet.load_state_dict(torch.load('.\models\_resnet50_model.pth', map_location='cpu', weights_only=True))
mobile = mobilenet_v2(num_classes=6) # change according to name
mobile.load_state_dict(torch.load('.\models\mobilenet_model.pth', map_location='cpu', weights_only=True))
google = googlenet(num_classes=6, aux_logits=False, init_weights=True) # change according to name
google.load_state_dict(torch.load('.\models\googlenet_model.pth', map_location='cpu', weights_only=True), strict=False)

resnet.eval()
mobile.eval()
google.eval()
models = [resnet, mobile, google]

class EnsembleModel(nn.Module): #------------------------WEIGHTED AVERAGE 
    def __init__(self, models, weights=None):
        super().__init__()
        self.models = nn.ModuleList(models)
        if weights:
            self.weights = torch.tensor(weights, dtype=torch.float32)
            self.weights = self.weights / self.weights.sum()
        else:
            self.weights = None

    def forward(self, x):
        outputs = []
        for i, model in enumerate(self.models):
            out = model(x)
            if self.weights is not None:
                out = self.weights[i] * out
            outputs.append(out)
        # Sum outputs (weighted if weights provided)
        ensemble_output = torch.stack(outputs, dim=0).sum(dim=0)
        return ensemble_output  # logits; apply softmax outside if needed
    
def evaluate_model(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            acc = correct / total
    #return correct / total  # accuracy
    return acc

accs = []
for i in models:
    acc = evaluate_model(i, test_loader, device)
    accs.append(acc)
    print(f"Model accuracy: {acc:.4f}")

weights = np.array(accs)
weights = weights / weights.sum()  # ensures sum(weights) == 1
weights = weights.tolist()

ensemble = EnsembleModel(models, weights=weights)
ensemble_acc = evaluate_model(ensemble, test_loader, device)
print(f"Ensemble model accuracy: {ensemble_acc:.4f}")

ensemble.eval()
ensemble.to(device)


#-------------------------------------------------------------------------------------

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
    pred = ensemble(img_tensor)
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



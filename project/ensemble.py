import torch
import torch.nn as nn
from torchvision.models import resnet50, mobilenet_v2, googlenet
import sys
from pathlib import Path
import torch
import numpy as np
from models.mobilenet import mobilenet_v2, test_loader
from models.resnet50 import resnet50, test_loader
from models.googlenet import googlenet, test_loader
from utils.dataloading import device

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
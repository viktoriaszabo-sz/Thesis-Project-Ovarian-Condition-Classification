import torch
import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils.dataloading import device
from utils.dataloading import dataset, transform, device
from ensemble import ensemble

# Get the absolute path of the project root
project = Path(__file__).parent  # Adjust based on your actual structure
sys.path.append(str(project))

ensemble.eval()
ensemble.to(device)

#-------------------------------------------------------------------------------------
#input testing image
img_path = '.\simple_cyst.jpg'

#here would come an actual implementation of a UI 
#-----------------------------------------------------------------------------------------------------------

img = Image.open(img_path).convert("RGB")   #do i need RGB? 
img_tensor = transform(img).unsqueeze(0).to(device)   

with torch.no_grad():
    pred = ensemble(img_tensor)
    probabilities = torch.nn.functional.softmax(pred, dim=1)  # debug - Converts logits to probabilities
    pred_label = torch.argmax(probabilities, dim=1) #debug
    predicted_class = list(dataset.class_mapping.keys())[list(dataset.class_mapping.values()).index(pred_label.item())]

plt.figure()    # plot the image in the end
plt.title(f"Predicted Label: {predicted_class}")
img_to_show = img_tensor.squeeze(0).detach().cpu()  # Shape: [3, H, W]
img_to_show = img_to_show.permute(1, 2, 0).numpy()  # Shape: [H, W, 3]
img_to_show = (img_to_show - img_to_show.min()) / (img_to_show.max() - img_to_show.min())
plt.imshow(img)
plt.xticks([])
plt.yticks([])
plt.savefig(f".\prediction_of_image.png", dpi=300)




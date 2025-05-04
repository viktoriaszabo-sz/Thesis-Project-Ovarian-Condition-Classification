"""-------------------------------------------------------------------------------------------------------
- create some data visualization 
REMEMBER: to run it in the project folder 
----------------------------------------------------------------------------------------------------------"""
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.models import mobilenet_v2
import sys
from pathlib import Path

# Get the absolute path of the project root
project = Path(__file__).parent  # Adjust based on your actual structure
sys.path.append(str(project))

from models.mobilenet import mobilenet_v2 #change according to name
from utils.dataloading import dataset, transform, test_dataset, device

model = mobilenet_v2(num_classes=6) # change according to name
model.load_state_dict(torch.load('.\models\mobilenet_model.pth', map_location='cpu', weights_only=True))
                                            #change according to model name
model.eval()
#---------------------------------------------------------------------------
#input testing image
img_path = '.\data\_filtered_ovary_diseases\simple_cyst.jpg'

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

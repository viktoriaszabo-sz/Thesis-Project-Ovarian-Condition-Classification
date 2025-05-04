import torch
import torch_directml # Use DirectML backend

#print(torch.version.cuda)  # Check if it matches your installed CUDA version (12.6)
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch_directml.device()

#model.to(device)
x = torch.rand(2, 2).to(device)
print(x.device)  # Should print "cuda:0"

""""
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch-directml 
.
"""

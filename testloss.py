import torch
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image

from network.network import Unet
from loss.loss import CLIPLoss


device = "cuda" if torch.cuda.is_available() else "cpu"
loss = CLIPLoss(device)

# x = loss.compute_text_direction("good","bad")
# print(x.shape)


src = torch.randn(1,3,224,224).to(device)
tar = torch.randn(1,3,224,224).to(device)
x = loss(src,"good",tar,"bad")
print(x,x.shape)

import time
start = time.time()
time.sleep(3)
end = time.time()
usetime = start - end
print(f"usetime: {usetime}" )

import clip
print(clip.available_models())
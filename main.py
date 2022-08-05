import torch
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image

from network.network import Unet
from loss.cliploss import CLIPLoss


class Model(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.loss = CLIPLoss(device)
        self.unet = Unet(device)

    def forward(self, pic, source, target):
        neo_pic = self.unet(pic.unsqueeze(0)).squeeze(0)
        loss = self.loss(pic, source, neo_pic, target)
        return neo_pic, loss




# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

lr = 0.001
model = Model(device)
cliploss = CLIPLoss(device)

opt = optim.Adam(model.parameters(), lr=lr)

topil = transforms.ToPILImage()
topic = transforms.ToTensor()

def train(iteration, pic, source, target):
    for i in range(iteration):
        opt.zero_grad()
        neo_pic, loss = model(pic, source, target)
        loss = loss.requires_grad_()
        loss.backward()
        opt.step()
        pil = topil(neo_pic.cpu())
        print("iter:", i + 1, "loss:", loss.item())
        if ((i + 1) % 10)==0:

            pil.save(f"{(i + 1) // 10}.jpg")
    neo_pic, loss = model(pic, source, target)
    pil = topil(neo_pic.cpu())
    pil.save(f"res.jpg")


pil = Image.open(f"ori.jpg")
pic = topic(pil)
pic.requires_grad = False
source = "tree"
target = "grass"
train(50, pic, source, target)




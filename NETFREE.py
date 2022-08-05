import torch
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image

from network.network import Unet
from loss.loss import CLIPLoss






device = "cuda" if torch.cuda.is_available() else "cpu"

lr = 0.001
# model = Unet(device)
cliploss = CLIPLoss(device)
mseloss = torch.nn.MSELoss()


topil = transforms.ToPILImage()
topic = transforms.ToTensor()

def train(iteration, pic, source, target):
    neo_pic = pic.detach()
    neo_pic.requires_grad = True
    opt = optim.Adam([neo_pic], lr=lr)
    for i in range(iteration):
        opt.zero_grad()
        loss = cliploss(pic,source,neo_pic,target)
        loss.backward()
        opt.step()
        pil = topil(neo_pic.squeeze(0).cpu())
        print("iter:", i + 1, "loss:", loss.item())
        if ((i + 1) % 10)==0:

            pil.save(f"{(i + 1) // 10}.jpg")
    pil = topil(neo_pic.squeeze(0).cpu())
    pil.save(f"res.jpg")


pil = Image.open(f"ori.jpg")
pil = transforms.Resize(size=(224, 224), interpolation=Image.BICUBIC)(pil)
pic = topic(pil).unsqueeze(0).to(device)
source = "green"
target = "red"
train(500, pic, source, target)
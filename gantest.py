import torch
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image

from network.gan import Net
from loss.loss import CLIPLoss

device = "cuda" if torch.cuda.is_available() else "cpu"

lr1 = 0.001
lr2 = 0.0001

model = Net().to(device)
cliploss = CLIPLoss(device)
mseloss = torch.nn.MSELoss()



topil = transforms.ToPILImage()
topic = transforms.ToTensor()


def train(iteration1, iteration2, pic, source, target):

    input = torch.randn(1, 512, 7, 7).to(device)
    style = cliploss.encode_images(pic).float().to(device)

    opt1 = optim.Adam(model.parameters(), lr=lr1)
    for i in range(iteration1):
        opt1.zero_grad()
        neo_pic = model(input,style)
        loss = mseloss(pic, neo_pic) * 1
        loss.backward()
        opt1.step()
        pil = topil(neo_pic.squeeze(0).cpu())
        print("iter:", i + 1, "loss:", loss.item())
        if ((i + 1) % 100) == 0:
            pil.save(f"./pic1/{(i + 1) // 100}.jpg")

    opt2 = optim.Adam(model.parameters(), lr=lr2)
    for i in range(iteration2):
        opt2.zero_grad()
        neo_pic = model(input,style)
        loss = cliploss(pic, source, neo_pic, target) * 1
        loss.backward()
        opt2.step()
        pil = topil(neo_pic.squeeze(0).cpu())
        print("iter:", i + 1, "loss:", loss.item())
        if ((i + 1) % 10) == 0:
            pil.save(f"./pic2/{(i + 1) // 10}.jpg")

    neo_pic = model(input,style)
    pil = topil(neo_pic.squeeze(0).cpu())
    pil.save(f"res.jpg")


pil = Image.open(f"ori.jpg")
pil = transforms.Resize(size=(224, 224), interpolation=Image.BICUBIC)(pil)
pic = topic(pil).unsqueeze(0).to(device)
pic.requires_grad = False

source = "noon"
target = "night"

train(1000, 500, pic, source, target)

# print(style.dtype)
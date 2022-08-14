import time
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image

from network.gan import Net
from loss.loss import CLIPLoss

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Net().to(device)
cliploss1 = CLIPLoss(device, direction_loss_type='cosine', clip_model='ViT-B/32')
cliploss2 = CLIPLoss(device, direction_loss_type='cosine', clip_model='ViT-B/16')
mseloss = torch.nn.MSELoss()

topil = transforms.ToPILImage()
topic = transforms.ToTensor()


def train(iteration1, iteration2, lr1, lr2, pic, source, target):
    input = torch.ones(1, 512, 7, 7).to(device) * 0.5
    style = cliploss1.encode_images(pic).float().to(device) * 0.5 \
            + cliploss2.encode_images(pic).float().to(device) * 0.5

    opt1 = optim.Adam(model.parameters(), lr=lr1)
    for i in range(iteration1):
        opt1.zero_grad()
        neo_pic = model(input, style)
        loss = mseloss(pic, neo_pic) * 1
        loss.backward()
        opt1.step()
        pil = topil(neo_pic.squeeze(0).cpu())
        print("iter:", i + 1, "loss:", loss.item())
        if ((i + 1) % 100) == 0:
            pil.save(f"./pic1/{(i + 1) // 100}.jpg")
    torch.save(model, 'net.pth')

    # model = torch.load('net.pth')

    opt2 = optim.Adam(model.parameters(), lr=lr2)
    for i in range(iteration2):
        opt2.zero_grad()
        neo_pic = model(input, style)
        loss = cliploss1(pic, source, neo_pic, target) * 0.5 + cliploss2(pic, source, neo_pic, target) * 0.5
        loss.backward()
        opt2.step()
        pil = topil(neo_pic.squeeze(0).cpu())
        print("iter:", i + 1, "loss:", loss.item())
        if ((i + 1) % 100) == 0:
            pil.save(f"./pic2/{(i + 1) // 100}.jpg")

    neo_pic = model(input, style)
    pil = topil(neo_pic.squeeze(0).cpu())
    pil.save(f"res.jpg")


pil = Image.open(f"ori1.jpg")
pil = transforms.Resize(size=(224, 224), interpolation=Image.BICUBIC)(pil)
pic = topic(pil).unsqueeze(0).to(device)
pic.requires_grad = False

lr1 = 0.001
lr2 = 0.00002
source = "sketch"
target = "HQ photo"
iteration1 = 1000
iteration2 = 100

start = time.time()
train(iteration1, iteration2, lr1, lr2, pic, source, target)
end = time.time()
usetime = end - start
print(f"usetime: {usetime}")

# print(style.dtype)

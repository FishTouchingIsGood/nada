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
cliploss2 = CLIPLoss(device, direction_loss_type='cosine', clip_model='RN101')
# torch.save(cliploss1.fc, 'fc1.pth')
# torch.save(cliploss2.fc, 'fc2.pth')
# cliploss1.fc = torch.load('fc1.pth')
# cliploss1.fc = torch.load('fc2.pth')


mseloss = torch.nn.MSELoss()

topil = transforms.ToPILImage()
topic = transforms.ToTensor()


def train(iteration1, iteration2, lr1, lr2, pic, source, target):
    pic.requires_grad = False

    input = torch.ones(1, 512, 7, 7).to(device)
    style = cliploss1.encode_images(pic).float().to(device) * 0.5 \
             + cliploss2.encode_images(pic).float().to(device) * 0.5


    opt1 = optim.Adam(model.parameters(), lr=lr1)
    for i in range(iteration1):
        opt1.zero_grad()
        neo_pic = model(input, style)
        # neo_pic = torch.nn.functional.interpolate(neo_pic, (224, 224), mode="bilinear")
        loss = mseloss(pic, neo_pic) * 1
        loss.backward()
        opt1.step()
        pil = topil(neo_pic.squeeze(0).cpu())
        print("iter:", i + 1, "loss:", loss.item())
        if ((i + 1) % 50) == 0:
            pil.save(f"./pic1/{(i + 1) // 50}.jpg")
    torch.save(model, 'net.pth')

    # model = torch.load('net.pth')

    neo_pic = model(input, style)
    pil = topil(neo_pic.squeeze(0).cpu())
    pil.save(f"ress.jpg")

    opt2 = optim.Adam(model.parameters(), lr=lr2)
    opt3 = optim.Adam(model.parameters(), lr=lr3)
    for i in range(iteration2):
        model.freeze_conv()
        opt2.zero_grad()
        neo_pic = model(input, style)
        # neo_pic = torch.nn.functional.interpolate(neo_pic, (224, 224), mode="bilinear")

        gob_loss1 = cliploss1.forward_gol(pic, source, neo_pic, target)
        gob_loss2 = cliploss2.forward_gol(pic, source, neo_pic, target)
        gob_loss = gob_loss1 * 0.5 + gob_loss2 * 0.5
        gob_loss.backward()
        opt2.step()


        model.freeze_linear()
        opt3.zero_grad()
        neo_pic = model(input, style)
        # neo_pic = torch.nn.functional.interpolate(neo_pic, (224, 224), mode='bilinear')
        dir_loss1 = cliploss1.forward_dir(pic, source, neo_pic, target)
        dir_loss2 = cliploss2.forward_dir(pic, source, neo_pic, target)
        dir_loss = dir_loss1 * 0.5 + dir_loss2 * 0.5
        dir_loss.backward()
        opt3.step()

        pil = topil(neo_pic.squeeze(0).cpu())
        print("iter:", i + 1, "gob_loss:", gob_loss.item(),"dir_loss:", dir_loss.item())
        if ((i + 1) % 10) == 0:
            pil.save(f"./pic2/{(i + 1) // 10}.jpg")

    neo_pic = model(input, style)
    pil = topil(neo_pic.squeeze(0).cpu())
    pil.save(f"res.jpg")
    # neo_pic = torch.nn.functional.interpolate(neo_pic, (224, 224), mode="bilinear")
    # pil = topil(neo_pic.squeeze(0).cpu())
    # pil.save(f"ress.jpg")


pil = Image.open(f"ori3.jpg")
pil = transforms.Resize(size=(224, 224), interpolation=Image.BICUBIC)(pil)
pic = topic(pil).unsqueeze(0).to(device)
pic.requires_grad = False

lr1 = 0.001
lr2 = 0.0001
lr3 = 0.00001

source = "photo"
target = "black and white"

iteration1 = 500
iteration2 = 200

# iteration1 = 100
# iteration2 = 100

start = time.time()
train(iteration1, iteration2, lr1, lr2, pic, source, target)
end = time.time()
usetime = end - start
print(f"usetime: {usetime}")

# print(style.dtype)

import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

import numpy as np

import math
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

model,pre = clip.load("RN50",device=device)

preprocess = transforms.Compose(
            [
                # transforms.Normalize(mean=(-1.0, -1.0, -1.0), std=(2.0, 2.0, 2.0)),
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
            ]
        )

topil = transforms.ToPILImage()
topic = transforms.ToTensor()

pil = Image.open(f"ori.jpg")
pil1 = transforms.Resize(size=(224, 224), interpolation=Image.BICUBIC)(pil)
pic = topic(pil1)
pic = preprocess(pic)

pic0 = pre(pil)

x = pic0==pic
print(x)
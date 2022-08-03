from loss.loss import CLIPLoss
import torch
import clip
import torchvision.transforms as transforms

device = "cuda"

func = CLIPLoss(device)

p = transforms.ToPILImage()

pic1 = (torch.ones(3, 224, 224)).to(device)
pic2 = ((torch.ones(3, 224, 224)) * 0.5).to(device)

str1 = "cat"
str2 = "cute cat"

# res = func(pic1, str1, pic2, str2)
# print(res)
func.preprocess(pic1)

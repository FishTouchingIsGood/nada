import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

import numpy as np

import math
import clip
from PIL import Image

from utils.text_templates import imagenet_templates, part_templates, imagenet_templates_small



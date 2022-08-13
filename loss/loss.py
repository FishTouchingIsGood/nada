import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

import numpy as np

import math
import clip
from PIL import Image

from utils.text_templates import imagenet_templates, part_templates, imagenet_templates_small


class DirectionLoss(torch.nn.Module):

    def __init__(self, loss_type='mse'):
        super(DirectionLoss, self).__init__()

        self.loss_type = loss_type

        self.loss_func = {
            'mse': torch.nn.MSELoss,
            'cosine': torch.nn.CosineSimilarity,
            'mae': torch.nn.L1Loss
        }[loss_type]()

    def forward(self, x, y):
        if self.loss_type == "cosine":
            return 1. - self.loss_func(x, y)

        return self.loss_func(x, y)


class CLIPLoss(torch.nn.Module):
    def __init__(self, device, direction_loss_type='cosine', clip_model='ViT-B/32'):
        super().__init__()

        self.device = device
        self.model, clip_preprocess = clip.load(clip_model, device=self.device)
        for x in self.model.parameters():
            x.requires_grad = False


        self.clip_preprocess = clip_preprocess

        self.preprocess = transforms.Compose(
            [
                # transforms.Normalize(mean=(-1.0, -1.0, -1.0), std=(2.0, 2.0, 2.0)),
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
            ]
        )

        # self.preprocess = transforms.Compose(
        #     [
        #         # transforms.Normalize(mean=(-1.0, -1.0, -1.0), std=(2.0, 2.0, 2.0)),
        #         # transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        #         transforms.ToPILImage()
        #     ] +
        #     clip_preprocess.transforms
        # )

        self.texture_loss = torch.nn.MSELoss()
        self.angle_loss = torch.nn.L1Loss()

        self.direction_loss = DirectionLoss(direction_loss_type)

        self.target_direction = None
        self.src_text_features = None

    def tokenize(self, strings):
        return clip.tokenize(strings)#.to(self.device)

    def encode_text(self, tokens):
        return self.model.encode_text(tokens)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = self.preprocess(images.squeeze(0).cpu())
        images = images.to(self.device).unsqueeze(0)
        return self.model.encode_image(images)

    def compose_text_with_templates(self, text: str, templates=imagenet_templates) -> list:
        return [template.format(text) for template in templates]

    def get_text_features(self, class_str: str, templates=imagenet_templates, norm: bool = True) -> torch.Tensor:
        template_text = self.compose_text_with_templates(class_str, templates)

        tokens = clip.tokenize(template_text).to(self.device)

        text_features = self.encode_text(tokens).detach()

        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def get_image_features(self, img: torch.Tensor, norm: bool = True) -> torch.Tensor:
        image_features = self.encode_images(img)

        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features

    def compute_text_direction(self, source_class: str, target_class: str) -> torch.Tensor:
        source_features = self.get_text_features(source_class)
        target_features = self.get_text_features(target_class)

        text_direction = (target_features - source_features).mean(axis=0, keepdim=True)
        text_direction /= text_direction.norm(dim=-1, keepdim=True)

        return text_direction

    def clip_directional_loss(self, src_img: torch.Tensor, source_class: str, target_img: torch.Tensor,
                              target_class: str) -> torch.Tensor:

        if self.target_direction is None:
            self.target_direction = self.compute_text_direction(source_class, target_class)

        src_encoding = self.get_image_features(src_img)
        target_encoding = self.get_image_features(target_img)

        edit_direction = (target_encoding - src_encoding)
        if edit_direction.sum().item() == 0:
            target_encoding = self.get_image_features(target_img + 1e-6)
            edit_direction = (target_encoding - src_encoding)

        edit_direction /= (edit_direction.clone().norm(dim=-1, keepdim=True))

        return self.direction_loss(edit_direction, self.target_direction).mean()




    def global_clip_loss(self, img: torch.Tensor, text) -> torch.Tensor:

        tokens = clip.tokenize(text).to(self.device)

        image = self.preprocess(img.squeeze(0).cpu())
        image = image.to(self.device).unsqueeze(0)
        logits_per_image, _ = self.model(image, tokens)

        return (1. - logits_per_image / 100).mean()



################################################################

    def set_text_features(self, source_class: str, target_class: str) -> None:
        source_features = self.get_text_features(source_class).mean(axis=0, keepdim=True)
        self.src_text_features = source_features / source_features.norm(dim=-1, keepdim=True)

        target_features = self.get_text_features(target_class).mean(axis=0, keepdim=True)
        self.target_text_features = target_features / target_features.norm(dim=-1, keepdim=True)

    # todo
    def clip_angle_loss(self, src_img: torch.Tensor, source_class: str, target_img: torch.Tensor, target_class: str) -> torch.Tensor:
        if self.src_text_features is None:
            self.set_text_features(source_class, target_class)

        cos_text_angle = self.target_text_features @ self.src_text_features.T
        text_angle = torch.acos(cos_text_angle)

        src_img_features = self.get_image_features(src_img).unsqueeze(2)
        target_img_features = self.get_image_features(target_img).unsqueeze(1)

        cos_img_angle = torch.clamp(target_img_features @ src_img_features, min=-1.0, max=1.0)
        img_angle = torch.acos(cos_img_angle)

        text_angle = text_angle.unsqueeze(0).repeat(img_angle.size()[0], 1, 1)
        cos_text_angle = cos_text_angle.unsqueeze(0).repeat(img_angle.size()[0], 1, 1)

        return self.angle_loss(cos_img_angle, cos_text_angle)
################################################################



    def forward(self, src_img: torch.Tensor, source_class: str, target_img: torch.Tensor, target_class: str):
        loss = 0.0


        loss += 0.01 * self.clip_directional_loss(src_img, source_class, target_img, target_class)
        loss += 0.1 * self.global_clip_loss(target_img, f"a {target_class}")
        # loss += 1 * self.texture_loss(src_img, target_img)
        # loss += 1 * self.clip_angle_loss(src_img, source_class, target_img, target_class)
        # print(loss1.item(),loss2.item(),loss3.item(),loss4.item())
        # loss += loss1 + loss2 + loss3 + loss4
        # loss.requires_grad = True
        return loss

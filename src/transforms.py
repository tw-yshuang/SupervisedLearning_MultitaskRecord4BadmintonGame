import math, random
from typing import List, Union, Callable

import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as TF


class CustomCompose:
    def __init__(self, transforms: List[Callable]) -> None:
        self.transforms = transforms

    def __call__(self, img: torch.Tensor, coordXYs: Union[torch.Tensor, None] = None):
        for t in self.transforms:
            if t.__module__ != 'torchvision.transforms.transforms':
                img, coordXYs = t(img, coordXYs)
            else:
                img = t(img)
        return img, coordXYs

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class IterativeCustomCompose:
    def __init__(self, transforms: List[Callable], transform_img_size=(512, 512), device: str = 'cuda') -> None:
        '''
        transform_img_size: (H, W)
        '''
        self.compose = CustomCompose(transforms)
        self.transform_img_size = transform_img_size
        self.device = device

    def __call__(self, batch_imgs: torch.Tensor, batch_coordXYs: Union[torch.Tensor, None] = None):
        process_batch_imgs = torch.zeros((*batch_imgs.shape[0:3], *self.transform_img_size), dtype=torch.float32).to(self.device)

        imgs: torch.Tensor
        coordXYs: torch.Tensor
        for i, (imgs, coordXYs) in enumerate(zip(batch_imgs, batch_coordXYs)):
            if coordXYs.sum() != 0:
                process_batch_imgs[i], batch_coordXYs[i] = self.compose(imgs, coordXYs)
            else:
                process_batch_imgs[i] = self.compose(imgs, None)[0]

        return process_batch_imgs, batch_coordXYs

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


def RandomCrop(crop_size=(640, 640), p=0.5):
    def __RandomCrop(img: torch.Tensor, coordXYs: Union[torch.Tensor, None] = None):
        if random.random() < p:
            orgH, orgW = img.shape[-2:]
            t, l, h, w = transforms.RandomCrop.get_params(img, crop_size)
            # imgs = [TF.crop(img, t, l, h, w) for img in imgs]
            img = TF.crop(img, t, l, h, w)

            if isinstance(coordXYs, (torch.Tensor, np.ndarray)) and coordXYs.nelement() != 0:
                coordXYs[0] -= l / orgH  # related with X
                coordXYs[1] -= t / orgW  # related with Y

        return img, coordXYs

    return __RandomCrop


def RandomResizedCrop(
    sizeHW: List[int] = (640, 640),
    scale: List[float] = (0.6, 1.6),
    ratio: List[float] = (3.0 / 5.0, 2.0 / 1.0),
    p: float = 0.5,
):
    def __RandomResizedCrop(img: torch.Tensor, coordXYs: Union[torch.Tensor, None] = None):
        if random.random() < p:
            orgH, orgW = img.shape[-2:]
            t, l, h, w = transforms.RandomResizedCrop.get_params(img, scale, ratio)
            img = TF.resized_crop(img, t, l, h, w, size=sizeHW, antialias=True)
            if isinstance(coordXYs, (torch.Tensor, np.ndarray)) and coordXYs.nelement() != 0:
                t, h = t / orgH, h / orgH
                l, w = l / orgW, w / orgW
                coordXYs[0] = (coordXYs[0] - l) / w
                coordXYs[1] = (coordXYs[1] - t) / h
        else:
            w, h = img.shape[-1], img.shape[-2]
            img = TF.resize(img, size=sizeHW)

        return img, coordXYs

    return __RandomResizedCrop


def RandomHorizontalFlip(p=0.5):
    def __HorizontalFlip(img: torch.Tensor, coordXYs: Union[torch.Tensor, None] = None):
        if random.random() < p:
            img = TF.hflip(img)
            if isinstance(coordXYs, (torch.Tensor, np.ndarray)) and coordXYs.nelement() != 0:
                coordXYs[0] = 1.0 - coordXYs[0]  # related with X

        return img, coordXYs

    return __HorizontalFlip


def RandomRotation(
    degrees: List[float],
    interpolation=transforms.InterpolationMode.BILINEAR,
    expand: bool = False,
    center: Union[List[int], None] = None,
    fill: Union[List[int], None] = None,
    p=0.5,
):
    def __RandomRotation(img: torch.Tensor, coordXYs: Union[torch.Tensor, None] = None):
        if random.random() < p:
            angle = transforms.RandomRotation.get_params(degrees)
            img = TF.rotate(img, angle, interpolation, expand, center, fill)
            if isinstance(coordXYs, (torch.Tensor, np.ndarray)) and coordXYs.nelement() != 0:
                coordX = coordXYs[0] - 0.5
                coordY = coordXYs[1] - 0.5

                angle = angle * math.pi
                cos = math.cos(angle)
                sin = math.sin(angle)
                coordXYs[0] = coordX * cos - coordY * sin + 0.5
                coordXYs[1] = coordX * sin + coordY * cos + 0.5

        return img, coordXYs

    return __RandomRotation

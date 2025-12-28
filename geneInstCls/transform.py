#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
  @Email:  guangmingwu2010@gmail.com
  @Copyright: go-hiroaki
  @License: MIT
"""
import random
import numpy as np
from torchvision import transforms
from torchvision.transforms import functional as F


class BrightShift(object):
    """
    Args:
        brightness (float) – How much to jitter brightness. brightness_factor is chosen uniformly from \
        [max(0, 1 - brightness), 1 + brightness].
     """

    def __init__(self, brightness):
        assert 0 <= brightness <= 1.0, "brightness should be in [0, 1.0]."
        self.brightness = brightness

    def __call__(self, img):
        bright = np.random.uniform(
            low=max(0, 1 - self.brightness), high=(1 + self.brightness))
        return F.adjust_brightness(img, bright)


class ContrastShift(object):
    """
    Args:
        contrast (float) – How much to jitter contrast. contrast_factor is chosen uniformly from \
        [max(0, 1 - contrast), 1 + contrast].
    """

    def __init__(self, contrast):
        assert 0 <= contrast <= 1.0, "contrast should be in [0, 1.0]."
        self.contrast = contrast

    def __call__(self, img):
        contrast = np.random.uniform(
            low=max(0, 1 - self.contrast), high=(1 + self.contrast))
        return F.adjust_contrast(img, contrast)


class SaturationShift(object):
    """
    Args:
        saturation (float) – How much to jitter saturation. saturation_factor is chosen uniformly from \
        [max(0, 1 - saturation), 1 + saturation].
    """

    def __init__(self, saturation):
        assert 0 <= saturation <= 1.0, "saturation should be in [0, 1.0]."
        self.saturation = saturation

    def __call__(self, img):
        saturation = np.random.uniform(
            low=max(0, 1 - self.saturation), high=(1 + self.saturation))
        return F.adjust_saturation(img, saturation)


class GammaShift(object):
    """
    Args:
        gamma (float) – Non negative real number, same as γ in the equation.\
        gamma larger than 1 make the shadows darker, while gamma smaller than 1 make dark regions lighter.
    """

    def __init__(self, gamma):
        assert 0 <= gamma <= 1.0, "gamma should be in [0, 1.0]."
        self.gamma = gamma

    def __call__(self, img):
        gamma = np.random.uniform(
            low=max(0, 1 - self.gamma), high=(1 + self.gamma))
        return F.adjust_gamma(img, gamma, gain=1)


class ColorShift(object):
    """
    Args:
        brightness (float) – How much to jitter brightness. brightness_factor is chosen uniformly from \
        [max(0, 1 - brightness), 1 + brightness].
        contrast (float) – How much to jitter contrast. contrast_factor is chosen uniformly from \
        [max(0, 1 - contrast), 1 + contrast].
        saturation (float) – How much to jitter saturation. saturation_factor is chosen uniformly from \
        [max(0, 1 - saturation), 1 + saturation].
        gamma (float) – Non negative real number, same as γ in the equation.\
        gamma larger than 1 make the shadows darker, while gamma smaller than 1 make dark regions lighter.
    """

    def __init__(self, brightness=0.3, contrast=0.3, saturation=0.3, gamma=0.3):
        self.color_coms = transforms.Compose([
            BrightShift(brightness),
            ContrastShift(contrast),
            SaturationShift(saturation),
            GammaShift(gamma)])

    def __call__(self, img):
        return self.color_coms(img)

    
class RandomVerticalFlip(object):
    """Randomly flip the src and the tar"""

    def __init__(self):
        pass

    def __call__(self, img):
        condition = random.randint(0, 1)
        if condition:
            img = F.vflip(img)
        return img


class RandomHorizontalFlip(object):
    """Randomly flip the src and the tar"""

    def __init__(self,):
        pass

    def __call__(self, img):
        condition = random.randint(0, 1)
        if condition:
            img = F.hflip(img)
        return img
    

class RandomCrop(object):
    """Randomly flip the src and the tar"""

    def __init__(self, dsize=448):
        self.dsize = dsize

    def __call__(self, img):
        w, h = img.size
        # random crop on bigger area
        x_pad = random.randint(0, w - self.dsize)
        y_pad = random.randint(0, h - self.dsize)
        return img.crop((x_pad, y_pad, x_pad + self.dsize, y_pad + self.dsize))
    

class CenterCrop(object):
    """Randomly flip the src and the tar"""

    def __init__(self, dsize=448):
        self.dsize = dsize

    def __call__(self, img):
        w, h = img.size
        # random crop on bigger area
        x_pad = (w - self.dsize) // 2
        y_pad = (h - self.dsize) // 2
        return img.crop((x_pad, y_pad, x_pad + self.dsize, y_pad + self.dsize))


def compose_transform():
    return transforms.Compose(
        [ColorShift(), 
         RandomVerticalFlip(), 
         RandomHorizontalFlip(),
         RandomCrop()
         ]
    )
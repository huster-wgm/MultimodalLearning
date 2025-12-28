#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
  @Email:  guangmingwu2010@gmail.com
  @Copyright: go-hiroaki
  @License: MIT
"""
import torch
from torch import nn


eps = 1e-6


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
        self.criterion = nn.L1Loss(reduction='mean')

    def __repr__(self):
        return "L1"

    def forward(self, output, target):
        loss = self.criterion(output, target)
        return loss


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
    
    def __repr__(self):
        return "MSE"

    def forward(self, output, target):
        loss = self.criterion(output, target)
        return loss


class PSNRLoss(nn.Module):
    def __init__(self):
        super(PSNRLoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')

    def __repr__(self):
        return "PSNR"

    def forward(self, output, target):
        mse = self.criterion(output, target)
        loss = 10 * torch.log10(1.0 / mse)
        return loss


class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    def __repr__(self):
        return "CE"

    def forward(self, output, target):
        loss = self.criterion(output, target)
        return loss


class FLoss(nn.Module):
    """
    Focal Loss
    Lin, Tsung-Yi, et al. \
    "Focal loss for dense object detection." \
    Proceedings of the IEEE international conference on computer vision. 2017.
    (modified from https://github.com/umbertogriffo/focal-loss-keras/blob/master/losses.py)
    """
    def __init__(self, gamma=1., weight=None, size_average=True):
        super(FLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average

    def __repr__(self):
        return 'Focal'

    def _get_weights(self, y_true, nb_ch):
        """
        args:
            y_true : 3-d ndarray in [batch_size, img_rows, img_cols]
            nb_ch : int 
        return [float] weights
        """
        batch_size, img_rows, img_cols = y_true.shape
        pixels = batch_size * img_rows * img_cols
        weights = [torch.sum(y_true==ch).item() / pixels for ch in range(nb_ch)]
        return weights

    def forward(self, output, target):
        output = torch.clamp(output, min=eps, max=(1. - eps))
        if output.shape[1] == 1:
            # binary focal loss
            alpha = 0.25
            loss = - (1.-alpha) * ((1.-output)**self.gamma)*(target*torch.log(output)) \
              - alpha * (output**self.gamma)*((1.-target)*torch.log(1.-output))
        else:
            # multi-class focal loss
            # weights = self._get_weigthts(torch.argmax(target, dim=1), target.shape[1])
            loss = - ((1.-output)**self.gamma)*(target*torch.log(output))

        if self.size_average: 
            return loss.mean()
        else: 
            return loss.sum()

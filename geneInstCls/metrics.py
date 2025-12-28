#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
  @Email:  guangmingwu2010@gmail.com \
           guozhilingty@gmail.com
  @Copyright: go-hiroaki & Chokurei
  @License: MIT
"""
import math
import torch
import random
import torch.nn as nn
import torch.nn.functional as F


esp = 1e-5


def _binarize(y_data, threshold):
    """
    args:
        y_data : [float] 2-d tensor in [batch_size, chs, img_rows, img_cols]
        threshold : [float] [0.0, 1.0]
    return 3-d binarized [int] y_data
    """
    y_data[y_data < threshold] = 0.0
    y_data[y_data >= threshold] = 1.0
    return y_data.int()


def _argmax(y_data, dim):
    """
    args:
        y_data : 4-d tensor in [batch_size, chs, img_rows, img_cols]
        dim : int
    return 3-d [int] y_data
    """
    return torch.argmax(y_data, dim).long()


def _get_tp(y_pred, y_true):
    """
    args:
        y_true : [int] 3-d in [batch_size, img_rows, img_cols]
        y_pred : [int] 3-d in [batch_size, img_rows, img_cols]
    return [float] true_positive
    """
    return torch.sum(y_true * y_pred).float()


def _get_fp(y_pred, y_true):
    """
    args:
        y_true : 3-d ndarray in [batch_size, img_rows, img_cols]
        y_pred : 3-d ndarray in [batch_size, img_rows, img_cols]
    return [float] false_positive
    """
    return torch.sum((1 - y_true) * y_pred).float()


def _get_tn(y_pred, y_true):
    """
    args:
        y_true : 3-d ndarray in [batch_size, img_rows, img_cols]
        y_pred : 3-d ndarray in [batch_size, img_rows, img_cols]
    return [float] true_negative
    """
    return torch.sum((1 - y_true) * (1 - y_pred)).float()


def _get_fn(y_pred, y_true):
    """
    args:
        y_true : 3-d ndarray in [batch_size, img_rows, img_cols]
        y_pred : 3-d ndarray in [batch_size, img_rows, img_cols]
    return [float] false_negative
    """
    return torch.sum(y_true * (1 - y_pred)).float()


class OAAcc(object):
    def __init__(self, des = "Overall Accuracy", thresh = 0.5):
        self.des = des
        self.thresh = 0.5

    def __repr__(self):
        return "OAcc"

    def __call__(self, y_pred, y_true):
        """
        args:
            y_true : 4-d ndarray in [batch_size, chs, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, chs, img_rows, img_cols]
            threshold : [0.0, 1.0]
        return (tp+tn)/total
        """
        y_pred = _argmax(y_pred, 1)
        nb_tp_tn = torch.sum(y_true == y_pred)
        mperforms = nb_tp_tn / batch_size
        return mperforms


class Precision(object):
    def __init__(self, des = "Precision", thresh = 0.5):
        self.des = des
        self.thresh = 0.5

    def __repr__(self):
        return "Prec"

    def __call__(self, y_pred, y_true):
        """
        args:
            y_true : 2-d ndarray in [batch_size, chs]
            y_pred : 2-d ndarray in [batch_size, chs]
        return mperforms
        """
        batch_size, chs = y_pred.shape
        device = y_true.device
        y_pred = _argmax(y_pred, 1)
        performs = torch.zeros(chs).to(device)
        for ch in range(chs):
            y_true_ch = torch.zeros(batch_size, 1)
            y_pred_ch = torch.zeros(batch_size, 1)
            y_true_ch[y_true == ch] = 1
            y_pred_ch[y_pred == ch] = 1
            nb_tp = _get_tp(y_pred_ch, y_true_ch)
            nb_fp = _get_fp(y_pred_ch, y_true_ch)
            performs[int(ch)] = nb_tp / (nb_tp + nb_fp + esp)
        mperforms = sum(performs) / chs
        return mperforms


class Recall(object):
    def __init__(self, des = "Recall", thresh = 0.5):
        self.des = des
        self.thresh = 0.5

    def __repr__(self):
        return "Reca"

    def __call__(self, y_pred, y_true):
        """
        args:
            y_true : 2-d ndarray in [batch_size, chs]
            y_pred : 2-d ndarray in [batch_size, chs]
        return mperforms
        """
        batch_size, chs = y_pred.shape
        device = y_true.device
        y_pred = _argmax(y_pred, 1)
        performs = torch.zeros(chs).to(device)
        for ch in range(chs):
            y_true_ch = torch.zeros(batch_size, 1)
            y_pred_ch = torch.zeros(batch_size, 1)
            y_true_ch[y_true == ch] = 1
            y_pred_ch[y_pred == ch] = 1
            nb_tp = _get_tp(y_pred_ch, y_true_ch)
            nb_fn = _get_fn(y_pred_ch, y_true_ch)
            performs[int(ch)] = nb_tp / (nb_tp + nb_fn + esp)
        mperforms = sum(performs) / chs
        return mperforms


class F1Score(object):
    def __init__(self, des = "F1Score", thresh = 0.5):
        self.des = des
        self.thresh = 0.5

    def __repr__(self):
        return "F1Sc"

    def __call__(self, y_pred, y_true):
        """
        args:
            y_true : 2-d ndarray in [batch_size, chs]
            y_pred : 2-d ndarray in [batch_size, chs]
        return mperforms
        """
        batch_size, chs = y_pred.shape
        device = y_true.device
        y_pred = _argmax(y_pred, 1)
        performs = torch.zeros(chs).to(device)
        for ch in range(chs):
            y_true_ch = torch.zeros(batch_size, 1)
            y_pred_ch = torch.zeros(batch_size, 1)
            y_true_ch[y_true == ch] = 1
            y_pred_ch[y_pred == ch] = 1
            nb_tp = _get_tp(y_pred_ch, y_true_ch)
            nb_fp = _get_fp(y_pred_ch, y_true_ch)
            nb_fn = _get_fn(y_pred_ch, y_true_ch)
            _precision = nb_tp / (nb_tp + nb_fp + esp)
            _recall = nb_tp / (nb_tp + nb_fn + esp)
            performs[int(ch)] = 2 * _precision * \
                _recall / (_precision + _recall + esp)
        mperforms = sum(performs) / chs
        return mperforms


class Kappa(object):
    def __init__(self, des = "Kappa", thresh = 0.5):
        self.des = des
        self.thresh = 0.5

    def __repr__(self):
        return "Kapp"

    def __call__(self, y_pred, y_true):
        """
        args:
            y_true : 2-d ndarray in [batch_size, chs]
            y_pred : 2-d ndarray in [batch_size, chs]
        return mperforms
        """
        batch_size, chs = y_pred.shape
        device = y_true.device
        y_pred = _argmax(y_pred, 1)
        performs = torch.zeros(chs).to(device)
        for ch in range(chs):
            y_true_ch = torch.zeros(batch_size, 1)
            y_pred_ch = torch.zeros(batch_size, 1)
            y_true_ch[y_true == ch] = 1
            y_pred_ch[y_pred == ch] = 1
            nb_tp = _get_tp(y_pred_ch, y_true_ch)
            nb_fp = _get_fp(y_pred_ch, y_true_ch)
            nb_tn = _get_tn(y_pred_ch, y_true_ch)
            nb_fn = _get_fn(y_pred_ch, y_true_ch)
            nb_total = nb_tp + nb_fp + nb_tn + nb_fn
            Po = (nb_tp + nb_tn) / nb_total
            Pe = ((nb_tp + nb_fp) * (nb_tp + nb_fn)
                  + (nb_fn + nb_tn) * (nb_fp + nb_tn)) / (nb_total**2)
            performs[int(ch)] = (Po - Pe) / (1 - Pe + esp)
        mperforms = sum(performs) / chs
        return mperforms


class Jaccard(object):
    def __init__(self, des = "Jaccard", thresh = 0.5):
        self.des = des
        self.thresh = 0.5

    def __repr__(self):
        return "Jacc"

    def __call__(self, y_pred, y_true):
        """
        args:
            y_true : 2-d ndarray in [batch_size, chs]
            y_pred : 2-d ndarray in [batch_size, chs]
        return mperforms
        """
        batch_size, chs = y_pred.shape
        device = y_true.device
        y_pred = _argmax(y_pred, 1)
        performs = torch.zeros(chs).to(device)
        for ch in range(chs):
            y_true_ch = torch.zeros(batch_size, 1)
            y_pred_ch = torch.zeros(batch_size, 1)
            y_true_ch[y_true == ch] = 1
            y_pred_ch[y_pred == ch] = 1
            _intersec = torch.sum(y_true_ch * y_pred_ch).float()
            _sum = torch.sum(y_true_ch + y_pred_ch).float()
            performs[int(ch)] = _intersec / (_sum - _intersec + esp)
        mperforms = sum(performs) / chs
        return mperforms


def generate_data(batch_size, chs):
    """
    args:
        batch_size : int
        chs : int
    return y_pred_fake, y_true_fake
    """
    y_true_fake = torch.zeros(batch_size, chs)
    y_pred_fake = torch.zeros(batch_size, chs)
    for i in range(batch_size):
        j = random.randint(0,2)
        k = random.randint(0,2)
        y_true_fake[i,j] = 1.0
        y_pred_fake[i,k] = 1.0
    return y_pred_fake, y_true_fake


if __name__ == "__main__":
    # test
    batch_size = 32
    for cuda in [True, False]:
        y_pred, y_true = generate_data(batch_size, 3)
        y_true = torch.argmax(y_true, 1).long()
        if cuda:
            y_pred = y_pred.cuda()
            y_true = y_true.cuda()

        print(y_pred.shape, y_true.shape)

        metric = OAAcc()
        print('mAccu:', metric(y_pred, y_true))

        metric = Precision()
        print('mPrec:', metric(y_pred, y_true))

        metric = Recall()
        print('mReca:',  metric(y_pred, y_true))

        metric = F1Score()
        print('mF1sc:', metric(y_pred, y_true))

        metric = Kappa()
        print('mKapp:', metric(y_pred, y_true))

        metric = Jaccard()
        print('mJacc:', metric(y_pred, y_true))
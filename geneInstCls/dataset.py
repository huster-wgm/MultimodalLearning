#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
  @Email:  guangmingwu2010@gmail.com
  @Copyright: go-hiroaki
  @License: MIT
"""
import os
import torch
import random
import numpy as np
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset
from transform import compose_transform


class SLICE(Dataset):
    ImageNet_mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
    ImageNet_std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))
    
    def __init__(self, setting: dict, phenotype: str, thresh: float, transform=False, balancing=False):
        """
        Args:
            root (str): root dir of dataset
        """
        # get datalist
        self.root = setting['root']
        self.imgs, self.ofeats, self.tfeats, self.labels = [], [], [], []
        self.probas = []
        self.clinical = pd.read_csv(os.path.join(self.root, 'clinical.csv'))
        cases = setting['cases']

        for case in cases:
            label = self.clinical[self.clinical['ID'] == case][phenotype].iloc[0]
            df = pd.read_csv(os.path.join('./experiments', setting['experiment'], 'preds/cluster', f"{case}.csv"))
            df.sort_values(by=['proba'], ascending=False, inplace=True, ignore_index=True)
            limit = 10
            for i, rec in df.iterrows():
                if rec['proba'] >= thresh or limit > i:
                    self.imgs.append(f'slices/{case}/{rec["filename"]}')
                    self.probas.append(rec['proba'])
                    self.ofeats.append(f'openclip/{case}/{rec["filename"].replace(".jpg", ".npy")}')
                    self.tfeats.append(f'tokenizer/{case}.npy')
                    self.labels.append(int(label))

        if phenotype == "Diagnosis":
            self.nb_cls = 3
        else:
            self.nb_cls = 2
        # label balancing
        if balancing and self.nb_cls != 3:
            self.label_balancing()
        # transform
        if transform:
            self.transform = compose_transform()
        else:
            self.transform = None

    def __len__(self):
        return len(self.imgs)
    
    def repeat(self, v, n):
#         print(f"Repeat >> {'positive' if label == 1 else 'negative'} x {n}")
        for idx in range(len(self.labels)):
            if self.labels[idx] == v:
                self.imgs += [self.imgs[idx]] * n
                self.probas += [self.probas[idx]] * n
                self.ofeats += [self.ofeats[idx]] * n
                self.tfeats += [self.tfeats[idx]] * n
                self.labels += [self.labels[idx]] * n
        # shuffle
        idxs = list(range(len(self.imgs)))
        random.shuffle(idxs)
        self.imgs = [self.imgs[i] for i in idxs]
        self.probas = [self.probas[i] for i in idxs]
        self.ofeats = [self.ofeats[i] for i in idxs]
        self.tfeats = [self.tfeats[i] for i in idxs]
        self.labels = [self.labels[i] for i in idxs]

    def label_balancing(self):
        posi, nega = sum(self.labels), len(self.labels) - sum(self.labels)
        ratio = nega / posi
        if ratio <= 2:
            return 0 # do-nothing
        else:
            self.repeat(1, int(ratio / 2 + 0.5))

    def __getitem__(self, idx):
        imgfile = os.path.join(self.root, self.imgs[idx])
        img = Image.open(imgfile)
        if self.transform is not None:
            img = self.transform(img)
        img = np.asarray(img) / 255
        img = img - self.ImageNet_mean
        img = img / self.ImageNet_std
        src = torch.from_numpy(img.transpose((2, 0, 1))).float()
        openclip_feat = torch.from_numpy(np.load(os.path.join(self.root, self.ofeats[idx]))[0])
        tokenizer_feat = torch.from_numpy(np.load(os.path.join(self.root, self.tfeats[idx]))[0])
        # print(openclip_feat.size(), tokenizer_feat.size())
        feat = torch.cat([openclip_feat, tokenizer_feat], dim = 0)
        # print(openclip_feat.size(), tokenizer_feat.size(), feat.size())
        
        tar = torch.from_numpy(np.array(self.labels[idx])).long()
        return {'src': src, 'feat': feat, 'tar': tar}
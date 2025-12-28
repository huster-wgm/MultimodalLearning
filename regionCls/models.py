#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
  @Email:  guangmingwu2010@gmail.com
  @Copyright: go-hiroaki
  @License: MIT
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor


class bResNet(torch.nn.Module):
    # https://pytorch.org/vision/stable/models.html
    def __init__(self, nb_class=1, opt='resnet18'):
        super(bResNet, self).__init__()
        # extraction
        m = eval(f'models.{opt}')(weights='DEFAULT')
        self.body = create_feature_extractor(
            m, return_nodes={'avgpool': 'body'})
        # self.transform = nn.MultiheadAttention(embed_dim, num_heads)
        # prediction
        self.predict = nn.Sequential(
            nn.Linear(512+1024, 512, bias=False),
            nn.Dropout(0.5),
            nn.Linear(512, nb_class, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x, f):
        n, c, w, h = x.size()
        x = self.body(x)['body']
        # print('x:', x.size(), 'f:', f.size())
        x = x.reshape((n, -1))
        x = torch.cat([x, f], dim=1)
        out = self.predict(x)
        return out
    
    
class bEfficientNet(torch.nn.Module):
    def __init__(self, nb_class=1, opt='efficientnet_b0'):
        super(bEfficientNet, self).__init__()
        # extraction
        m = eval(f'models.{opt}')(weights='DEFAULT')
        self.body = create_feature_extractor(
            m, return_nodes={'avgpool': 'body'})
        # prediction
        self.predict = nn.Sequential(
            nn.Linear(1280+1024, 512, bias=True),
            nn.Dropout(0.5),
            nn.Linear(512, nb_class, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x, f):
        n, c, w, h = x.size()
        x = self.body(x)['body']
        # print('x:', x.size(), 'f:', f.size())
        x = x.reshape((n, -1))
        x = torch.cat([x, f], dim=1)
        out = self.predict(x)
        return out


def build_model(name, nb_class, opt):
    return eval(name)(nb_class, opt)
    
    
def main():
    x_tensor = torch.rand(1, 3, 512, 512)
    f_tensor = torch.rand(1, 1024)
    m = build_model('bResNet', 1, 'resnet18')
    out = m(x_tensor, f_tensor)
    print('bResNet', x_tensor.shape, f_tensor.shape, out.shape)
    m = build_model('bEfficientNet', 1, 'efficientnet_b0')
    out = m(x_tensor,f_tensor)
    print('bEfficientNet', x_tensor.shape, f_tensor.shape, out.shape)


if __name__ == '__main__':
    main()
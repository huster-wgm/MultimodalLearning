#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
  @Email:  guangmingwu2010@gmail.com
  @Copyright: go-hiroaki
  @License: MIT
"""
import os
import time
import torch
import argparse

from dataset import SLICE
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.manual_seed(42)


category = {0: 'Background', 1: 'Cells'}

    

class Predictor(object):
    def __init__(self, ckpt: str, cuda: bool):
        from models import build_model
        net = build_model('bEfficientNet', 1, 'efficientnet_b0')
        net.load_state_dict(torch.load(ckpt))
        if cuda:
            net.cuda()
        self.net = net.eval()
    
    @staticmethod
    def _binarize(y_data, threshold):
        """
        args:
            y_data : [float] 4-d tensor in [batch_size, chs, img_rows, img_cols]
            threshold : [float] [0.0, 1.0]
        return 3-d binarized [int] y_data
        """
        y_data[y_data < threshold] = 0.0
        y_data[y_data >= threshold] = 1.0
        return y_data
            
    def predict(self, dataset, batch_size, cuda):
        # setup data loader
        data_loader = DataLoader(dataset, batch_size, num_workers=4,
                                 shuffle=False, pin_memory=True,)
        start = time.time()
        ys, gen_ys = [], []
        with torch.set_grad_enabled(False):
            for idx, sample in enumerate(data_loader):
                # get tensors from sample
                x, f, y = sample["src"], sample["feat"], sample["tar"]
                if cuda:
                    x = x.cuda()
                    f = f.cuda()
                    y = y.cuda()
                # forwading
                gen_y = self.net(x, f)
                # print(gen_y.shape)
                ys.append(y.detach().cpu())
                gen_ys.append(gen_y.detach().cpu())
        # loss & acc
        ys = torch.cat(ys, dim = 0).numpy()
        gen_ys = torch.cat(gen_ys, dim = 0).numpy()
        print(f"Time >> {time.time() - start : 0.2f}\n")
        # binarized
        binary = self._binarize(gen_ys, 0.5)
        print(ys.shape, binary.shape)
        print(classification_report(ys.astype("uint8"), binary.astype("uint8")))

    
def main(args):
    predictor = Predictor(args.checkpoint, args.cuda)
    print(f"\n\nEvaluating on >> validation set \n")
    val_set = SLICE(args.ip_dir, 'val', True)
    # print(f"\n\nEvaluating on >> testing set \n")
    # tst_set = SLICE(args.ip_dir, 'tst', True)
    # predictor.predict(tst_set, args.batch_size, args.cuda)
    return 0
    
              
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ArgumentParser')
    parser.add_argument('--ip_dir', type=str, default='TCGA-COADx5', 
                        help='root dir of dataset for training models')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint/fewshot_iter_1000.pth',
                        help='trigger type for logging')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch_size for training ')
    parser.add_argument('--cuda', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='using cuda for optimization')
    args = parser.parse_args()
    main(args)

            
        
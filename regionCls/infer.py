#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
  @Email:  guangmingwu2010@gmail.com
  @Copyright: go-hiroaki
  @License: MIT
"""

import os
import glob
import json
import time
import torch
import argparse
import numpy as np

from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from shapely.geometry import box


category = {0: 'Background', 1: 'Cells'}

    
class SLICE(Dataset):
    """ SLICE datasets """
    ImageNet_mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
    ImageNet_std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))


    def __init__(self, root: str, case: str, transform=False):
        """
        Args:
            root (str): root dir of dataset
            case (str): case id
        """
        # get datalist
        self.root = root
        with open(os.path.join(root, 'slices', f"{case}.txt"), "r") as f:
            files = [line.strip() for line in f.readlines()]
        self.imgs = [os.path.join('slices', case, f) for f in files]
        self.feats = [os.path.join('openclip', case, f.replace(".jpg", ".npy")) for f in files]
        
        # color transform
        if transform:
            from transform import CenterCrop
            self.transform = CenterCrop()
        else:
            self.transform = None

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        imgfile = os.path.join(self.root, self.imgs[idx])
        img = Image.open(imgfile)
        if self.transform is not None:
            img = self.transform(img)
        img = np.asarray(img) / 255
        img = img - self.ImageNet_mean
        img = img / self.ImageNet_std
        src = torch.from_numpy(img.transpose((2, 0, 1))).float()
        feat = torch.from_numpy(np.load(os.path.join(self.root, self.feats[idx]))[0])
        return {'src': src, 'feat': feat}
    

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
            
    def infer(self, dataset, batch_size, cuda):
        # setup data loader
        data_loader = DataLoader(dataset, batch_size, num_workers=4,
                                 shuffle=False, pin_memory=True,)
        gen_ys = []
        with torch.set_grad_enabled(False):
            for idx, sample in enumerate(data_loader):
                # get tensors from sample
                x, f = sample["src"], sample["feat"]
                if cuda:
                    x = x.cuda()
                    f = f.cuda()
                # forwading
                gen_y = self.net(x, f).detach()
                if cuda:
                    gen_y = gen_y.cpu()
                gen_ys.append(gen_y)
        # loss & acc
        gen_ys = torch.cat(gen_ys, dim = 0).numpy()
        return gen_ys

    
def main(args):
    # init cfg
    cfg = os.path.join('./experiments', args.experiment, 'config.json')
    with open(cfg, 'r') as f:
        expr = json.load(f)

    predictor = Predictor(f'./checkpoint/{expr["cluster_expr"]["ckpt"]}', args.cuda)
    save_dir = os.path.join('./experiments', args.experiment, 'preds', 'cluster')
    os.makedirs(save_dir, exist_ok=True)
    
    for sp in ['trn', 'val', 'tst']:
        setting = expr['inst_expr'][f"{sp}_set"]
        for idx, case in enumerate(setting['cases']):
            slcdata = SLICE(setting['root'], case, transform=True)
            t0 = time.time()
            gen_ys = predictor.infer(slcdata, args.batch_size, args.cuda)
            print(f"[{idx:03d}/{setting['num_cases']}] Inferring {case} with {len(slcdata)} @ {time.time() - t0 : 0.1f} sec")
            # save result in csv file
            with open(os.path.join(save_dir, f"{case}.csv"), "w") as fp:
                fp.write(f"filename,category_id,proba,WKT\n")
                for prob, imgfile in zip(gen_ys, slcdata.imgs):
                    cat_id = 0 if prob[0] < 0.5 else 1
                    # cat_name = category[cat_id]
                    fname = os.path.basename(imgfile)
                    x, y, w, h = map(int, fname.split("_")[1:5])
                    bbox = box(x, -y, x + w, -y -h)
                    fp.write(f"{fname},{cat_id},{prob[0]:0.3f},\"{bbox.wkt}\"\n")
    return 0
    
              
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ArgumentParser')
    parser.add_argument('--experiment', type=str, required=True,
                        help='trigger type for logging')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch_size for training ')
    parser.add_argument('--cuda', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='using cuda for optimization')
    args = parser.parse_args()
    main(args)

            
        

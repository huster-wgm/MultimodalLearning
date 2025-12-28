#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
  @Email:  guangmingwu2010@gmail.com
  @Copyright: go-hiroaki
  @License: MIT
"""

import os
import json
import time
import torch
import argparse
import numpy as np
import pandas as pd

from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from shapely.geometry import box
from transform import CenterCrop

    
class SLICE(Dataset):
    """ SLICE datasets """
    ImageNet_mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
    ImageNet_std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))


    def __init__(self, root: str, experiment: str, case: str, phenotype: str, thresh: float):
        """
        Args:
            root (str): root dir of dataset
            case (str): case id
        """
        # get datalist
        self.root = root
        self.imgs, self.ofeats, self.tfeats = [], [], []
        self.probas = []
        self.clinical = pd.read_csv(os.path.join(self.root, 'clinical.csv'))
        # print(root, case, phenotype)
        self.label = self.clinical[self.clinical['ID'] == case][phenotype].iloc[0]
        df = pd.read_csv(os.path.join('./experiments', experiment, 'preds/cluster', f"{case}.csv"))
        df.sort_values(by=['proba'], ascending=False, inplace=True, ignore_index=True)
        limit = 10
        for i, rec in df.iterrows():
            if rec['proba'] >= thresh or limit > i:
                self.imgs.append(f'slices/{case}/{rec["filename"]}')
                self.probas.append(rec['proba'])
                self.ofeats.append(f'openclip/{case}/{rec["filename"].replace(".jpg", ".npy")}')
                self.tfeats.append(f'tokenizer/{case}.npy')

        # color transform
        self.transform = CenterCrop()

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
        openclip_feat = torch.from_numpy(np.load(os.path.join(self.root, self.ofeats[idx]))[0])
        tokenizer_feat = torch.from_numpy(np.load(os.path.join(self.root, self.tfeats[idx]))[0])
        # print(openclip_feat.size(), tokenizer_feat.size())
        feat = torch.cat([openclip_feat, tokenizer_feat], dim = 0)
        return {'src': src, 'feat': feat}
    

class Predictor(object):
    def __init__(self, ckpt: str, nb_cls: int, cuda: bool):
        from models import build_model
        net = build_model('bEfficientNet', nb_cls, 'efficientnet_b0')
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
        gen_ys = torch.softmax(torch.cat(gen_ys, dim = 0), dim = 1).numpy()
        return gen_ys

    
def main(args):
    # init cfg
    cfg = os.path.join('./experiments', args.experiment, 'config.json')
    with open(cfg, 'r') as f:
        expr = json.load(f)
    
    save_dir = os.path.join('./experiments', args.experiment, 'preds', args.phenotype)
    os.makedirs(save_dir, exist_ok=True)

    # init predictor
    if args.phenotype == "Diagnosis":
        nb_cls = 3
    else:
        nb_cls = 2
    proba_thresh = 0.80
    predictor = Predictor(f'./checkpoint/{expr["inst_expr"][f"{args.phenotype}_ckpt"]}', nb_cls, args.cuda)
    
    # start infer
    for sp in ['trn', 'val', 'tst']:
        setting = expr['inst_expr'][f"{sp}_set"]
        for idx, case in enumerate(setting['cases']):
            infer_set = SLICE(setting['root'], args.experiment, case, args.phenotype, proba_thresh)
            t0 = time.time()
            gen_ys = predictor.infer(infer_set, args.batch_size, args.cuda)
            print(f"[{idx:03d}/{setting['num_cases']}] Inferring {case} with {len(infer_set)} @ {time.time() - t0 : 0.1f} sec")
            # save result in csv file
            with open(os.path.join(save_dir, f"{case}.csv"), "w") as fp:
                probas = ",".join([f'proba_{i}' for i in range(nb_cls)])
                fp.write(f"filename,c_prob,{probas},label,WKT\n")
                for prob, c_prob, imgfile in zip(gen_ys, infer_set.probas, infer_set.imgs):
                    # cat_id = 0 if prob < 0.5 else 1
                    # cat_name = category[cat_id]
                    fname = os.path.basename(imgfile)
                    x, y, w, h = map(int, fname.split("_")[1:5])
                    bbox = box(x, -y, x + w, -y -h)
                    probas = ",".join([f'{i:0.3f}' for i in prob])
                    fp.write(f"{fname},{c_prob},{probas},{infer_set.label},\"{bbox.wkt}\"\n")
    return 0
    
              
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ArgumentParser')
    parser.add_argument('--experiment', type=str, required=True,
                        help='trigger type for logging')
    parser.add_argument('--phenotype', type=str, required=True,
                        help='pheno type of clinical info')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch_size for training ')
    parser.add_argument('--cuda', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='using cuda for optimization')
    args = parser.parse_args()
    main(args)

            
        
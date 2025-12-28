#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
  @Email:  guangmingwu2010@gmail.com
  @Copyright: go-hiroaki
  @License: MIT
"""

import os
import json
import argparse
import numpy as np

from aucci import auc_ci_Delong
from sklearn import metrics
from dataset import SLICE


def binarized(x, thresh=0.5):
    return [1 if i > thresh else 0 for i in x]


class EVALUATOR(object):
    def __init__(self, nb_cls, preds, ys):
        self.performs = {}
        self.performs['num'] = len(ys)
        
        for cls in range(nb_cls):
            p = [i[cls] for i in preds]
            y = [1 if i == cls else 0 for i in ys]
            perf = self.evaluate(p, y)
            self.performs[f'num_pos_{cls}'] = sum(y)
            self.performs[f'cls_{cls}'] = perf

    def evaluate(self, p, y):
        perform = {}
        perform.update(self.recall(p, y))
        perform.update(self.precision(p, y))
        perform.update(self.overall(p, y))
        perform.update(self.f1score(p, y))
        perform.update(self.auc_ci(p, y))
        perform.update(self.auc(p, y))
        return perform

    @staticmethod
    def auc(p, y):
        fpr, tpr, _ = metrics.roc_curve(y, p, pos_label=1)
#         auc = metrics.auc(fpr, tpr)
        return {'fpr': list(fpr), 'tpr': list(tpr)}
    
    @staticmethod
    def auc_ci(p, y):
        y = np.array(binarized(y)).astype(int)
        p = np.array(p)
        auc, auc_var, ci = auc_ci_Delong(
            y_true=y,
            y_scores=p)
#         print(auc, auc_var, ci)
        return {'CI': list(ci), 'auc_var': float(auc_var), 'auc': float(auc)}

    @staticmethod
    def precision(p, y):
        p, y = binarized(p), binarized(y)
        prec = metrics.precision_score(y, p, pos_label=1)
        return {'prec': prec}

    @staticmethod
    def recall(p, y):
        p, y = binarized(p), binarized(y)
        reca = metrics.recall_score(y, p, pos_label=1)
        return {'reca': reca}
    
    @staticmethod
    def overall(p, y):
        p, y = binarized(p), binarized(y)
        accu = metrics.accuracy_score(y, p)
        return {'accu': accu}
    
    @staticmethod
    def f1score(p, y):
        p, y = binarized(p), binarized(y)
        f1sc = metrics.f1_score(y, p, pos_label=1)
        return {'f1sc': f1sc}


def main(args):
    # init cfg
    cfg = os.path.join('./experiments', args.experiment, 'config.json')
    with open(cfg, 'r') as f:
        expr = json.load(f)

    save_dir = os.path.join('./experiments', args.experiment, 'json')
    os.makedirs(save_dir, exist_ok=True)

    # init datasets
    for sp in ['trn', 'val', 'tst']:
        setting = expr['inst_expr'][f"{sp}_set"]
        sp_data = SLICE(setting, expr['name'], args.phenotype, args.cls)
        print(f"Dataset : {setting['root']} ==> {sp} : {setting['num_cases']};")
        # init evaluator
        evaluator = EVALUATOR(sp_data.nb_cls, sp_data.p, sp_data.y)
        with open(f'{save_dir}/{args.cls}_{args.phenotype}_{sp}.json', 'w') as f:
            json.dump(evaluator.performs, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ArgumentParser')
    parser.add_argument('--experiment', type=str, required=True,
                        help='trigger type for logging')
    parser.add_argument('--cls', type=str, default='PTC', choices=['all', 'PTC', 'AUS', 'BEN'], 
                        help='diagnostic classification')
    parser.add_argument('--phenotype', type=str, default='MSI', 
                        help='target gene&msi for training models')
    args = parser.parse_args()
    main(args)
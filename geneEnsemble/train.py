#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
  @Email:  guangmingwu2010@gmail.com
  @Copyright: go-hiroaki
  @License: MIT
"""

import os
import argparse
import numpy as np
from dataset import SLICE

from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import StratifiedKFold



def main(args):
    save_dir = f'{args.ip_dir}/json'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # init datasets
    trn_set = SLICE(args.ip_dir, 'trn', args.phenotype, 'all')
    val_set = SLICE(args.ip_dir, 'val', args.phenotype, 'all')
    tst_ptc = SLICE(args.ip_dir, 'tst', args.phenotype, 'PTC')
    tst_aus = SLICE(args.ip_dir, 'tst', args.phenotype, 'AUS')
    print(f"Dataset : {args.ip_dir} ==> trn : {len(trn_set)}; val : {len(val_set)}")
    # init evaluator
    evaluator = metrics.roc_auc_score
    # init k-fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # # init svm model
    # clf = svm.SVC(probability=True)
    # clf.fit(trn_set.p, trn_set.y)
    # # train
    # trn_yp = clf.predict_proba(trn_set.p)[:,1] 
    # val_yp = clf.predict_proba(val_set.p)[:,1]
    # eval_trn = evaluator(trn_set.y, trn_yp)
    # eval_val = evaluator(val_set.y, val_yp)
    # print(f"Train AUC : {eval_trn}; Val AUC : {eval_val}")

    # # test
    # ptc_yp = clf.predict_proba(tst_ptc.p)[:,1] 
    # aus_yp = clf.predict_proba(tst_aus.p)[:,1]
    eval_ptc = evaluator(tst_ptc.y, tst_ptc.p)
    eval_aus = evaluator(tst_aus.y, tst_aus.p)
    print(f"Test AUS-AUC : {eval_ptc}; Test AUS-AUC : {eval_aus}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ArgumentParser')
    parser.add_argument('--ip_dir', type=str, default='TCGA-COAD', 
                        help='root dir of dataset for training models')
    parser.add_argument('--cls', type=str, default='PTC', choices=['PTC', 'AUS', 'BEN'], 
                        help='diagnostic classification')
    parser.add_argument('--phenotype', type=str, default='MSI', 
                        help='target gene&msi for training models')
    args = parser.parse_args()
    main(args)
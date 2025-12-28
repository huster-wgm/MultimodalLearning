#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
  @Email:  guangmingwu2010@gmail.com
  @Copyright: go-hiroaki
  @License: MIT
"""
import os
import json
import torch
import argparse
from dataset import SLICE

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.manual_seed(42)


def init_model(nb_cls, cuda, lr):
    import torch.optim as optim
    from models import build_model
    net = build_model('bEfficientNet', nb_cls , 'efficientnet_b0')
    if cuda:
        net.cuda()
    net.optimizer = optim.Adam(
        net.parameters(), lr=lr)
    return net


def init_trainer(args, expr):
    from runner import Trainer
    return Trainer(args, expr)


def main(args):
    if args.cuda and not torch.cuda.is_available():
        raise ValueError("GPUs are not available, please run at cpu mode")

    # init cfg
    cfg = os.path.join('./experiments', args.experiment, 'config.json')
    with open(cfg, 'r') as f:
        expr = json.load(f)
    expr_setting = expr['inst_expr']
    expr_setting['trn_set']['experiment'] = args.experiment
    expr_setting['val_set']['experiment'] = args.experiment

    # init datasets
    proba_thresh = 0.80
    trn_set = SLICE(expr_setting['trn_set'], args.phenotype, proba_thresh, True, True)
    val_set = SLICE(expr_setting['val_set'], args.phenotype, proba_thresh, True)

    print("Dataset : ==> Trn : {} ; Val : {} ;".format(len(trn_set), len(val_set)))

    # initialize network
    net = init_model(trn_set.nb_cls, args.cuda, args.lr)

    # initialize runner
    run = init_trainer(args, f"{args.experiment}_inst_{args.phenotype}")
    print(f"Start training {args.experiment}_inst_{args.phenotype} ...")

    run.training(net, [trn_set, val_set])
    run.save_log()
    run.learning_curve()

    expr['inst_expr'][f'{args.phenotype}_ckpt'] = "{}.pth".format(run.repr)
    
    with open(cfg, 'w', encoding='utf-8') as f:
        json.dump(expr, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ArgumentParser')
    parser.add_argument('--trigger', type=str, default='epoch', choices=['epoch', 'iter'],
                        help='trigger type for logging')
    parser.add_argument('--interval', type=int, default=1,
                        help='interval for logging')
    parser.add_argument('--terminal', type=int, default=100,
                        help='terminal for training ')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch_size for training ')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate for optimization')
    parser.add_argument('--experiment', type=str, required=True,
                        help='experiment name')
    parser.add_argument('--phenotype', type=str, required=True,
                        help='pheno type of clinical info')
    parser.add_argument('--cuda', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='using cuda for optimization')
    args = parser.parse_args()
    optim_betas = (0.9, 0.999)
    main(args)
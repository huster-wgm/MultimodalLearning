#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
  @Email:  guangmingwu2010@gmail.com
  @Copyright: go-hiroaki
  @License: MIT
"""
import os
import time
import shutil
import losses
import metrics
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


DIR = os.path.dirname(os.path.abspath(__file__))
Logs_DIR = os.path.join(DIR, '../logs')
Checkpoint_DIR = os.path.join(DIR, '../checkpoint')
if not os.path.exists(os.path.join(Logs_DIR, 'raw')):
    os.makedirs(os.path.join(Logs_DIR, 'raw'))
if not os.path.exists(os.path.join(Logs_DIR, 'curve')):
    os.makedirs(os.path.join(Logs_DIR, 'curve'))

if not os.path.exists(Checkpoint_DIR):
    os.mkdir(Checkpoint_DIR)


class Base(object):
    def __init__(self, args, expr, loss='CELoss', metric='F1Score'):
        self.args = args
        self.expr = expr
        self.date = time.strftime("%h%d_%H")
        self.repr = "{}_{}_{}".format(
            self.expr, self.args.trigger, self.args.terminal)
        self.epoch = 0
        self.iter = 0
        self.logs = []
        self.criterion = eval('{}.{}()'.format('losses', loss))
        self.evaluator = eval('{}.{}()'.format('metrics', metric))
        self.header = ["epoch", "iter"]
        for stage in ['trn', 'val']:
            for key in [repr(self.criterion),repr(self.evaluator),"FPS"]:
                self.header.append("{}_{}".format(stage, key))

    def logging(self, verbose=True):
        self.logs.append([self.epoch, self.iter] + self.trn_log + self.val_log)
        if verbose:
            str_a = ['{}:{:05d}'.format(k,v) for k,v in zip(self.header[:2], [self.epoch, self.iter])]
            str_b = ['{}:{:.3f}'.format(k,v) for k,v in zip(self.header[2:], self.trn_log + self.val_log)]
            print(', '.join(str_a + str_b))

    def save_log(self):
        self.logs = pd.DataFrame(self.logs,
                                 columns=self.header)
        self.logs.to_csv(os.path.join(Logs_DIR, 'raw', '{}.csv'.format(self.repr)), 
                         index=False, float_format='%.3f')

        speed_info = [self.repr, self.logs.iloc[:, 4].mean(), self.logs.iloc[:, 7].mean()]
        df = pd.DataFrame([speed_info],
                          columns=["experiment", self.header[4], self.header[7]])
        if os.path.exists(os.path.join(Logs_DIR, 'speed.csv')):
            with open(os.path.join(Logs_DIR, 'speed.csv'), 'a') as f:
                f.write(",".join(map(str, speed_info)) + '\n')
        else:
            df.to_csv(os.path.join(Logs_DIR, 'speed.csv'), index=False, float_format='%.3f')

    def save_checkpoint(self, net):
        torch.save(net.state_dict(), os.path.join(Checkpoint_DIR, "{}.pth".format(self.repr)))

    def learning_curve(self, idxs=[2,3,5,6]):
        import seaborn as sns
        import matplotlib.pyplot as plt
        plt.switch_backend('agg')
        # set style
        sns.set_context("paper", font_scale=1.5,)
        # sns.set_style("ticks", {
        #     "font.family": "Times New Roman",
        #     "font.serif": ["Times", "Palatino", "serif"]})

        for idx in idxs:
            plt.plot(self.logs[self.args.trigger],
                     self.logs[self.header[idx]], label=self.header[idx])
        plt.ylabel(" {} / {} ".format(repr(self.criterion), repr(self.evaluator)))
        if self.args.trigger == 'epoch':
            plt.xlabel("Epochs")
        else:
            plt.xlabel("Iterations")
        plt.suptitle("Training log of {}".format(self.expr))
        # remove top&left line
        # sns.despine()
        plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
        plt.savefig(os.path.join(Logs_DIR, 'curve', '{}.png'.format(self.repr)),
                    format='png', bbox_inches='tight', dpi=144)


class Trainer(Base):
    def training(self, net, datasets):
        """
          Args:
            net: (object) net & optimizer
            datasets : (list) [train, val] dataset object
        """
        args = self.args
        best_trn_perform, best_val_perform = -1, -1
        steps = len(datasets[0]) // args.batch_size
        if steps * args.batch_size < len(datasets[0]):
            steps += 1

        if args.trigger == 'epoch':
            args.epochs = args.terminal
            args.iters = steps * args.terminal
            args.iter_interval = steps * args.interval
        else:
            args.epochs = args.terminal // steps + 1
            args.iters = args.terminal
            args.iter_interval = args.interval

        net.train()
        trn_loss, trn_acc = [], []
        start = time.time()
        for epoch in range(1, args.epochs + 1):
            self.epoch = epoch
            # setup data loader
            data_loader = DataLoader(datasets[0], args.batch_size, num_workers=4,
                                     shuffle=True, pin_memory=True,)
            for idx, sample in enumerate(data_loader):
                self.iter += 1
                if self.iter > args.iters:
                    self.iter -= 1
                    break
                # get tensors from sample
                x, f, y = sample["src"], sample["feat"], sample["tar"]
                if args.cuda:
                    x = x.cuda()
                    f = f.cuda()
                    y = y.cuda()
                # forwading
                gen_y = net(x, f)
                # print(gen_y.shape)
                loss = self.criterion(gen_y, y)
                # update parameters
                net.optimizer.zero_grad()
                loss.backward()
                net.optimizer.step()
                # update taining condition
                trn_loss.append(loss.detach().item())
                trn_acc.append(self.evaluator(gen_y.detach(), y.detach()).item())
                # validating
                if self.iter % args.iter_interval == 0:
                    trn_fps = (args.iter_interval * args.batch_size) / (time.time() - start)
                    self.trn_log = [round(sum(trn_loss) / len(trn_loss), 3), 
                                    round(sum(trn_acc) / len(trn_acc), 3),
                                    round(trn_fps, 3)]
                    self.validating(net, datasets[1])
                    self.logging(verbose=True)
                    if self.val_log[1] >= best_val_perform:
                        best_trn_perform = self.trn_log[1]
                        best_val_perform = self.val_log[1]
                        checkpoint_info = [self.repr, self.epoch, self.iter,
                                           best_trn_perform, best_val_perform]
                        # save better checkpoint
                        self.save_checkpoint(net)
                    # reinitialize
                    start = time.time()
                    trn_loss, trn_acc = [], []
                    net.train()

        df = pd.DataFrame([checkpoint_info],
                          columns=["experiment", "best_epoch", "best_iter", self.header[3], self.header[6]])
        if os.path.exists(os.path.join(Checkpoint_DIR, 'checkpoint.csv')):
            with open(os.path.join(Checkpoint_DIR, 'checkpoint.csv'), 'a') as f:
                f.write(",".join(map(str, checkpoint_info)) + '\n')
        else:
            df.to_csv(os.path.join(Checkpoint_DIR, 'checkpoint.csv'), index=False, float_format="%.3f")

        print("Best {} Performance: \n".format(repr(self.evaluator)))
        print("\t Trn:", best_trn_perform)
        print("\t Val:", best_val_perform)

    def validating(self, net, dataset):
        """
          Args:
            net: (object) pytorch net
            batch_size: (int)
            dataset : (object) dataset
          return [loss, acc]
        """
        args = self.args
        data_loader = DataLoader(dataset, args.batch_size, num_workers=4,
                                 shuffle=True, pin_memory=True,)
        start = time.time()
        net.eval()
        ys, gen_ys = [], []
        with torch.set_grad_enabled(False):
            for idx, sample in enumerate(data_loader):
                # get tensors from sample
                x, f, y = sample["src"], sample["feat"], sample["tar"]
                if args.cuda:
                    x = x.cuda()
                    f = f.cuda()
                    y = y.cuda()
                # forwading
                gen_y = net(x, f)
                ys.append(y.detach())
                gen_ys.append(gen_y.detach())
        # loss & acc
        ys = torch.cat(ys, dim = 0)
        gen_ys = torch.cat(gen_ys, dim = 0)
        val_loss = self.criterion(gen_ys, ys).item()
        val_acc = self.evaluator(gen_ys, ys).item()
        # FPS
        val_fps = len(dataset) / (time.time() - start)
        self.val_log = [round(val_loss, 3), 
                        round(val_acc, 3),
                        round(val_fps, 3)]
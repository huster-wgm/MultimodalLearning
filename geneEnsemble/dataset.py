#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
  @Email:  guangmingwu2010@gmail.com
  @Copyright: go-hiroaki
  @License: MIT
"""

import os
import numpy as np
import pandas as pd
    
    
class SLICE(object):
    """ Deep Feature datasets """
    diag_map = {
        'PTC': 0,
        'AUS': 1,
        'BEN': 2
    }

    def filter_by_cls(self, clinical, cls: str, cases: list):
        valid_cases = []
        for case in cases:
            label = clinical[clinical['ID'] == case]['Diagnosis'].iloc[0]
            if self.diag_map[cls] == label: 
                valid_cases.append(case)
        return valid_cases
            
    @staticmethod
    def categorize_into_bins(c_probas: list, l_probas: list, num_bins: int):
        bins = np.linspace(0, 1, num_bins)
        indexs = np.digitize(l_probas, bins)
        # print(len(indexs), len(c_probas), len(l_probas))

        l_freqs = [0] * (num_bins - 1)
        for i, j in enumerate(indexs):
            # print(j, l_probas[i])
            j = min(j, num_bins - 1)
            l_freqs[j-1] += 1
        # normalize
        l_freqs = [f / len(c_probas) for f in l_freqs]

        c_freqs = [[] for _ in range(num_bins - 1)]
        for i, j in enumerate(indexs):
            j = min(j, num_bins - 1)
            c_freqs[j-1].append(c_probas[i])

        # print(c_freqs)
        for i in range(num_bins-1):
            c_freqs[i] = 0.0 if len(c_freqs[i]) == 0 else sum(c_freqs[i]) / len(c_freqs[i])
        
        # l freq weighted by c_freq
        lw_freqs = [l_freqs[i] * c_freqs[i] for i in range(num_bins - 1)]
        sum_lw_freqs = sum(lw_freqs)
        lw_freqs = [lw / sum_lw_freqs for lw in lw_freqs]
        
        # print(l_freqs, '\n', c_freqs, '\n', lw_freqs)
        return l_freqs, c_freqs, lw_freqs
    
    @staticmethod
    def freq2proba(l_freqs: list, c_freqs: list, lw_freqs: list, num_bins: int):
        idx = np.argmax(l_freqs)
        if idx < 0.5 * (num_bins - 1):
            proba = idx / (num_bins - 1) + 0.1 * (1 - l_freqs[idx]) * (num_bins - 1) / 10
        else:
            proba = idx / (num_bins - 1) + 0.1 * l_freqs[idx] * (num_bins - 1) / 10
        # print(idx, proba)
        return proba
        
    def __init__(self, root, cases, experiment: str, phenotype: str, cls: str):
        """
        Args:
            root (str): root dir of dataset
            split (str): trn / val / tst
            cls (str): PTC/AUS/BEN
            phenotype (str): target gene & msi
            topN (int): N number of top
        """

        clinical = pd.read_csv(os.path.join(root, 'clinical.csv'))

        if cls != 'all':
            cases = self.filter_by_cls(clinical, cls, cases)
        self.cases = cases
        
        if phenotype == "Diagnosis":
            self.nb_cls = 3
        else:
            self.nb_cls = 2

        print(f"Number of valid cases in {cls}: {len(cases)}")
        # get datalist
        gene_p, gene_y = [], []
        for idx, case in enumerate(cases):
            # add feature
            df = pd.read_csv(os.path.join('./experiments', experiment, 'preds', phenotype, f'{case}.csv'))
            # print(df.head())
            c_probas = df['c_prob'].tolist()
            l_probas = df['proba_1'].tolist()
            label = df.iloc[0, 4]
            l_freqs, c_freqs, lw_freqs = self.categorize_into_bins(c_probas, l_probas, 11)
            proba = self.freq2proba(l_freqs, c_freqs, lw_freqs, 11)
            gene_p.append([1-proba, proba])
            # add label
            gene_y.append(int(label))
        # print('positive rate >> ', sum(gene_y) / len(gene_y))
        self.p, self.y = np.array(gene_p), np.array(gene_y)
        
    def __len__(self):
        return len(self.y)
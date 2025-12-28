import os
import glob
import json
import shutil 

import numpy as np
import pandas as pd

from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib as mpl

config = {
    "font.family":'Times New Roman',
    'font.size': 12,
    'axes.facecolor': 'white',
    'xtick.direction':'in',
    'ytick.direction':'in',
}
rcParams.update(config)

# font_name = "Yahei Mono"
# mpl.rcParams['font.family']= font_name # 指定字体，实际上相当于修改 matplotlibrc 文件　只不过这样做是暂时的　下次失效
# mpl.rcParams['axes.unicode_minus']=False # 正确显示负号，防止变成方框


def freq2proba(l_freqs: list, c_freqs: list, lw_freqs: list, num_bins: int):
    idx = np.argmax(l_freqs)
    if idx < 0.5 * (num_bins - 1):
        proba = idx / (num_bins - 1) + 0.1 * (1 - l_freqs[idx]) * (num_bins - 1) / 10
    else:
        proba = idx / (num_bins - 1) + 0.1 * l_freqs[idx] * (num_bins - 1) / 10
    # print(idx, proba)
    return proba


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


def find_the_best_n_cases(root_dir, phenotype, cases, n):
    valid_cases, valid_scores = [], []
    for case in cases:
        df = pd.read_csv(f'{root_dir}/preds/{phenotype}/{case}.csv')
        c_probas = df['c_prob'].tolist()
        l_probas = df['proba_1'].tolist()
        label = df.iloc[0, 4]
        l_freqs, c_freqs, lw_freqs = categorize_into_bins(c_probas, l_probas, 11)
        proba = freq2proba(l_freqs, c_freqs, lw_freqs, 11)

        if label == 1 and proba >= 0.8:
            valid_cases.append(case)
            valid_scores.append(proba)
    
    idxs = np.argsort(valid_scores)[::-1]
    
    valid_cases = [valid_cases[i] for i in idxs]
    return valid_cases[:n]


def bins2label(num_bins: int):
    bins = np.linspace(0, 1, num_bins)
    labels = []
    for i in range(len(bins) - 1):
        labels.append(f'{bins[i]:.1f}-{bins[i+1]:.1f}')
    return labels


def main(expr: str):
    iroot = f'{expr}/json'
    oroot = f'{expr}/vis6'

    if os.path.exists(oroot):
        shutil.rmtree(oroot)
    
    os.makedirs(oroot)

    for phenotype in ['BRAF', 'RAS']:
        for sp in ['trn', 'val', 'tst']:
            with open(f'{iroot}/all_{phenotype}_{sp}.json', 'r') as f:
                recs = json.load(f)
            cases = recs['cases']
            valid_cases = find_the_best_n_cases(expr, phenotype, cases, 10)
            for idx, case in enumerate(valid_cases):
                df = pd.read_csv(f'{expr}/preds/{phenotype}/{case}.csv')
                c_probas = df['c_prob'].tolist()
                l_probas = df['proba_1'].tolist()
                label = df.iloc[0, 4]
                l_freqs, c_freqs, lw_freqs = categorize_into_bins(c_probas, l_probas, 11)
                proba = freq2proba(l_freqs, c_freqs, lw_freqs, 11)
                
                # plt.figure(figsize=(10, 5))
                # x1 = [i for i in range(10)]
                # # x2 = [i*2+1 for i in range(10)]
                # plt.bar(x1, l_freqs, color='blue', alpha=0.5)
                # # plt.bar(range(10), c_freqs, color='r', alpha=0.5)
                # # plt.bar(x2, lw_freqs, color='purple', alpha=0.5)
                # # plt.title(f'{case} / {phenotype} = {label}')
                # print(f"Handling >>{oroot} {sp}_{phenotype} {case}")
                # plt.xticks(np.arange(0., 1.1, 0.1), fontproperties='Times New Roman', size=24)
                # plt.yticks(np.arange(0., 1.1, 0.1), fontproperties='Times New Roman', size=24)
                # plt.ylabel('Frequency', fontproperties='Times New Roman', size=32)
                # plt.xlabel('Probability', fontproperties='Times New Roman', size=32)
                # plt.savefig(f"{oroot}/{sp}_{phenotype}_{idx}_{case}.pdf", format="pdf")
                # plt.close()
                
                
                labels = bins2label(11)

                x = np.arange(len(labels))
                width = 0.4

                # fig, ax = plt.subplots(figsize=(10, 5))
                fig, ax = plt.subplots(figsize=(10, 5), tight_layout=True)

                rect = ax.bar(x, l_freqs, width)
                ax.set_xticks(x)
                ax.set_xticklabels(labels, fontproperties='Times New Roman', size=12)
                ax.set_yticks(np.arange(0., 1.1, 0.1), fontproperties='Times New Roman', size=12)
                # ax.legend()
                plt.ylim([0.00, 1.05])
                def autolabel(rects):
                    for rect in rects:
                        height = rect.get_height()
                        ax.annotate(f'{height:0.2f}',
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),
                                size=12,
                                textcoords="offset points",
                                ha='center', va='bottom')
                autolabel(rect)      
                # plt.tight_layout()
                plt.ylabel('Frequency', fontproperties='Times New Roman', size=20)
                plt.xlabel('Patch-level Probability', fontproperties='Times New Roman', size=20)
                plt.savefig(f"{oroot}/{sp}_{phenotype}_{idx}_{case}.pdf", format="pdf")
                plt.close()

    return 0


if __name__ == '__main__':

    for iteration in range(5):
        main(f"./expr_kFold_{iteration}")
    
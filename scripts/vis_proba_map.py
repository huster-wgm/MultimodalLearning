import os
import json
import shutil 
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageOps


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


def categorized(df, cat: int, num: int):
    df.sort_values(by=['proba_1'], ascending=False, inplace=True, ignore_index=True)
    step = 1.0 / cat
    intervals = list(np.arange(0, 1.0, step))[::-1]
    groups = {}
    is_valid = True
    for i, ival in enumerate(intervals):
        low, high = ival, ival + step
        group = df[(low <= df['proba_1']) & (df['proba_1'] < high)]
        # group.sort_values(by=['proba'], ascending=False, inplace=True, ignore_index=True)
        # print(i, group['proba'].tolist())
        files = group['filename'].tolist()
        groups[i] = files[:num]
        if len(files) < num:
            is_valid = False
    return groups, is_valid


def groups_to_one(root_dir, groups, case, ncol, nrow, palette):
    # merge patches to one
    height, width = 512 + 60, 512 + 60
    fusion = np.zeros((height * nrow, width * ncol, 3), dtype=np.uint8)
    for i in range(nrow):
        for j in range(ncol):
            if j >= len(groups[i]):
                patch = Image.new('RGB', (width, height), (255, 255, 255))
            else:
                patch = Image.open(os.path.join(f'{root_dir}/slices', case, groups[i][j]))
                # patch = ImageOps.expand(patch, border=5, fill=(0,0,0))
                patch = ImageOps.expand(patch, border=20, fill=tuple(palette[i]))
                patch = ImageOps.expand(patch, border=10, fill=(240,240,240))
            fusion[i * height: (i + 1) * height, j * width: (j + 1) * width] = np.array(patch)
    fusion = ImageOps.expand(Image.fromarray(fusion), border=10, fill=(255,255,255))
    # gen palette
    legend = np.zeros((height * nrow, width * 1, 3), dtype=np.uint8)
    for i in range(nrow):
        patch = palette[np.ones((512, 512), dtype=np.uint8) * i]
        patch = Image.fromarray(patch)
        patch = ImageOps.expand(patch, border=30, fill=(240,240,240))
        legend[i * height: (i + 1) * height, :] = np.array(patch)
    legend = ImageOps.expand(Image.fromarray(legend), border=10, fill=(255,255,255))
    final = np.concatenate([np.array(fusion), np.array(legend)], axis=1)
    return Image.fromarray(final)


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


if __name__ == "__main__":
    ncol, nrow = 4, 4
    # https://colorhunt.co/palette/ebe4d1b4b4b326577ce55604
    palette = np.array([(229, 86, 4), (38, 87, 124), (180, 180, 179), (235, 228, 209)]).astype('uint8')
    print(palette.shape)
    src_dir = '../dataset/2023-Thyroid'
    for i in range(5):
        expr = f'./expr_kFold_{i}'
        iroot = f'{expr}/json'
        oroot = f'{expr}/vis5'

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
                    groups, is_valid = categorized(df, nrow, ncol)
                    if not is_valid:
                        continue
                    
                    fusion = groups_to_one(src_dir, groups, case, ncol, nrow, palette)
                    print(f"Handling >>{oroot} {sp}_{phenotype} {case}")
                    fusion.save(f"{oroot}/{sp}_{phenotype}_{idx}_{case}.png")

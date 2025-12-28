import os
import glob 
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw, ImageFont


def load_patch(imgfile: str, filename: str, proba: float):
    patch = Image.open(imgfile)
    text = f' {filename} \n prob = {proba:.2f} '
    font = ImageFont.truetype('Arial.ttf', 30)
    # draw text
    draw = ImageDraw.Draw(patch)
    draw.text((250, 75), text, 'red', font=font, anchor='md')
    bbox = draw.textbbox((250, 75), text, font=font, anchor='md')
    # enlarge bbox
    bbox = (bbox[0] - 10, bbox[1] - 10, bbox[2] + 10, bbox[3] + 10)
    draw.rectangle(bbox, outline='red', width=5)
    return patch


def select_top_patches(root: str, case: str, num: int):
    # load preds.csv
    df = pd.read_csv(f'{root}/anno/{case}.csv')
    df.sort_values(by=['category_id'], ascending=False, inplace=True, ignore_index=True)
    top_probas = df['category_id'].tolist()[:num]
    top_filename = df['filename'].tolist()[:num]
    
    # load patches
    img_patches = []
    for filename, proba in zip(top_filename, top_probas):
        imgfile = os.path.join(f'{root}/slices', case, filename)
        img_patches.append(load_patch(imgfile, filename, proba))
        
    return img_patches


def fused_patches(img_patches: list, ncol: int, nrow: int, padding: int):
    height, width = 512 + 2* padding , 512 + 2*padding # patch size
    fusion = np.zeros((height * nrow, width * ncol, 3), dtype=np.uint8)
    for i in range(nrow):
        for j in range(ncol):
            patch = img_patches[i * ncol + j]
            patch = ImageOps.expand(patch, border=padding, fill=(240,240,240))
            fusion[i * height: (i + 1) * height, j * width: (j + 1) * width] = np.array(patch)
    fusion = ImageOps.expand(Image.fromarray(fusion), border=padding, fill=(240,240,240))
    return fusion


def main(root_dir: str):
    ncol, nrow = 8, 5
    num = ncol * nrow
    with open(os.path.join(root_dir, "trn-few_all.txt"), "r") as f:
        cases = [line.strip() for line in f.readlines()]

    for case in cases:
        img_patches = select_top_patches(root_dir, case, num)
        fusion = fused_patches(img_patches, ncol, nrow, padding=15)
        print("Saving fused image...", case)
        fusion.save(f'{root_dir}/anno/{case}.png')

    return 0


if __name__ == '__main__':
    datasets = ['./dataset/2023-Thyroid']
    for data in datasets:
        main(data)
    
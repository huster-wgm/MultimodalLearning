
import os
import glob
import torch
import open_clip
import numpy as np
from PIL import Image


#root = "./dataset/2023-Thyroid"  #SYSU8H 
root = "./dataset/GY20240226"  #GY
model, _, preprocess = open_clip.create_model_and_transforms('RN50', pretrained='openai')
model.cuda()

os.makedirs(os.path.join(root, 'openclip'), exist_ok=True)
with torch.no_grad(), torch.cuda.amp.autocast():
    with open(os.path.join(root, 'all.txt'), 'r') as f:
        recs = [line.strip() for line in f.readlines()]
    for idx, rec in enumerate(recs):
        files = glob.glob(os.path.join(root, 'slices', rec, '*.jpg'))
        os.makedirs(os.path.join(root, 'openclip', rec), exist_ok=True)
        print(f"Handling >> {rec}, {idx}/{len(recs)}")
        for file in files:
            filename = os.path.basename(file).replace('.jpg', '.npy')
            if os.path.exists(os.path.join(root, 'openclip', rec, filename)):
                continue
            img = preprocess(Image.open(file)).unsqueeze(0)
            img = img.cuda()
            feat = model.encode_image(img).cpu().numpy()
            np.save(os.path.join(root, 'openclip', rec, filename), feat)

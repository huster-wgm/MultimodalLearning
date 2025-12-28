
import os
import torch
import open_clip
import numpy as np
import pandas as pd



#root = "./dataset/2023-Thyroid"  #SYSU8H 
root = "./dataset/GY20240226"  #GY
model, _, preprocess = open_clip.create_model_and_transforms('RN50', pretrained='openai')

tokenizer = open_clip.get_tokenizer('ViT-B-32')

os.makedirs(os.path.join(root, 'tokenizer'), exist_ok=True)
df = pd.read_csv(os.path.join(root, 'clinical.csv'))
desps = []
for idx, rec in df.iterrows():
    print(f"Handling >> {rec['ID']}, {idx}/{df.shape[0]}")
    desp = f"gender {rec['Gender']}, age {rec['Age']}"
    desps.append(desp)

desps = tokenizer(desps)
features = model.encode_text(desps)
print(len(desps), len(features))
with torch.no_grad(), torch.cuda.amp.autocast():
    for idx in range(len(features)):
        feat = features[idx:idx+1,:].numpy()
        print(f"{idx}/{len(features)}, {feat.shape}")
        filename = f"{df.iloc[idx,0]}.npy"
        np.save(os.path.join(root, 'tokenizer', filename), feat)

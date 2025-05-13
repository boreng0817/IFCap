import json
import os
import clip
import numpy as np
from tqdm import tqdm
from PIL import Image
import pickle
import torch

@torch.no_grad()
def main(encoder, proprecess, dataset, clip_name):

    annotation_path = f"annotations/{dataset}/test_captions.json"
    outpath = f"annotations/{dataset}/test_captions_{clip_name}.pickle"
    rootpath = f"annotations/{dataset}/frames/"

    annotations = json.load(open(annotation_path, 'r'))
    
    results = []
    for video in tqdm(annotations):
        caption = annotations[video]
        image_paths = rootpath + video
        ims = [proprecess(Image.open(image_paths + '/' + im)) for im in os.listdir(image_paths)]
        ims = torch.stack(ims).to(device)
        embeddings = encoder.encode_image(ims).to('cpu')
        video_feature = torch.mean(embeddings, dim=0)
        results.append([video, video_feature, caption])


    with open(outpath, 'wb') as outfile:
        pickle.dump(results, outfile)

if __name__=='__main__':
    idx = 1
    dataset = ['msvd', 'msrvtt'][idx]

    device = 'cuda:0'
    clip_type = 'ViT-B/32'
    clip_name = clip_type.replace('/', '')
    encoder, proprecess = clip.load(clip_type, device)

    main(encoder, proprecess, dataset, clip_name)

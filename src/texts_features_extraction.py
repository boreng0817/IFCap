import os
import clip
import pickle
import torch
from tqdm import tqdm

@torch.no_grad()
def main(device: str, clip_type: str, inpath: str, outpath: str):

    device = device
    encoder, _ = clip.load(clip_type, device)

    with open(inpath, 'rb') as infile:
        captions_with_entities = pickle.load(infile) # [[[entity1, entity2, ...], caption], ...]

    for idx in tqdm(range(len(captions_with_entities))):
        caption = captions_with_entities[idx][1]
        tokens = clip.tokenize(caption, truncate = True).to(device)
        embeddings = encoder.encode_text(tokens).squeeze(dim = 0).to('cpu')
        captions_with_entities[idx].append(embeddings)
    
    with open(outpath, 'wb') as outfile:
        pickle.dump(captions_with_entities, outfile)
    
    return captions_with_entities

if __name__ == '__main__':

    device = 'cuda:0'
    clip_type = 'ViT-B/32' # change here for different clip backbone (ViT-B/32, RN50x4)
    clip_name = clip_type.replace('/', '')

    idx = 3 # change here! 0 -> coco training data, 1 -> flickr30k training data
    datasets = ['coco', 'flickr30k', 'msvd', 'msrvtt'][idx]
    inpath = f'./annotations/{datasets}/{datasets}_with_entities.pickle'
    outpath = f'./annotations/{datasets}/{datasets}_texts_features_{clip_name}.pickle'
    

    if os.path.exists(outpath):
        with open(outpath, 'rb') as infile:
            captions_with_features = pickle.load(infile)
    else:
        captions_with_features = main(device, clip_type, inpath, outpath)

    import random
    print(f'datasets for {inpath}')
    print(f'The length of datasets: {len(captions_with_features)}')
    idx = random.randint(0, len(captions_with_features) - 1)
    caption_with_features = captions_with_features[idx]
    detected_entities, caption, caption_features = caption_with_features
    print(detected_entities, caption, caption_features.size(), caption_features.dtype)

    encoder, _ = clip.load(clip_type, device)
    with torch.no_grad():
        embeddings = encoder.encode_text(clip.tokenize(caption, truncate = True).to(device)).squeeze(dim = 0).to('cpu')
    print(abs(embeddings - caption_features).mean())
    print(idx)

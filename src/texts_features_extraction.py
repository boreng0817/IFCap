import os
import clip
import pickle
import torch
import json
from tqdm import tqdm
@torch.no_grad()
def main(device: str, clip_type: str, inpath: str, outpath: str):

    device = device
    encoder, _ = clip.load(clip_type, device)
    feat_rt_entities = {}
    with open(inpath, 'r') as infile:
        rt_entities = json.load(infile) # [[[entity1, entity2, ...], caption], ...]

    batch_size = 256
    for idx, (key, val) in tqdm(enumerate(rt_entities.items())):
        # caption = rt_entities[1]
        tokens = clip.tokenize(val, truncate = True).to(device)
        embeddings = encoder.encode_text(tokens).squeeze(dim = 0).to('cpu')
        feat_rt_entities[key] = embeddings

    with open(outpath, 'wb') as outfile:
        pickle.dump(feat_rt_entities, outfile)

    return feat_rt_entities

if __name__ == '__main__':

    idx = 0 # change here! 0 -> coco training data, 1 -> flickr30k training data
    device = 'cuda:0'
    clip_type = 'ViT-B/32' # change here for different clip backbone (ViT-B/32, RN50x4)
    clip_name = clip_type.replace('/', '')

    inpath = [
    'annotations/coco/rt_add_noise_image_caps_train.json',
    'annotations/flickr30k/flickr30k_with_entities.pickle']
    outpath = [
    f'annotations/coco/coco_rt_add_noise_texts_features.pickle',
    f'annotations/flickr30k/flickr30k_texts_features_{clip_name}.pickle']

    if os.path.exists(outpath[idx]):
        with open(outpath[idx], 'rb') as infile:
            captions_with_features = pickle.load(infile)
    else:
        captions_with_features = main(device, clip_type, inpath[idx], outpath[idx])

    import random
    print(f'datasets for {inpath[idx]}')
    print(f'The length of datasets: {len(captions_with_features)}')
    caption_with_features = captions_with_features[random.randint(0, len(captions_with_features) - 1)]
    detected_entities, caption, caption_features = caption_with_features
    print(detected_entities, caption, caption_features.size(), caption_features.dtype)

    encoder, _ = clip.load(clip_type, device)
    with torch.no_grad():
        embeddings = encoder.encode_text(clip.tokenize(caption, truncate = True).to(device)).squeeze(dim = 0).to('cpu')
    print(abs(embeddings - caption_features).mean())


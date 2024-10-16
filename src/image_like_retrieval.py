import os
import pickle
import json
import argparse
import random
import clip
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from load_annotations import load_captions
from utils import noise_injection

# global variable
clip_model, preprocess = None, None

def get_captions_path(domain):
    datasets = {
            'coco' : './annotations/coco/train_captions.json',
            'flickr30k' : './annotations/flickr30k/train_captions.json',
            'nocaps' : './annotations/nocaps/nocaps_corpus.json',
            }
    return datasets[domain]

def get_image_path(domain):
    datasets = {
            'coco' : './annotations/coco/val2014/',
            'flickr30k' : './annotations/flickr30k/flickr30k-images/',
            'nocaps' : './annotations/nocaps/images/',
            }
    return datasets[domain]

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_caption_features(clip_feature_path, train_captions, args):
    if os.path.exists(clip_feature_path):
        with open(clip_feature_path, 'rb') as f:
            caption_features = pickle.load(f)
    else:
        features = []
        batch_size = 256
        with torch.no_grad():
            for i in tqdm(range(0, len(train_captions), batch_size)):
                batch_captions = train_captions[i: i + batch_size]
                clip_captions = clip.tokenize(batch_captions, truncate=True).to(args.device)
                clip_features = clip_model.encode_text(clip_captions)
                features.append(clip_features)

            caption_features = torch.cat(features).to('cpu')
        with open(clip_feature_path, 'wb') as f:
            pickle.dump(caption_features, f)

    return caption_features

def image_like_retrieval_train(train_captions, output_path, caption_features, args):

    retrieved_captions = {}

    noise_features = noise_injection(caption_features,
                                     variance=args.variance,
                                     device=args.device).to(torch.float16)

    for i in tqdm(range(noise_features.shape[0])):
        noise_feature = noise_features[i].unsqueeze(0)
        similarity = noise_feature @ caption_features.T
        similarity[0][i] = 0
        niber = []
        for _ in range(args.K):
            _, max_id = torch.max(similarity, dim=1)
            niber.append(max_id.item())
            similarity[0][max_id.item()] = 0
        retrieved_captions[train_captions[i]] = [train_captions[k] for k in niber]

    with open(output_path, 'w') as f:
        json.dump(retrieved_captions, f, indent=4)

def retrieve_caption_test(image_path, annotations, train_captions, output_path, caption_features, args):

    bs = 256
    image_ids = list(annotations.keys())
    image_features = []
    with torch.no_grad():
        for idx in tqdm(range(0, len(image_ids), bs)):
            image_input = [preprocess(Image.open(os.path.join(image_path, i)))
                           for i in image_ids[idx:idx + bs]]
            image_features.append(clip_model.encode_image(torch.tensor(np.stack(image_input)).to(args.device)))
        image_features = torch.concat(image_features)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        retrieved_captions = {}

        for i in tqdm(range(image_features.shape[0])):
            image_feature = image_features[i].unsqueeze(0).to(args.device)
            similarity = image_feature @ caption_features.T
            niber = []
            for _ in range(args.L):
                _, max_id = torch.max(similarity, dim=1)
                niber.append(max_id.item())
                similarity[0][max_id.item()] = 0

            retrieved_captions[image_ids[i]] = [train_captions[k] for k in niber]

        with open(output_path, 'w') as f:
            json.dump(retrieved_captions, f, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type = int, default = 30, help = 'Random seed')
    parser.add_argument('--L', type = int, default = 9, help = 'The number of retrieved captions for Image Like Retrieval')
    parser.add_argument('--K', type = int, default = 5, help = 'The number of retrieved captions for Entity Filtering')
    parser.add_argument('--variance', type = float, default = 0.04, help = 'Variance for noise injection')
    parser.add_argument('--domain_test', default = 'coco', help = 'Name of test dataset', choices=['coco', 'flickr30k', 'nocaps'])
    parser.add_argument('--domain_source', default = 'coco', help = 'Name of source dataset', choices=['coco', 'flickr30k'])
    parser.add_argument('--device', default = 'cuda:1', help = 'Cuda device')
    parser.add_argument('--variant', default = 'RN50x64', help = 'CLIP variant')
    parser.add_argument('--test_only', action = 'store_true', help = 'No ILR')

    args = parser.parse_args()

    global clip_model, preprocess

    set_seed(args.seed)
    clip_model, preprocess = clip.load(args.variant, device=args.device)

    captions_path = get_captions_path(args.domain_source)
    datasets = f'{args.domain_source}_captions'
    image_path = get_image_path(args.domain_test)

    train_output_path = f'./annotations/{args.domain_test}/{args.domain_test}_train_seed{args.seed}_var{args.variance}.json'
    test_output_path = f'./annotations/retrieved_sentences/caption_{args.domain_source}_image_{args.domain_test}_{args.L}.json'
    clip_feature_path = f'./annotations/{args.domain_source}/text_feature_clip{args.variant}.pickle'
    print('train_output_path', train_output_path)
    print('test_output_path', test_output_path)
    print('clip_feature_path', clip_feature_path)

    train_captions = load_captions(datasets, captions_path)
    caption_features = load_caption_features(clip_feature_path, train_captions, args).to(args.device)
    caption_features = caption_features / caption_features.norm(dim=-1, keepdim=True)

    with open(f"./annotations/{args.domain_test}/test_captions.json", 'r') as f:
        annotations = json.load(f)

    if not args.test_only and not os.path.exists(train_output_path):
        print('Perform image-like retrieval')
        image_like_retrieval_train(train_captions, train_output_path, caption_features, args)

    if not os.path.exists(test_output_path):
        print('Perform image-to-text retrieval')
        retrieve_caption_test(image_path, annotations, train_captions, test_output_path, caption_features, args)

if __name__=='__main__':
    main()

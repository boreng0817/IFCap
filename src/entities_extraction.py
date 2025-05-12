import os
import nltk
import pickle
from typing import List
from nltk.stem import WordNetLemmatizer
from load_annotations import load_captions
from tqdm import tqdm

def main(captions: List[str], path: str) -> None:
    # writing list file, i.e., [[[entity1, entity2,...], caption], ...] 

    lemmatizer = WordNetLemmatizer()
    new_captions = []
    for caption in tqdm(captions):
        detected_entities = []
        pos_tags = nltk.pos_tag(nltk.word_tokenize(caption)) # [('woman': 'NN'), ...]
        for entities_with_pos in pos_tags:
            if entities_with_pos[1] == 'NN' or entities_with_pos[1] == 'NNS':
                entity = lemmatizer.lemmatize(entities_with_pos[0].lower().strip())
                detected_entities.append(entity)
        detected_entities = list(set(detected_entities))
        new_captions.append([detected_entities, caption])
    
    with open(path, 'wb') as outfile:
        pickle.dump(new_captions, outfile)
    

if __name__ == '__main__':

    idx = 3 # only need to change here! 0 -> coco training data, 1 -> flickr30k training data
    datasets = ['coco', 'flickr30k', 'msvd', 'msrvtt'][idx]

    captions_path = f'./annotations/{datasets}/train_captions.json'
    out_path = f'./annotations/{datasets}/{datasets}_with_entities.pickle'
    datasets += '_captions'
    
    if os.path.exists(out_path[idx]):
        print('Read!')
        with open(out_path[idx], 'rb') as infile:
            captions_with_entities = pickle.load(infile)
        print(f'The length of datasets: {len(captions_with_entities)}')
        captions_with_entities = captions_with_entities[:20]
        for caption_with_entities in captions_with_entities:
            print(caption_with_entities)
        
    else:
        print('Writing... ...')
        captions = load_captions(datasets, captions_path)
        main(captions, out_path)

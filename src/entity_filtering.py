import json
import nltk
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer

def main(captions, path: str) -> None:
    # writing list file, i.e., [[[entity1, entity2,...], caption], ...]

    lemmatizer = WordNetLemmatizer()
    new_captions = {}
    for key in tqdm(captions.keys()):
        caps = captions[key]
        if not isinstance(caps, list):
            caps = [caps]
        detected_entities = {}
        for cap in caps:
            pos_tags = nltk.pos_tag(nltk.word_tokenize(cap)) # [('woman': 'NN'), ...]
            for entities_with_pos in pos_tags:
                if entities_with_pos[1] == 'NN' or entities_with_pos[1] == 'NNS':
                    entity = lemmatizer.lemmatize(entities_with_pos[0].lower().strip())
                    if entity not in detected_entities:
                        detected_entities[entity] = 0
                    detected_entities[entity] += 1
        words, freqs = zip(*detected_entities.items())
        li = list(zip(freqs, words))
        li.sort(reverse=True)
        new_captions[key] = li

    with open(path, 'w') as file:
        json.dump(new_captions, file)

if __name__ == '__main__':

    IDX = 4
    inputs = [
            "caption_coco_image_coco_9.json",
            "caption_flickr30k_image_flickr30k_7.json",
            "caption_coco_image_nocaps_7.json",
            "caption_msvd_image_msvd_7.json",
            "caption_msrvtt_image_msrvtt_7.json",
            ]

    captions_path = f'annotations/retrieved_sentences/{inputs[IDX]}'
    out_path = f'annotations/retrieved_entity/{inputs[IDX]}'

    with open(captions_path, 'r') as file:
        captions = json.load(file)
    main(captions, out_path)

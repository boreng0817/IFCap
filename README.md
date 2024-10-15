# IFCap
IFCap: Image-like Retrieval and Frequency-based Entity Filtering for Zero-shot Captioning, EMNLP2024

---
### Conda environment
```bash
conda create -n ifcap python=3.9
conda activate ifcap
pip install -r requirements.txt
```

### Data preparation
Download annotations, evaluation tools, and best checkpoints.
```
bash scripts/download.sh
```

For COCO,
```
# image-like retrieval
python src/image_like_retrieval.py

# entity filtering
python src/entity_filtering.py # with IDX=0
```

For Flickr30k,
```
# image-like retrieval
python src/image_like_retrieval.py --domain flickr30k --L 7

# entity filtering
python src/entity_filtering.py # with IDX=1
```

### Training
For COCO,
```
# bash scripts/train_coco.sh CUDA_DEVICE TEST_NAME RT_PATH
bash scripts/train_coco.sh 0 coco annotations/coco/coco_train_seed30_var0.04.json
```

For Flickr30k
```
# bash scripts/train_flickr30k.sh CUDA_DEVICE TEST_NAME RT_PATH
bash scripts/train_flickr30k.sh 0 flickr annotations/flickr30k/flickr30k_train_seed30_var0.04.json
```

### Inference
For COCO,
```
bash scripts/eval_coco.sh train_coco 0 \
	'--entity_filtering --ef_entity_path image_coco_caption_coco_9.json --rt_sentence_path caption_coco_test_9.json --K 5' \
	coco-indomain \
	4
```

For Flickr30k,
```
bash scripts/eval_flickr30k.sh train_flickr30k 0 \
	'--entity_filtering --ef_entity_path image_flickr30k_caption_flickr30k_7.json --rt_sentence_path caption_flickr30k_test_7.json --K 3' \
	flickr-indomain \
	14
```


## Citation
If you use this code for your research, please cite:
```
@article{lee2024ifcap,
  title={IFCap: Image-like Retrieval and Frequency-based Entity Filtering for Zero-shot Captioning},
  author={Lee, Soeun and Kim, Si-Woo and Kim, Taewhan and Kim, Dong-Jin},
  journal={arXiv preprint arXiv:2409.18046},
  year={2024}
}
```

## Acknowledgments

This repository is based on [ViECap](https://github.com/FeiElysia/ViECap), [Knight](https://github.com/junyangwang0410/Knight) and [pycocotools](https://github.com/sks3i/pycocoevalcap) repositories. Thanks for sharing the source codes!

***

# IFCap
IFCap: Image-like Retrieval and Frequency-based Entity Filtering for Zero-shot Captioning, EMNLP2024

![poster_emnlp](https://github.com/user-attachments/assets/05bd3d10-627e-4e5f-84ed-0df12e172784)

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
# place your coco images in annotations/coco/val2014

# image-like retrieval
python src/image_like_retrieval.py

# entity filtering
python src/entity_filtering.py # with IDX=0
```

For Flickr30k,
```
# place your flickr30k images in annotations/flickr30k/flickr30k-images

# image-like retrieval
python src/image_like_retrieval.py --domain_source flickr30k --domain_test flickr30k --L 7

# entity filtering
python src/entity_filtering.py # with IDX=1
```

For NoCaps,
```
# download images of NoCaps validation
# In my case, it took about 2 hours.
cd annotations/nocaps/
python download.py 

# image-to-text retrieval
python src/image_like_retrieval.py --test_only --domain_test nocaps --L 7
```

For MSVD and MSRVTT
```
cd annotations/{msvd, msrvtt}
python build_dataset/build_dataset.py

# Place corresponding videos in annotations/{msvd, msrvtt}/videos
cd ../..
python src/sample_frame.py # idx 0 for MSVD, 1 for MSRVTT

# For extracting feature
python src/entities_extraction.py # idx 2 for MSVD, 3 for MSRVTT
python src/texts_features_extraction.py # idx 2 for MSVD, 3 for MSRVTT
python src/video_features_extraction.py # idx 0 for MSVD, 1 for MSRVTT


# image-to-text retrieval
python src/image_like_retrieval.py --domain_source msvd --domain_test msvd --video --L 7
python src/image_like_retrieval.py --domain_source msrvtt --domain_test msrvtt --video --L 7

# entity filtering
python src/entity_filtering.py # with idx 3 for MSVD, 4 for MSRVTT
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

For MSVD
```
# bash scripts/train_msvd.sh CUDA_DEVICE TEST_NAME RT_PATH
bash scripts/train_msvd.sh 0 msvd annotations/msvd/msvd_train_seed30_var0.04.json
```

For MSRVTT
```
# bash scripts/train_msrvtt.sh CUDA_DEVICE TEST_NAME RT_PATH
bash scripts/train_msrvtt.sh 0 msrvtt annotations/msrvtt/msrvtt_train_seed30_var0.04.json
```

### Inference
[COCO]
```
bash scripts/eval_coco.sh train_coco 0 \
	'--entity_filtering --retrieved_info caption_coco_image_coco_9.json --K 5' \
	coco-indomain \
	4
```

[Flickr30k]
```
bash scripts/eval_flickr30k.sh train_flickr30k 0 \
	'--entity_filtering --retrieved_info caption_flickr30k_image_flickr30k_7.json --K 3' \
	flickr-indomain \
	14
```

[NoCaps]
```
bash scripts/eval_nocaps.sh train_coco 0 \
	'--retrieved_info caption_coco_image_nocaps_7.json' \
	coco-indomain \
	5
```

[COCO -> Flickr30k]
```
bash scripts/eval_flickr30k.sh train_coco 0 \
	'--entity_filtering --retrieved_info caption_flickr30k_image_flickr30k_7.json --K 3' \
	coco-indomain \
	5
```

[Flickr30k -> COCO]
```
bash scripts/eval_coco.sh train_flickr30k 0 \
	'--entity_filtering --retrieved_info caption_coco_image_coco_9.json --K 4' \
	flickr-indomain \
	14
```

[MSVD]
```
bash scripts/eval_msvd.sh train_msvd 0 '' msvd 9
```

[MSRVTT]
```
bash scripts/eval_msrvtt.sh train_msrvtt 0 '' msrvtt 9
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

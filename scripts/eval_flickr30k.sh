SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER/..

EXP_NAME=$1
DEVICE=$2
OTHER_ARGS=$3
CKPT_NAME=$4
ckpt_num=$5
WEIGHT_PATH=checkpoints/$EXP_NAME/"${CKPT_NAME}-00${ckpt_num}.pt"
FLICKR_OUT_PATH=inference_result/${CKPT_NAME}/"${CKPT_NAME}-00${ckpt_num}"

TIME_START=$(date "+%Y-%m-%d-%H-%M-%S")
LOG_FOLDER=inference_result/${CKPT_NAME}/"${CKPT_NAME}-00${ckpt_num}"
mkdir -p $LOG_FOLDER

FLICKR_LOG_FILE="$LOG_FOLDER/FLICKR_${TIME_START}.log"

$CPU_SET python src/validation.py \
--device cuda:$DEVICE \
--clip_model ViT-B/32 \
--language_model gpt2 \
--continuous_prompt_length 10 \
--clip_project_length 10 \
--top_k 3 \
--threshold 0.3 \
--using_image_features \
--name_of_datasets flickr30k \
--path_of_val_datasets ./annotations/flickr30k/test_captions.json \
--name_of_entities_text vinvl_vgoi_entities \
--image_folder ./annotations/flickr30k/flickr30k-images/ \
--prompt_ensemble \
--weight_path=$WEIGHT_PATH \
--out_path=$FLICKR_OUT_PATH \
--using_hard_prompt \
--soft_prompt_first \
--using_greedy_search \
--k 5 \
$OTHER_ARGS \
|& tee -a  ${FLICKR_LOG_FILE}

echo "==========================FLICKR EVAL================================"
python evaluation/cocoeval.py --result_file_path $FLICKR_OUT_PATH/flickr30k*.json |& tee -a  ${FLICKR_LOG_FILE}


SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
EXP_NAME=$1
DEVICE=$2
OTHER_ARGS=$3
CKPT_NAME=$4
ckpt_num=$5

cd $SHELL_FOLDER/..

WEIGHT_PATH=checkpoints/$EXP_NAME/"${CKPT_NAME}-00${ckpt_num}.pt"
COCO_OUT_PATH=inference_result/${CKPT_NAME}/"${CKPT_NAME}-00${ckpt_num}"

TIME_START=$(date "+%Y-%m-%d-%H-%M-%S")
LOG_FOLDER=inference_result/${CKPT_NAME}/"${CKPT_NAME}-00${ckpt_num}"
mkdir -p $LOG_FOLDER

COCO_LOG_FILE="$LOG_FOLDER/COCO_${TIME_START}.log"

python src/validation.py \
--device cuda:$DEVICE \
--clip_model ViT-B/32 \
--language_model gpt2 \
--continuous_prompt_length 10 \
--clip_project_length 10 \
--top_k 3 \
--using_image_features \
--threshold 0.4 \
--name_of_datasets coco \
--path_of_val_datasets annotations/coco/test_captions.json \
--name_of_entities_text coco_entities \
--image_folder /data/dataset/coco_2014/val2014/ \
--prompt_ensemble \
--weight_path=$WEIGHT_PATH \
--out_path=$COCO_OUT_PATH \
--using_hard_prompt \
--soft_prompt_first \
--k 5 \
--domain coco \
$OTHER_ARGS \

echo "==========================COCO EVAL================================"
python evaluation/cocoeval.py --result_file_path $COCO_OUT_PATH/coco*.json |& tee -a  ${COCO_LOG_FILE}

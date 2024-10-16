SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)

cd $SHELL_FOLDER/..

EXP_NAME=$1
DEVICE=$2
OTHER_ARGS=$3
CKPT_NAME=$4
ckpt_num=$5
WEIGHT_PATH=checkpoints/$EXP_NAME/"${CKPT_NAME}-00${ckpt_num}.pt"
NOCAPS_OUT_PATH=inference_result/${CKPT_NAME}/"${CKPT_NAME}-00${ckpt_num}"

TIME_START=$(date "+%Y-%m-%d-%H-%M-%S")
LOG_FOLDER=inference_result/${CKPT_NAME}/"${CKPT_NAME}-00${ckpt_num}"
mkdir -p $LOG_FOLDER

NOCAPS_LOG_FILE="$LOG_FOLDER/NOCAPS_${TIME_START}"

python src/validation.py \
--device cuda:$DEVICE \
--clip_model ViT-B/32 \
--language_model gpt2 \
--continuous_prompt_length 10 \
--clip_project_length 10 \
--top_k 3 \
--threshold 0.2 \
--using_image_features \
--name_of_datasets nocaps \
--path_of_val_datasets ./annotations/nocaps/nocaps_corpus.json \
--name_of_entities_text vinvl_vgoi_entities \
--image_folder ./annotations/nocaps/ \
--prompt_ensemble \
--weight_path=$WEIGHT_PATH \
--out_path=$NOCAPS_OUT_PATH \
--using_hard_prompt \
--soft_prompt_first \
$OTHER_ARGS \
|& tee -a  "${NOCAPS_LOG_FILE}.log"

echo "==========================NOCAPS IN-DOAMIN================================"
python evaluation/cocoeval.py --result_file_path  ${NOCAPS_OUT_PATH}/indomain*.json |& tee -a  ${NOCAPS_LOG_FILE}_in.log
echo "==========================NOCAPS NEAR-DOAMIN================================"
python evaluation/cocoeval.py --result_file_path  ${NOCAPS_OUT_PATH}/neardomain*.json |& tee -a  ${NOCAPS_LOG_FILE}_near.log
echo "==========================NOCAPS OUT-DOAMIN================================"
python evaluation/cocoeval.py --result_file_path  ${NOCAPS_OUT_PATH}/outdomain*.json |& tee -a  ${NOCAPS_LOG_FILE}_out.log
echo "==========================NOCAPS ALL-DOAMIN================================"
python evaluation/cocoeval.py --result_file_path  ${NOCAPS_OUT_PATH}/overall*.json |& tee -a  ${NOCAPS_LOG_FILE}_all.log

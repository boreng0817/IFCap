SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER/..

DEVICE=$1
EXP_NAME=`echo "$(basename $0)" | cut -d'.' -f1` 
LOG_FILE=logs/$EXP_NAME
PREFIX=$2
RT_PATH=$3

TIME_START=$(date "+%Y-%m-%d-%H-%M-%S")
LOG_FOLDER=logs/${EXP_NAME}
LOG_FILE="$LOG_FOLDER/${TIME_START}.log"
OTHER_ARGS=$4
mkdir -p $LOG_FOLDER

echo "=========================================================="
echo "RUNNING EXPERIMENTS: $EXP_NAME, saving in checkpoints/$EXP_NAME"
echo "=========================================================="

python src/main.py \
--bs 80 \
--lr 0.00002 \
--epochs 10 \
--device cuda:$DEVICE \
--random_mask \
--prob_of_random_mask 0.4 \
--clip_model ViT-B/32 \
--language_model gpt2 \
--using_hard_prompt \
--soft_prompt_first \
--prefix $PREFIX \
--path_of_datasets ./annotations/msvd/msvd_texts_features_ViT-B32.pickle \
--out_dir checkpoints/$EXP_NAME \
--use_amp \
--num_workers 4 \
--k 5 \
--rt_path $RT_PATH \
$OTHER_ARGS \

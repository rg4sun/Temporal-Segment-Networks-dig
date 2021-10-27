#!/bin/bash
# train tsn with one GPU
# shellcheck disable=SC1068

#ROOT_PATH=/home/shh_ucf
ROOT_PATH=/data2/hdu/shh
REPO_PATH=${ROOT_PATH}/tsn-mindspore-gpu
DATA_DIR=${ROOT_PATH}/data/data_extracted/ucf101/tvl1

#GPU_ID=1 # on huawei
# GPU_ID=0
WORKERS=4 # on huawei
# WORKERS=1

MODALITY=$1
EPOCH=$2
PRE_CKPT_NAME=$3
GPU_ID=$4

CKPT_PRE_PATH=${ROOT_PATH}/ckpt-station/${PRE_CKPT_NAME}
CKPT_SAVE_DIR=$ROOT_PATH/records-${MODALITY}/train-records/epoch-${EPOCH}/${MODALITY}_ckpt

if [ $# != 4 ]; then
  echo "Usage: bash run_train.sh [MODALITY] [EPOCH] [PRE_CKPT_NAME] [GPU_ID]"
  exit 1
fi


cd $ROOT_PATH

if [ ! -d records-${MODALITY} ];then
        echo 1st time run, creating records-${MODALITY} dir
        mkdir records-${MODALITY}
fi

cd records-${MODALITY}

if [ ! -d train-records ];then
        mkdir train-records
        echo train-records dir created!
else
        echo train-records exist!Go on...
fi

cd train-records

mkdir epoch-${EPOCH} && cd epoch-${EPOCH}

echo Start ${MODALITY} training, log saved as ${MODALITY}_train_${EPOCH}.log in `pwd`/${MODALITY}_train_${EPOCH}.log
python ${REPO_PATH}/mindoptimizer_gpu.py \
  --modality=$MODALITY \
  --epochs=$EPOCH \
  --device_id=$GPU_ID \
  --workers=$WORKERS \
  --ckpt_save_dir=$CKPT_SAVE_DIR \
  --pretrained_path=$CKPT_PRE_PATH \
  --platform='GPU' \
  --dataset_path=$DATA_DIR > ${MODALITY}_train_${EPOCH}.log &

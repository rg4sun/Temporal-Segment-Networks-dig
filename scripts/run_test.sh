#!/bin/bash
# test tsn with one GPU
# shellcheck disable=SC1068

#ROOT_PATH=/home/shh_ucf
ROOT_PATH=/data2/hdu/shh
REPO_PATH=${ROOT_PATH}/tsn-mindspore-gpu
DATA_DIR=${ROOT_PATH}/data/data_extracted/ucf101/tvl1
TEST_LIST=${ROOT_PATH}/data/data_extracted/ucf101/ucf101_val_split_1_rawframes.txt

#GPU_ID=1 # on huawei
# GPU_ID=0
WORKERS=1 # on huawei
# WORKERS=1

MODALITY=$1
CKPT_NAME=$2
GPU_ID=$3

SCORE_SAVE_PATH=${ROOT_PATH}/scores_${MODALITY}

CKPT_PATH=${ROOT_PATH}/ckpt-station/${CKPT_NAME}


if [ $# != 3 ]; then
  echo "Usage: bash run_test.sh [MODALITY] [CKPT_NAME] [GPU_ID]"
  exit 1
fi

# 这个可以不写也没事，py脚本里也加了路径检查
if [ ! -d ${SCORE_SAVE_PATH} ];then
  echo ${SCORE_SAVE_PATH} doesn\'t  exist, now creatind...
  mkdir ${SCORE_SAVE_PATH}
fi

cd ${SCORE_SAVE_PATH}

echo Now testing TSN in ${MODALITY} modality, log saved as ${MODALITY}_test.log in `pwd`/${MODALITY}_test.log
python ${REPO_PATH}/test_tsn.py \
  --modality=${MODALITY} \
  --weights=${CKPT_PATH} \
  --device_id=${GPU_ID} \
  --workers=${WORKERS} \
  --dataset_path=${DATA_DIR} \
  --test_list=${TEST_LIST} \
  --save_scores=${SCORE_SAVE_PATH} > ${MODALITY}_test.log &
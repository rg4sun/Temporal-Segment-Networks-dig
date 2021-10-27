#!/bin/bash
# distributed train tsn in flow mode

ROOT_PATH=/data2/hdu/shh
REPO_PATH=$ROOT_PATH/tsn-mindspore-gpu
DATA_DIR=/data2/hdu/shh/data/data_extracted/ucf101/tvl1
CKPT_PRE_PATH=/data2/hdu/shh/ckpt-records/RGB_ckpt_distributed-1-23/best.ckpt # 200+20+[100]

MODALITY=$1
EPOCH=$2

CKPT_SAVE_DIR=$ROOT_PATH/${MODALITY}_ckpt_distributed
# 加{}限定变量范围，不加的话他会把 MODALITY_ckpt_distributed 一整个视作变量

if [ $# != 2 ]; then
  echo "Usage: bash run_distribute_train_gpu.sh  [MODALITY] [EPOCH]"
  exit 1
fi

cd $ROOT_PATH

if [ ! -d distrubued-train-$MODALITY ];then
  echo 1st time run, creating distributed-train-$MODALITY dirs
  mkdir distributed-train-$MODALITY
fi

cd distributed-train-$MODALITY
mkdir distributed-epoch-$EPOCH && cd distributed-epoch-$EPOCH

echo Start distributed $MODALITY training, log saved as ${MODALITY}_train_$EPOCH.log in `pwd`/${MODALITY}_train_${EPOCH}.log

CUDA_VISIBLE_DEVICES=1,2,3 mpirun --allow-run-as-root -n 3 \
  --output-filename epoch_log-$EPOCH \
  --merge-stderr-to-stdout \
  python $REPO_PATH/mindoptimizer_distibuted.py --run_distribute=True \
  --epochs=$EPOCH \
  --modality=$MODALITY \
  --ckpt_save_dir=$CKPT_SAVE_DIR \
  --pretrained_path=$CKPT_PRE_PATH \
  --platform='GPU' \
  --dataset_path=$DATA_DIR > ${MODALITY}_train_${EPOCH}.log 2>&1 &


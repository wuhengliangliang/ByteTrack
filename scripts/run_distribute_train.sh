#!/bin/bash
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===========================================================================
if [[ $# -lt 4 || $# -gt 5 ]];then
    echo "Usage1: bash run_distribute_train.sh [DATASET_PATH] [RANK_TABLE_FILE] [BACKBONE]  for first data aug epochs"
    echo "Usage2: bash run_distribute_train.sh [DATASET_PATH] [RANK_TABLE_FILE] [BACKBONE] [RESUME_CKPT] for last no data aug epochs"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

DATASET_PATH=$(get_real_path $1)
RANK_TABLE_FILE=$(get_real_path $2)
BACKBONE=$3
CKPT_PATH=$(get_real_path $4)

echo $DATASET_PATH
echo $RANK_TABLE_FILE
echo $BACKBONE

if [ ! -d $DATASET_PATH ]
then
    echo "error: DATASET_PATH=$DATASET_PATH is not a directory"
exit 1
fi
if [ ! -f $RANK_TABLE_FILE ]
then
    echo "error: RANK_TABLE_FILE=$RANK_TABLE_FILE is not a file"
exit 1
fi

export DEVICE_NUM=8
export RANK_SIZE=8
export RANK_TABLE_FILE=$RANK_TABLE_FILE
export HCCL_CONNECT_TIMEOUT=1200
cpus=`cat /proc/cpuinfo| grep "processor"| wc -l`
avg=`expr $cpus \/ $RANK_SIZE`
gap=`expr $avg \- 1`
if [ $# == 4 ]
then
  echo "Start to launch first data augment epochs..."
  for((i=0; i<${DEVICE_NUM}; i++))
  do
      start=`expr $i \* $avg`
      end=`expr $start \+ $gap`
      cmdopt=$start"-"$end
      export DEVICE_ID=$i
      export RANK_ID=$i
      rm -rf ./train_parallel$i
      mkdir ./train_parallel$i
      cp ../*.py ./train_parallel$i
      cp ../*.yaml ./train_parallel$i
      cp -r ../yolox ./train_parallel$i
      cp -r ../model_utils ./train_parallel$i
      cd ./train_parallel$i || exit
      echo "start training for rank $RANK_ID, device $DEVICE_ID"
      env > env.log
      taskset -c $cmdopt python train.py \
          --data_path=$DATASET_PATH \
          --backbone=$BACKBONE \
          --load_path=$CKPT_PATH \
          --data_aug=True \
          --is_distributed=1 \
          --lr=0.011 \
          --max_epoch=70 \
          --warmup_epochs=5 \
          --no_aug_epochs=10  \
          --min_lr_ratio=0.001 \
          --eval_interval=10 \
          --lr_scheduler=yolox_warm_cos_lr  > log.txt 2>&1 &
      cd ..
  done
fi
if [ $# == 5 ]
then
  echo "Start to launch last no data augment epochs..."
  for((i=0; i<${DEVICE_NUM}; i++))
  do
      start=`expr $i \* $avg`
      end=`expr $start \+ $gap`
      cmdopt=$start"-"$end
      export DEVICE_ID=$i
      export RANK_ID=$i
      rm -rf ./train_parallel$i
      mkdir ./train_parallel$i
      cp ../*.py ./train_parallel$i
      cp ../*.yaml ./train_parallel$i
      cp -r ../yolox ./train_parallel$i
      cp -r ../model_utils ./train_parallel$i
      cd ./train_parallel$i || exit
      echo "start training for rank $RANK_ID, device $DEVICE_ID"
      env > env.log
      taskset -c $cmdopt python train.py \
          --data_dir=$DATASET_PATH \
          --yolox_no_aug_ckpt=$RESUME_CKPT \
          --backbone=$BACKBONE \
          --data_aug=False \
          --is_distributed=1 \
          --lr=0.011 \
          --max_epoch=285 \
          --warmup_epochs=5 \
          --no_aug_epochs=15  \
          --min_lr_ratio=0.001 \
          --eval_interval=1 \
          --lr_scheduler=yolox_warm_cos_lr  > log.txt 2>&1 &
      cd ..
  done
fi

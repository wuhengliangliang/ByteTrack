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
# ============================================================================

if [[ $# -lt 3 || $# -gt 4 ]];then
    echo "Usage1: bash run_standalone_train.sh  [DATASET_PATH] [BACKBONE]  for first data aug epochs"
    echo "Usage2: bash run_standalone_train.sh  [DATASET_PATH] [BACKBONE] [LATEST_CKPT] for last no data aug epochs"
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
BACKBONE=$2
CKPT_PATH=$(get_real_path $3)
echo $DATASET_PATH
echo $BACKBONE

if [ ! -d $DATASET_PATH ]
then
    echo "error: DATASET_PATH=$DATASET_PATH is not a directory"
exit 1
fi

export DEVICE_NUM=1
export DEVICE_ID=6
export RANK_ID=0
export RANK_SIZE=1

if [ -d "train" ];
then
    rm -rf ./train
fi

mkdir ./train
cp ../*.py ./train
cp ../*.yaml ./train
cp -r ../yolox ./train
cp -r ../model_utils ./train
cd ./train || exit
echo "start training for device $DEVICE_ID"
env > env.log

if [ $# == 3 ]
then
  echo "Start to launch first data augment epochs..."
  python train.py \
        --data_path=$DATASET_PATH \
        --data_aug=True \
        --is_distributed=0 \
        --eval_interval=10 \
        --load_path=$CKPT_PATH
        --backbone=$BACKBONE > log.txt 2>&1 &
fi

if [ $# == 4 ]
then
  echo "Start to launch last no data augment epochs..."
  CKPT_FILE=$(get_real_path $3)
  echo $CKPT_FILE
  python train.py \
      --data_path=$DATASET_PATH \
      --data_aug=False \
      --is_distributed=0 \
      --eval_interval=1 \
      --backbone=$BACKBONE \
      --yolox_no_aug_ckpt=$CKPT_FILE > log.txt 2>&1 &
fi

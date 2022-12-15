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
# =======================================================================================
"""
for evaluate
"""
import os
import datetime
from tqdm import tqdm
import glob
from pathlib import Path
from collections import defaultdict
from collections import OrderedDict
import motmetrics as motm
from model_utils.config import config
from mindspore.common.tensor import Tensor
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import context

from yolox.utils.logger import get_logger
from yolox.utils.util import DetectionEngine
from yolox.model.yolox import DetectionBlock
from yolox.data.mosaicdetection import create_eval_dataloader
from yolox.utils.initializer import default_recurisive_init
from yolox.tracker.byte_tracker import ByteTracker


def run_test():
    """The function of eval"""
    config.data_root = os.path.join(config.data_dir)
    result_path = config.output_path
    img_path = os.path.join(result_path, 'img_data')
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    # logger
    config.outputs_dir = os.path.join(
        config.log_path, datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S')
    )
    rank_id = int(os.getenv('RANK_ID', '0'))
    config.logger = get_logger(config.outputs_dir, rank_id)

    # context.reset_auto_parallel_context()
    # parallel_mode = ParallelMode.STAND_ALONE
    # context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=1)
    # # ------------------network create----------------------------------------------------------------------------
    # config.logger.info('Begin Creating Network....')
    # if config.backbone == "yolox_darknet53":
    #     backbone = "yolofpn"
    # else:
    #     backbone = "yolopafpn"
    # network = DetectionBlock(config, backbone=backbone)  # default yolo-darknet53
    # default_recurisive_init(network)
    # config.logger.info(config.val_ckpt)
    # if os.path.isfile(config.val_ckpt):
    #     param_dict = load_checkpoint(config.val_ckpt)
    #     load_param_into_net(network, param_dict)
    #     config.logger.info('load model %s success', config.val_ckpt)
    # else:
    #     config.logger.info('%s doesn''t exist or is not a pre-trained file', config.val_ckpt)
    #     raise FileNotFoundError('{} not exist or not a pre-trained file'.format(config.val_ckpt))
    ds = create_eval_dataloader(config, config.data_root)
    # ds = create_yolox_dataset(data_root, anno_file, is_training=False, batch_size=config.per_batch_size, device_num=1,
    #                           rank=rank_id)
    data_size = ds.get_dataset_size()
    config.logger.info(
        'Finish loading the dataset, totally %s images to eval, iters %s' % (data_size * config.per_batch_size, \
                                                                                 data_size))
    # network.set_train(False)
    # # init detection engine
    # tracker = ByteTracker(config)
    # detection = DetectionEngine(config)
    # video_names = defaultdict()
    # results = []
    # results_folder = './track_results'
    config.logger.info('Start Preprocess....')
    for iterator, data in enumerate(
            tqdm(ds.create_dict_iterator(num_epochs=1, output_numpy=True), total=data_size,
                 colour="GREEN")):
        image = data['image']
        # print(image.shape)
        img_info = data['img_info'][0]
        img_id = data['img_id']

        frame_id = int(img_info[2].item())
        video_id = int(img_info[3].item())
        img_file_name = img_info[4]
        img_shape = [[int(img_info[0])], [int(img_info[1])]]
        video_name = img_file_name.decode().split('/')[0]
        # print(img_info,"--",img_id[0][0],"--",frame_id,"--", img_file_name)
        # print('+++++',str(img_file_name).split('/')[0].split("'")[1],str(img_file_name).split('/')[1])
        file_name = str(img_file_name).split('/')[2].split('.')[0]
        file_name_format = "{}_{}_{}.bin".format(str(img_file_name).split('/')[0].split("'")[1],
                                                 str(img_file_name).split('/')[1],file_name)
        img_file_path = os.path.join(img_path, file_name_format)
        image.tofile(img_file_path)
        # assert 1==2
        # print(video_name)
        # file_name = "{}_{}_{}.bin".format(str(img_id[0]), str(img_info[0]), str(img_info[1]))
        # img_file_path = os.path.join(img_path, file_name)
        # image_data.tofile(img_file_path)

    config.logger.info('Completed')



if __name__ == '__main__':
    run_test()

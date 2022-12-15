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
import numpy as np
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
# from yolox.model.yolox import DetectionBlock
from yolox.data.mosaicdetection import create_eval_dataloader
from yolox.utils.initializer import default_recurisive_init
from yolox.tracker.byte_tracker import ByteTracker


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    config.logger.info('save results to {}'.format(filename))

def compare_dataframes(gts, ts):
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:
            config.logger.info('Comparing {}...'.format(k))
            accs.append(motm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
            names.append(k)
        else:
            config.logger.warning('No ground truth for {}, skipping.'.format(k))

    return accs, names

def run_test():
    """The function of eval"""
    config.data_root = os.path.join(config.data_dir)

    # devid = int(os.getenv('DEVICE_ID', '0'))
    # context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, save_graphs=False, device_id=devid)

    # logger
    config.outputs_dir = os.path.join(
        config.log_path, datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S')
    )
    rank_id = int(os.getenv('RANK_ID', '0'))
    config.logger = get_logger(config.outputs_dir, rank_id)
    #
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
    # init detection engine
    tracker = ByteTracker(config)
    detection = DetectionEngine(config)
    video_names = defaultdict()
    results = []
    results_folder = './track_results'
    result_path = config.result_path
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)
    config.logger.info('Start inference...')
    for iterator, data in enumerate(
            tqdm(ds.create_dict_iterator(num_epochs=1, output_numpy=True), total=data_size,
                 colour="GREEN")):
        image = data['image']
        img_info = data['img_info'][0]
        img_id = data['img_id']

        frame_id = int(img_info[2].item())
        video_id = int(img_info[3].item())
        img_file_name = img_info[4]
        img_shape = [[int(img_info[0])], [int(img_info[1])]]
        # print(img_info,"--",img_id,"--",frame_id,"--", img_file_name)
        # print(img_shape)
        file_name = str(img_file_name).split('/')[2].split('.')[0]
        video_name = img_file_name.decode().split('/')[0]
        # print(video_name, "   ", video_id)
        if video_name == 'MOT17-05-FRCNN' or video_name == 'MOT17-06-FRCNN':
            config.track_buffer = 14
            print("track buffer update...", config.track_buffer)
        elif video_name == 'MOT17-13-FRCNN' or video_name == 'MOT17-14-FRCNN':
            config.track_buffer = 25
        else:
            config.track_buffer = 30

        if video_name == 'MOT17-01-FRCNN':
            config.track_thresh = 0.65
        elif video_name == 'MOT17-06-FRCNN':
            config.track_thresh = 0.65
        elif video_name == 'MOT17-12-FRCNN':
            config.track_thresh = 0.7
        elif video_name == 'MOT17-14-FRCNN':
            config.track_thresh = 0.67
        elif video_name in ['MOT20-06', 'MOT20-08']:
            config.track_thresh = 0.3
        else:
            config.track_thresh = 0.6

        if video_name not in video_names:
            video_names[video_id] = video_name
        if frame_id == 1:
            tracker = ByteTracker(config)
            if len(results) != 0:
                result_filename = os.path.join(results_folder, '{}.txt'.format(video_names[video_id - 1]))
                write_results(result_filename, results)
                results = []
        # outputs = network(Tensor(image))
        file_name = str(img_file_name).split('/')[2].split('.')[0]
        file_name_format = "{}_{}_{}_0.bin".format(str(img_file_name).split('/')[0].split("'")[1],
                                                 str(img_file_name).split('/')[1], file_name)
        file_name_out = os.path.join(result_path, file_name_format)
        outputs = np.fromfile(file_name_out, dtype=np.float32).reshape((1, 23625, 6))

        prediction = outputs
        detect_outputs = detection.detection(prediction, img_shape, img_id[0])

        # run tracking
        if detect_outputs[0] is not None:
            online_targets = tracker.update(detect_outputs[0], img_info, config.input_size)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > config.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
            # save results
            results.append((int(frame_id), online_tlwhs, online_ids, online_scores))

        if iterator == data_size - 1:
            result_filename = os.path.join(results_folder, '{}.txt'.format(video_names[video_id]))
            write_results(result_filename, results)
        # print(detect_outputs, results)
    config.logger.info('Calculating mAP...')
    result_file_path = detection.evaluate_prediction()
    config.logger.info('result file path: %s', result_file_path)
    eval_result, _, _ = detection.get_eval_result()
    eval_print_str = '\n=============coco eval result=========\n' + eval_result
    config.logger.info(eval_print_str)

    #evaluate MOTA
    motm.lap.default_solver = 'lap'
    if config.val_ann == 'val_half.json':
        gt_type = '_val_half'
    else:
        gt_type = ''
    config.logger.info('GT Type: %s', gt_type)
    if config.mot20:
        gtfiles = glob.glob(os.path.join(config.data_root+'/train', '*/gt/gt{}.txt'.format(gt_type)))
    else:
        gtfiles = glob.glob(os.path.join(config.data_root+'/train', '*/gt/gt{}.txt'.format(gt_type)))
    config.logger.info('GT Files: %s', gtfiles)
    tsfiles = [f for f in glob.glob(os.path.join(results_folder, '*.txt')) if
               not os.path.basename(f).startswith('eval')]
    config.logger.info('Found {} groundtruths and {} test files.'.format(len(gtfiles), len(tsfiles)))
    config.logger.info('Available LAP solvers {}'.format(motm.lap.available_solvers))
    config.logger.info('Default LAP solver \'{}\''.format(motm.lap.default_solver))
    config.logger.info('Loading files.')

    gt = OrderedDict([(Path(f).parts[-3], motm.io.loadtxt(f, fmt='mot15-2D', min_confidence=1)) for f in gtfiles])
    ts = OrderedDict(
        [(os.path.splitext(Path(f).parts[-1])[0], motm.io.loadtxt(f, fmt='mot15-2D', min_confidence=-1)) for f in
         tsfiles])

    mh = motm.metrics.create()
    accs, names = compare_dataframes(gt, ts)

    config.logger.info('Running metrics')
    metrics = ['recall', 'precision', 'num_unique_objects', 'mostly_tracked',
               'partially_tracked', 'mostly_lost', 'num_false_positives', 'num_misses',
               'num_switches', 'num_fragmentations', 'mota', 'motp', 'num_objects']
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)

    div_dict = {
        'num_objects': ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations'],
        'num_unique_objects': ['mostly_tracked', 'partially_tracked', 'mostly_lost']}

    for divisor in div_dict:
        for divided in div_dict[divisor]:
            summary[divided] = (summary[divided] / summary[divisor])
    fmt = mh.formatters
    change_fmt_list = ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations', 'mostly_tracked',
                       'partially_tracked', 'mostly_lost']
    for k in change_fmt_list:
        fmt[k] = fmt['mota']
    print(motm.io.render_summary(summary, formatters=fmt, namemap=motm.io.motchallenge_metric_names))

    metrics = motm.metrics.motchallenge_metrics + ['num_objects']
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
    print(motm.io.render_summary(summary, formatters=mh.formatters, namemap=motm.io.motchallenge_metric_names))
    config.logger.info('Completed')



if __name__ == '__main__':
    run_test()

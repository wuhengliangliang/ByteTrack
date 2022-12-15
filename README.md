# 目录

<!-- TOC -->

- [ByteTrack描述](#ByteTrack描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [训练](#单卡训练)
        - [分布式训练](#分布式训练)
    - [评估过程](#评估过程)
        - [评估](#评估)
    - [导出mindir模型](#导出mindir模型)
    - [推理过程](#推理过程)
        - [用法](#用法)
            - [相关说明](#相关说明)
        - [结果](#结果)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#ModelZoo主页)

<!-- TOC -->

# ByteTrack描述

ByteTrack 是一个简单、快速、强大的多对象跟踪器，通过关联每个 Detection Box 进行多对象跟踪。
致力于在学术研究和工业界之间架起一座桥梁。了解更多的网络细节，请参考ByteTrack论文。\
[论文](https://arxiv.org/abs/2110.06864): ```Bytetrack is accepted by ECCV 2022!```

[官方代码](https://github.com/ifzhang/ByteTrack)

# 模型架构

ByteTrack 使用当前性能非常优秀的检测器 YOLOX 得到检测结果。
\在数据关联的过程中，和 SORT 一样，只使用卡尔曼滤波来预测当前帧的跟踪轨迹在下一帧的位置，
\预测的框和实际的检测框之间的 IoU 作为两次匹配时的相似度，通过匈牙利算法完成匹配。

# 数据集

使用的数据集:使用COCO格式的数据集格式

支持的数据集: COCO 或者与 MS COCO 格式相同的数据集

支持的标注: COCO 或者与 MS COCO 相同格式的标注

- 目录结构如下，由用户定义目录和文件的名称

    ```ext

	   datasets
		   |——————mot
		   |        └——————train
		   |        └——————test
		   └——————crowdhuman
		   |         └——————Crowdhuman_train
		   |         └——————Crowdhuman_val
		   |         └——————annotation_train.odgt
		   |         └——————annotation_val.odgt
		   └——————MOT20
		   |        └——————train
		   |        └——————test
		   └——————Cityscapes
		   |        └——————images
		   |        └——————labels_with_ids
		   └——————ETHZ
					└——————eth01
					└——————...
					└——————eth07

    ```
然后，您需要将数据集转换为 COCO 格式并混合不同的训练数据：
cd <ByteTrack_HOME>
python3 tools/convert_mot17_to_coco.py
python3 tools/convert_mot20_to_coco.py
python3 tools/convert_crowdhuman_to_coco.py
python3 tools/convert_cityperson_to_coco.py
python3 tools/convert_ethz_to_coco.py

在混合不同的数据集之前，您需要按照mix_xxx.py中的操作创建一个数据文件夹和链接。最后，您可以混合训练数据：
cd <ByteTrack_HOME>
python3 tools/mix_data_ablation.py
python3 tools/mix_data_test_mot17.py
python3 tools/mix_data_test_mot20.py

- 如果用户需要自定义数据集，则需要将数据集格式转化为coco数据格式，并且，json文件中的数据要和图片数据对应好。

# 环境要求

- 硬件（Ascend）
    - 使用Ascend处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# 快速入门

- 通过官方网站安装Mindspore后，您可以按照如下步骤进行训练和评估


- 选择backbone:训练支持 yolox-darknet53 以及 yolox-x, 在训练之前需要指定backbone的名称，比如在default_config.yaml文件指定backbone为
	"yolox_darknet53"或者"yolox_x",你也可以在命令行手动指定backbone的名称，如 ```python train.py --backbone="yolox_x"```
- 训练分为前70轮和后10轮，区别主要在于后10轮的训练关闭了数据增强以及使用了L1 loss，若您不打算训练完80轮便打算终止，请将default_config.yaml文件中的total_epoch调小。


- 在本地进行训练

    ```shell
  # 单卡训练
	python train.py data_path=$DATASET_PATH data_aug=True is_distributed=0 eval_interval=10 load_path=yolox_x.ckpt backbone=yolox_x
    ```
	
  ```shell
  # 通过shell脚本进行8卡训练
	bash run_distribute_train.sh xxx/dataset/  yolox-x  pretrained/yolox_x.ckpt
  ```
  
- 在本地进行评估

    ```shell
    python eval.py --data_dir=./dataset/xxx --val_ckpt=your_val_ckpt_file_path --backbone=yolox-x --eval_batch_size=1

	```

# 脚本说明

## 脚本及样例代码

```text
    |----README_CN.md
    |----ascend310_infer
    |    |----build.sh
    |    |----CMakeLists.txt
    |    |----inc
    |    |    |----utils.h
    |    |----src
    |    |    |----main.cc
    |    |    |----utils.cc
    |----model_utils
    |    |----__init__.py
    |    |----config.py
    |    |----device_adapter.py
    |    |----local_adapter.py
    |    |----moxing_adapter.py
    |----scripts
    |    |----run_distribute_train.sh
    |    |----run_infer_310.sh
    |    |----run_eval.sh
    |    |----run_standalone_train.sh
	|----data
	|    |----__init__.py
    |    |----mosaicdetection.py
    |    |----mot.py
    |    |----transform.py
	|    |----mot.py
	|    |----yolox_dataset.py
    |    |----data_augment.py
	|----model
	|    |----__init__.py
    |    |----boxes.py
    |    |----darknet.py
    |    |----network_blocks.py
    |    |----yolox.py
	|    |----yolo_fpn.py
	|    |----yolo_pafpn.py
	|----tracker
    |    |----basetrack.py
    |    |----byte_tracker.py
    |    |----kalman_filter.py
    |    |----matching.py
    |----utils
    |    |----initializer.py
    |    |----logger.py
    |    |----util.py
    |    |----__init__.py
    |----train.py
    |----eval.py
    |----export.py
    |----postprocess.py
    |----preprocess.py
    |----default_config.yaml
```

## 脚本参数

train.py中主要的参数如下:

```text

--backbone                  训练的主干网络，默认为yolox_darknet53,你也可以设置为yolox_x
--data_aug                  是否开启数据增强，默认为True，在前面的训练轮次是开启的，最后的训练轮次关闭
--device_target
                            实现代码的设备，默认为'Ascend'
--outputs_dir               训练信息的保存文件目录
--save_graphs               是否保存图文件，默认为False
--max_epoch                 开启数据增强的训练轮次，默认为70
--total_epoch               总的训练轮次，默认为80
--no_aug_epochs             不开启数据增强，默认为10
--data_dir                  数据集的目录
--need_profiler
                            是否使用profiler。 0表示否，1表示是。 默认值：0
--per_batch_size            训练的批处理大小。 默认值：4
--max_gt                    图片中gt的最大数量，默认值：1000
--num_classes               数据集中类别的个数，默认值：1
--input_size                输入网络的尺度大小，默认值：[800, 1440]
--fpn_strides               fpn缩放的步幅，默认：[8, 16, 32]
--use_l1                    是否使用L1 loss，默认为False
--use_syc_bn                是否开启同步BN，默认True
--n_candidate_k             动态k中候选iou的个数，默认为10
--lr                        学习率，默认为0.01
--min_lr_ratio              学习率衰减比率，默认为0.05
--warmup_epochs             warm up 轮次，默认为2
--weight_decay              权重衰减，默认为0.0005
--momentum                  动量默认为0.9
--log_interval              日志记录间隔步数，默认为30
--ckpt_interval             保存checkpoint间隔。 默认值：-1
--is_save_on_master         在master或all rank上保存ckpt，1代表master，0代表all ranks。 默认值：1
--is_distributed            是否分发训练，1代表是，0代表否。 默认值：1
--rank                      分布式本地进程序号。 默认值：0
--group_size                设备进程总数。 默认值：1
--run_eval                  是否开启边训练边推理。默认为False
--device_num                采用8卡训练模式，默认为8
```

## 训练过程

由于 ByteTrack采用了YOLOX网络，而YOLOX网络使用了强大的数据增强，在ImageNet上的预训练模型参数不再重要，因此所有的训练都将从头开始训练。训练分为两步：第一步是从头训练并开启数据增强，第二步是使用第一步训练好的检查点文件作为预训练模型并关闭数据增强训练。

### 单卡训练

在Ascend设备上，使用python脚本直接开始训练(单卡)

    python命令启动

    ```shell
    # 单卡训练
    python train.py data_path=$DATASET_PATH data_aug=True is_distributed=0 eval_interval=10 load_path=yolox_x.ckpt backbone=yolox_x
    ```

    shell脚本启动

    ```shell
    bash run_standalone_train.sh  [DATASET_PATH] [BACKBONE] [PRETRAINED_CKPT]
    ```

    第一步训练结束后，在默认文件夹中找到最后一个轮次保存的检查点文件，并且将文件路径作为第二步训练的参数输入，如下所示：

### 分布式训练

在Ascend设备上，使用shell脚本执行分布式训练示例(8卡)

- 第一步

  ```shell

  # 通过shell脚本进行8卡训练
  bash run_distribute_train.sh xxx/dataset/ yolox-x pretrained/yolox_x.ckpt
  ```
  上述shell脚本将在后台运行分布式训练。 您可以通过train/log.txt文件查看结果。 得到如下损失值：

    ```log
    ...
	2022-09-07 18:13:19,017:INFO:epoch: 7 step: [30/766], loss: 4.2296, overflow: False, scale: 4096, lr: 0.039906, avg step time: 3399.11 ms
	2022-09-07 18:15:02,108:INFO:epoch: 7 step: [60/766], loss: 5.4202, overflow: False, scale: 4096, lr: 0.039902, avg step time: 3436.35 ms
	2022-09-07 18:16:43,538:INFO:epoch: 7 step: [90/766], loss: 5.1512, overflow: False, scale: 4096, lr: 0.039899, avg step time: 3380.99 ms
	2022-09-07 18:18:24,509:INFO:epoch: 7 step: [120/766], loss: 4.0267, overflow: False, scale: 4096, lr: 0.039895, avg step time: 3365.65 ms
	2022-09-07 18:20:05,368:INFO:epoch: 7 step: [150/766], loss: 5.5103, overflow: False, scale: 4096, lr: 0.039891, avg step time: 3361.93 ms
	2022-09-07 18:21:45,653:INFO:epoch: 7 step: [180/766], loss: 4.6702, overflow: False, scale: 4096, lr: 0.039887, avg step time: 3342.81 ms
	2022-09-07 18:23:26,179:INFO:epoch: 7 step: [210/766], loss: 5.4270, overflow: False, scale: 4096, lr: 0.039883, avg step time: 3350.84 ms
	2022-09-07 18:25:06,799:INFO:epoch: 7 step: [240/766], loss: 4.5086, overflow: False, scale: 4096, lr: 0.039879, avg step time: 3354.00 ms
	2022-09-07 18:26:46,874:INFO:epoch: 7 step: [270/766], loss: 5.4404, overflow: False, scale: 4096, lr: 0.039875, avg step time: 3335.79 ms
	2022-09-07 18:28:27,289:INFO:epoch: 7 step: [300/766], loss: 5.4157, overflow: False, scale: 4096, lr: 0.039871, avg step time: 3347.13 ms
	2022-09-07 18:30:07,555:INFO:epoch: 7 step: [330/766], loss: 5.7149, overflow: False, scale: 4096, lr: 0.039867, avg step time: 3342.19 ms
	2022-09-07 18:31:47,641:INFO:epoch: 7 step: [360/766], loss: 4.8638, overflow: False, scale: 4096, lr: 0.039862, avg step time: 3336.18 ms
	2022-09-07 18:33:27,448:INFO:epoch: 7 step: [390/766], loss: 5.0072, overflow: False, scale: 4096, lr: 0.039858, avg step time: 3326.85 ms
	2022-09-07 18:35:07,407:INFO:epoch: 7 step: [420/766], loss: 4.9238, overflow: False, scale: 4096, lr: 0.039853, avg step time: 3331.93 ms
	2022-09-07 18:36:46,999:INFO:epoch: 7 step: [450/766], loss: 4.9315, overflow: False, scale: 4096, lr: 0.039849, avg step time: 3319.70 ms
	2022-09-07 18:38:26,593:INFO:epoch: 7 step: [480/766], loss: 5.0438, overflow: False, scale: 4096, lr: 0.039844, avg step time: 3319.78 ms
	2022-09-07 18:40:06,158:INFO:epoch: 7 step: [510/766], loss: 5.0004, overflow: False, scale: 4096, lr: 0.039840, avg step time: 3318.81 ms
	2022-09-07 18:41:45,116:INFO:epoch: 7 step: [540/766], loss: 4.2428, overflow: False, scale: 4096, lr: 0.039835, avg step time: 3298.57 ms
	2022-09-07 18:43:24,390:INFO:epoch: 7 step: [570/766], loss: 5.2726, overflow: False, scale: 4096, lr: 0.039830, avg step time: 3309.13 ms
	2022-09-07 18:45:03,704:INFO:epoch: 7 step: [600/766], loss: 5.5637, overflow: False, scale: 4096, lr: 0.039825, avg step time: 3310.44 ms
	2022-09-07 18:46:45,732:INFO:epoch: 7 step: [630/766], loss: 5.0450, overflow: False, scale: 4096, lr: 0.039820, avg step time: 3400.88 ms
	2022-09-07 18:48:25,264:INFO:epoch: 7 step: [660/766], loss: 4.8747, overflow: False, scale: 4096, lr: 0.039815, avg step time: 3317.73 ms
	2022-09-07 18:50:05,139:INFO:epoch: 7 step: [690/766], loss: 4.3879, overflow: False, scale: 4096, lr: 0.039810, avg step time: 3329.14 ms
	2022-09-07 18:51:44,589:INFO:epoch: 7 step: [720/766], loss: 3.9797, overflow: False, scale: 4096, lr: 0.039805, avg step time: 3314.99 ms
	2022-09-07 18:53:26,479:INFO:epoch: 7 step: [750/766], loss: 4.2583, overflow: False, scale: 4096, lr: 0.039800, avg step time: 3396.30 ms
	2022-09-07 18:54:35,593:INFO:epoch: 7 epoch time 2578.55s loss: 4.1540, overflow: False, scale: 4096
    ...

    ```

## 评估过程

### 评估

#### python命令启动

```shell
python eval.py --data_dir=./dataset/xxx --val_ckpt=your_val_ckpt_file_path --per_batch_size=8 --backbone=yolox_x
```

backbone参数指定为yolox_darknet53或者yolox_x,上述python命令将在后台运行。 您可以通过```%Y-%m-%d_time_%H_%M_%S.log```文件查看结果。

#### shell脚本启动

```shell
bash run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH] [BACKBONE] [BATCH_SIZE]
```

```log

   ===============================coco eval result===============================
	 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.598
	 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.866
	 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.688
	 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.160
	 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.515
	 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.716
	 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.042
	 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.318
	 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.648
	 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.212
	 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.583
	 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.755
	
	                IDF1	IDP	     IDR	Rcll	Prcn	GT	MT	PT	ML	 FP	 FN	IDs	FM	MOTA	MOTP	IDt	IDa	IDm	num _ objects
MOT17-04-SDP	91.6%	93.1%	90.2%	93.9%	96.9%	69	61	6	2	734	1478 14	55	90.8%	0.135	7	9	3	24178
MOT17-11-SDP	69.2%	73.3%	65.5%	79.4%	88.9%	44	21	11	12	449	931	 14	26	69.1%	0.140	6	11	3	4517
MOT17-02-FRCNN	51.4%	59.4%	45.3%	63.6%	83.3%	53	19	25	9  1255 3600 85	198	50.0%	0.204	66	20	8	9880
MOT17-02-DPM	51.4%	59.4%	45.3%	63.6%	83.3%	53	19	25	9  1255	3600 85	198	50.0%	0.204	66	20	8	9880
MOT17-09-DPM	75.5%	82.0%	69.9%	83.6%	98.1%	22	16	5	1	47	471	 15	27	81.5%	0.160	19	2	6	2879
MOT17-05-SDP	72.1%	80.4%	65.4%	77.9%	95.9%	71	32	30	9	113	741	 21	46	73.9%	0.182	32	7	19	3357
MOT17-11-FRCNN	69.2%	73.3%	65.5%	79.4%	88.9%	44	21	11	12	449	931	 14	26	69.1%	0.140	6	11	3	4517
MOT17-13-FRCNN	72.0%	90.5%	59.7%	63.8%	96.7%	44	20	13	11	68 1142  8	26	61.4%	0.249	8	5	5	3156
MOT17-02-SDP	51.4%	59.4%	45.3%	63.6%	83.3%	53	19	25	9  1255 3600 85	198	50.0%	0.204	66	20	8	9880
MOT17-05-DPM	72.1%	80.4%	65.4%	77.9%	95.9%	71	32	30	9   113	741	 21	46	73.9%	0.182	32	7	19	3357
MOT17-04-FRCNN	91.6%	93.1%	90.2%	93.9%	96.9%	69	61	6	2	734	1478 14	55	90.8%	0.135	7	9	3	24178
MOT17-10-SDP	69.9%	78.2%	63.2%	75.7%	93.7%	36	14	20	2	300	1437 29	93	70.2%	0.222	17	15	5	5923
MOT17-13-SDP	72.0%	90.5%	59.7%	63.8%	96.7%	44	20	13	11	68	1142 8	26	61.4%	0.249	8	5	5	3156
MOT17-09-FRCNN	75.5%	82.0%	69.9%	83.6%	98.1%	22	16	5	1	47	471	 15	27	81.5%	0.160	19	2	6	2879
MOT17-05-FRCNN	72.2%	80.8%	65.3%	77.6%	96.0%	71	32	29	10	108	753	 23	46	73.7%	0.182	28	9	15	3357
MOT17-11-DPM	69.2%	73.3%	65.5%	79.4%	88.9%	44	21	11	12	449	931	 14	26	69.1%	0.140	6	11	3	4517
MOT17-09-SDP	75.5%	82.0%	69.9%	83.6%	98.1%	22	16	5	1	47	471	 15	27	81.5%	0.160	19	2	6	2879
MOT17-13-DPM	72.0%   90.5%	59.7%	63.8%	96.7%	44	20	13	11	68	1142 8	26	61.4%	0.249	8	5	5	3156
MOT17-10-FRCNN	69.9%   78.2%	63.2%	75.7%	93.7%	36	14	20	2	300	1437 29	93	70.2%	0.222	17	15	5	5923
MOT17-10-DPM	69.9%   78.2%	63.2%   75.7%	93.7%	36	14	20	2	300	1437 29	93	70.2%	0.222	17	15	5	5923
MOT17-04-DPM	91.6%   93.1%   90.2%   93.9%	96.9%	69	61	6	2	734	1478 14	55	90.8%	0.135	7	9	3	24178
0VERALL	      77.4%   83.1%   72.5%   81.8%	93.7% 1017 549 329 139 8893 29412 560  1413 76.0% 0.163	461	209	143	161670

```

## 导出mindir模型

```shell

python export.py --backbone [backbone] --val_ckpt [CKPT_PATH] --file_format [MINDIR/AIR]

```

参数```backbone```用于指定主干网络，你可以选择 yolox_darknet53 或者是 yolox_x ,```val_ckpt```用于导出的模型文件

## 推理过程

### 用法

#### 相关说明

- 首先要通过执行export.py导出mindir文件，同理可在配置文件中制定默认backbone的类型
- 通过preprocess.py将数据集转为二进制文件
- 执行postprocess.py将根据mindir网络输出结果进行推理，并保存评估指标等结果

执行完整的推理脚本如下：

```shell

# Ascend310 推理
bash run_infer_310.sh [MINDIR_PATH] [DATA_DIR] [DEVICE_ID]
```

### 结果

推理结果保存在当前路径，通过cat acc.log中看到最终精度结果。

```text

                                    yolox-x

=============coco eval result=========
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.594
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.888
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.674
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.203
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.524
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.690
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.040
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.314
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.645
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.322
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.592
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.730

2022-10-29 11:10:45,140:INFO:GT Type: _val_half
2022-10-29 11:10:45,142:INFO:GT Files: ['/home/stu/pl/ByteTrackMot-dev/bytetrackmot/mot/train/MOT17-09-FRCNN/gt/gt_val_half.txt', '/home/stu/pl/ByteTrackMot-dev/bytetrackmot/mot/train/MOT17-10-FRCNN/gt/gt_val_half.txt', '/home/stu/pl/ByteTrackMot-dev/bytetrackmot/mot/train/MOT17-02-FRCNN/gt/gt_val_half.txt', '/home/stu/pl/ByteTrackMot-dev/bytetrackmot/mot/train/MOT17-13-DPM/gt/gt_val_half.txt', '/home/stu/pl/ByteTrackMot-dev/bytetrackmot/mot/train/MOT17-05-DPM/gt/gt_val_half.txt', '/home/stu/pl/ByteTrackMot-dev/bytetrackmot/mot/train/MOT17-04-FRCNN/gt/gt_val_half.txt', '/home/stu/pl/ByteTrackMot-dev/bytetrackmot/mot/train/MOT17-05-SDP/gt/gt_val_half.txt', '/home/stu/pl/ByteTrackMot-dev/bytetrackmot/mot/train/MOT17-09-SDP/gt/gt_val_half.txt', '/home/stu/pl/ByteTrackMot-dev/bytetrackmot/mot/train/MOT17-04-DPM/gt/gt_val_half.txt', '/home/stu/pl/ByteTrackMot-dev/bytetrackmot/mot/train/MOT17-10-DPM/gt/gt_val_half.txt', '/home/stu/pl/ByteTrackMot-dev/bytetrackmot/mot/train/MOT17-10-SDP/gt/gt_val_half.txt', '/home/stu/pl/ByteTrackMot-dev/bytetrackmot/mot/train/MOT17-13-SDP/gt/gt_val_half.txt', '/home/stu/pl/ByteTrackMot-dev/bytetrackmot/mot/train/MOT17-11-FRCNN/gt/gt_val_half.txt', '/home/stu/pl/ByteTrackMot-dev/bytetrackmot/mot/train/MOT17-02-DPM/gt/gt_val_half.txt', '/home/stu/pl/ByteTrackMot-dev/bytetrackmot/mot/train/MOT17-04-SDP/gt/gt_val_half.txt', '/home/stu/pl/ByteTrackMot-dev/bytetrackmot/mot/train/MOT17-13-FRCNN/gt/gt_val_half.txt', '/home/stu/pl/ByteTrackMot-dev/bytetrackmot/mot/train/MOT17-05-FRCNN/gt/gt_val_half.txt', '/home/stu/pl/ByteTrackMot-dev/bytetrackmot/mot/train/MOT17-11-SDP/gt/gt_val_half.txt', '/home/stu/pl/ByteTrackMot-dev/bytetrackmot/mot/train/MOT17-02-SDP/gt/gt_val_half.txt', '/home/stu/pl/ByteTrackMot-dev/bytetrackmot/mot/train/MOT17-09-DPM/gt/gt_val_half.txt', '/home/stu/pl/ByteTrackMot-dev/bytetrackmot/mot/train/MOT17-11-DPM/gt/gt_val_half.txt']
2022-10-29 11:10:45,143:INFO:Found 21 groundtruths and 21 test files.
2022-10-29 11:10:45,143:INFO:Available LAP solvers ['lap', 'scipy']
2022-10-29 11:10:45,144:INFO:Default LAP solver 'lap'
2022-10-29 11:10:45,144:INFO:Loading files.
2022-10-29 11:10:53,097:INFO:Comparing MOT17-02-DPM...
2022-10-29 11:10:53,763:INFO:Comparing MOT17-02-FRCNN...
2022-10-29 11:10:54,414:INFO:Comparing MOT17-02-SDP...
2022-10-29 11:10:56,108:INFO:Comparing MOT17-04-DPM...
2022-10-29 11:10:57,573:INFO:Comparing MOT17-04-FRCNN...
2022-10-29 11:10:59,098:INFO:Comparing MOT17-04-SDP...
2022-10-29 11:11:01,585:INFO:Comparing MOT17-05-DPM...
2022-10-29 11:11:02,139:INFO:Comparing MOT17-05-FRCNN...
2022-10-29 11:11:02,698:INFO:Comparing MOT17-05-SDP...
2022-10-29 11:11:03,303:INFO:Comparing MOT17-09-DPM...
2022-10-29 11:11:03,734:INFO:Comparing MOT17-09-FRCNN...
2022-10-29 11:11:04,125:INFO:Comparing MOT17-09-SDP...
2022-10-29 11:11:04,508:INFO:Comparing MOT17-10-DPM...
2022-10-29 11:11:05,046:INFO:Comparing MOT17-10-FRCNN...
2022-10-29 11:11:05,587:INFO:Comparing MOT17-10-SDP...
2022-10-29 11:11:06,145:INFO:Comparing MOT17-11-DPM...
2022-10-29 11:11:06,789:INFO:Comparing MOT17-11-FRCNN...
2022-10-29 11:11:07,424:INFO:Comparing MOT17-11-SDP...
2022-10-29 11:11:08,052:INFO:Comparing MOT17-13-DPM...
2022-10-29 11:11:08,531:INFO:Comparing MOT17-13-FRCNN...
2022-10-29 11:11:09,009:INFO:Comparing MOT17-13-SDP...
2022-10-29 11:11:09,482:INFO:Running metrics
                Rcll  Prcn   GT    MT    PT    ML    FP    FN  IDs   FM  MOTA  MOTP num_objects
MOT17-02-DPM   66.0% 81.9%   53 32.1% 52.8% 15.1% 14.6% 34.0% 1.0% 2.1% 50.3% 0.205        9880
MOT17-02-FRCNN 66.0% 81.9%   53 32.1% 52.8% 15.1% 14.6% 34.0% 1.0% 2.1% 50.3% 0.205        9880
MOT17-02-SDP   66.0% 81.9%   53 32.1% 52.8% 15.1% 14.6% 34.0% 1.0% 2.1% 50.3% 0.205        9880
MOT17-04-DPM   92.9% 97.7%   69 85.5% 11.6%  2.9%  2.2%  7.1% 0.1% 0.3% 90.6% 0.141       24178
MOT17-04-FRCNN 92.9% 97.7%   69 85.5% 11.6%  2.9%  2.2%  7.1% 0.1% 0.3% 90.6% 0.141       24178
MOT17-04-SDP   92.9% 97.7%   69 85.5% 11.6%  2.9%  2.2%  7.1% 0.1% 0.3% 90.6% 0.141       24178
MOT17-05-DPM   77.8% 96.4%   71 43.7% 40.8% 15.5%  2.9% 22.2% 0.6% 1.1% 74.3% 0.186        3357
MOT17-05-FRCNN 77.6% 96.4%   71 43.7% 40.8% 15.5%  2.9% 22.4% 0.7% 1.1% 74.1% 0.186        3357
MOT17-05-SDP   77.8% 96.4%   71 43.7% 40.8% 15.5%  2.9% 22.2% 0.6% 1.1% 74.3% 0.186        3357
MOT17-09-DPM   83.5% 99.1%   22 72.7% 22.7%  4.5%  0.8% 16.5% 0.4% 1.0% 82.4% 0.155        2879
MOT17-09-FRCNN 83.5% 99.1%   22 72.7% 22.7%  4.5%  0.8% 16.5% 0.4% 1.0% 82.4% 0.155        2879
MOT17-09-SDP   83.5% 99.1%   22 72.7% 22.7%  4.5%  0.8% 16.5% 0.4% 1.0% 82.4% 0.155        2879
MOT17-10-DPM   72.5% 96.2%   36 41.7% 47.2% 11.1%  2.8% 27.5% 0.6% 1.4% 69.1% 0.222        5923
MOT17-10-FRCNN 72.5% 96.2%   36 41.7% 47.2% 11.1%  2.8% 27.5% 0.6% 1.4% 69.1% 0.222        5923
MOT17-10-SDP   72.5% 96.2%   36 41.7% 47.2% 11.1%  2.8% 27.5% 0.6% 1.4% 69.1% 0.222        5923
MOT17-11-DPM   80.2% 88.3%   44 50.0% 27.3% 22.7% 10.6% 19.8% 0.4% 0.7% 69.2% 0.143        4517
MOT17-11-FRCNN 80.2% 88.3%   44 50.0% 27.3% 22.7% 10.6% 19.8% 0.4% 0.7% 69.2% 0.143        4517
MOT17-11-SDP   80.2% 88.3%   44 50.0% 27.3% 22.7% 10.6% 19.8% 0.4% 0.7% 69.2% 0.143        4517
MOT17-13-DPM   64.3% 97.0%   44 45.5% 29.5% 25.0%  2.0% 35.7% 0.3% 0.8% 62.0% 0.240        3156
MOT17-13-FRCNN 64.3% 97.0%   44 45.5% 29.5% 25.0%  2.0% 35.7% 0.3% 0.8% 62.0% 0.240        3156
MOT17-13-SDP   64.3% 97.0%   44 45.5% 29.5% 25.0%  2.0% 35.7% 0.3% 0.8% 62.0% 0.240        3156
OVERALL        81.5% 94.0% 1017 53.1% 33.0% 13.9%  5.2% 18.5% 0.4% 0.9% 75.9% 0.167      161670
                IDF1   IDP   IDR  Rcll  Prcn   GT  MT  PT  ML   FP    FN IDs    FM  MOTA  MOTP IDt IDa IDm num_objects
MOT17-02-DPM   54.2% 60.7% 48.9% 66.0% 81.9%   53  17  28   8 1444  3363 100   203 50.3% 0.205  72  25   7        9880
MOT17-02-FRCNN 54.2% 60.7% 48.9% 66.0% 81.9%   53  17  28   8 1444  3363 100   203 50.3% 0.205  72  25   7        9880
MOT17-02-SDP   54.2% 60.7% 48.9% 66.0% 81.9%   53  17  28   8 1444  3363 100   203 50.3% 0.205  72  25   7        9880
MOT17-04-DPM   89.5% 91.8% 87.3% 92.9% 97.7%   69  59   8   2  539  1719  24    73 90.6% 0.141  12  14   7       24178
MOT17-04-FRCNN 89.5% 91.8% 87.3% 92.9% 97.7%   69  59   8   2  539  1719  24    73 90.6% 0.141  12  14   7       24178
MOT17-04-SDP   89.5% 91.8% 87.3% 92.9% 97.7%   69  59   8   2  539  1719  24    73 90.6% 0.141  12  14   7       24178
MOT17-05-DPM   72.7% 81.5% 65.7% 77.8% 96.4%   71  31  29  11   97   746  21    37 74.3% 0.186  30   7  17        3357
MOT17-05-FRCNN 72.1% 80.9% 65.1% 77.6% 96.4%   71  31  29  11   96   751  23    37 74.1% 0.186  27   9  14        3357
MOT17-05-SDP   72.7% 81.5% 65.7% 77.8% 96.4%   71  31  29  11   97   746  21    37 74.3% 0.186  30   7  17        3357
MOT17-09-DPM   77.2% 84.4% 71.1% 83.5% 99.1%   22  16   5   1   22   475  11    29 82.4% 0.155  12   2   4        2879
MOT17-09-FRCNN 77.2% 84.4% 71.1% 83.5% 99.1%   22  16   5   1   22   475  11    29 82.4% 0.155  12   2   4        2879
MOT17-09-SDP   77.2% 84.4% 71.1% 83.5% 99.1%   22  16   5   1   22   475  11    29 82.4% 0.155  12   2   4        2879
MOT17-10-DPM   66.1% 76.9% 58.0% 72.5% 96.2%   36  15  17   4  168  1626  38    82 69.1% 0.222  23  16   4        5923
MOT17-10-FRCNN 66.1% 76.9% 58.0% 72.5% 96.2%   36  15  17   4  168  1626  38    82 69.1% 0.222  23  16   4        5923
MOT17-10-SDP   66.1% 76.9% 58.0% 72.5% 96.2%   36  15  17   4  168  1626  38    82 69.1% 0.222  23  16   4        5923
MOT17-11-DPM   68.1% 71.6% 65.0% 80.2% 88.3%   44  22  12  10  478   896  16    32 69.2% 0.143   8  10   2        4517
MOT17-11-FRCNN 68.1% 71.6% 65.0% 80.2% 88.3%   44  22  12  10  478   896  16    32 69.2% 0.143   8  10   2        4517
MOT17-11-SDP   68.1% 71.6% 65.0% 80.2% 88.3%   44  22  12  10  478   896  16    32 69.2% 0.143   8  10   2        4517
MOT17-13-DPM   71.1% 89.2% 59.1% 64.3% 97.0%   44  20  13  11   63  1128   8    26 62.0% 0.240   4   6   3        3156
MOT17-13-FRCNN 71.1% 89.2% 59.1% 64.3% 97.0%   44  20  13  11   63  1128   8    26 62.0% 0.240   4   6   3        3156
MOT17-13-SDP   71.1% 89.2% 59.1% 64.3% 97.0%   44  20  13  11   63  1128   8    26 62.0% 0.240   4   6   3        3156
OVERALL        76.4% 82.2% 71.3% 81.5% 94.0% 1017 540 336 141 8432 29864 656  1446 75.9% 0.167 480 242 129      161670																			
																
													
请浏览官网[主页](https://gitee.com/mindspore/models)。

@misc{ByteTrack,
  title={ByteTrack: Multi-Object Tracking by Associating Every Detection Box},
  author={Liang Peng,JiChen Zhao},
  year={2022}
}

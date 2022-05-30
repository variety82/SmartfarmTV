
#%%
import mmcv
from mmcv import Config
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.apis import single_gpu_test
from mmcv.runner import load_checkpoint
import os
from mmcv.parallel import MMDataParallel
import pandas as pd
from pandas import DataFrame
from pycocotools.coco import COCO
import numpy as np
import torch 

import os
import random
import time
import json
import warnings 
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
#from utils import label_accuracy_score, add_hist
import cv2

import numpy as np
import pandas as pd
from tqdm import tqdm

# 전처리를 위한 라이브러리
from pycocotools.coco import COCO
import torchvision
import torchvision.transforms as transforms

#!pip install albumentations==0.4.6
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 시각화를 위한 라이브러리
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from matplotlib.patches import Patch

#!pip install webcolors
import webcolors

plt.rcParams['axes.grid'] = False






classes = ["Cabbage"]
'''
# config file 들고오기
cfg = Config.fromfile('/opt/ml/mmdetection/mywork/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_20e_coco.py')

root='./data/'

epoch = 'latest'

# dataset config 수정
cfg.data.test.classes = classes
cfg.data.test.img_prefix = root + 'val/images_normal/'
cfg.data.test.ann_file = root + 'coco/val_normal_annotations.json'
cfg.data.test.pipeline[1]['img_scale'] = (512,512) # Resize
cfg.data.test.test_mode = True

cfg.data.samples_per_gpu = 4

cfg.seed=2021
cfg.gpu_ids = [1]
cfg.work_dir = './work_dirs/cascade_rcnn_x101_64x4d_fpn_20e_coco/'

#cfg.model.roi_head.bbox_head.num_classes = 1

cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
cfg.model.train_cfg = None

# build dataset & dataloader
dataset = build_dataset(cfg.data.test)
data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

# checkpoint path
checkpoint_path = os.path.join(cfg.work_dir, f'{epoch}.pth')
'''
from mmdet.apis import init_detector, inference_detector

config_file = '/opt/ml/mmdetection/mywork/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_20e_coco.py'
checkpoint_file = './work_dirs/cascade_rcnn_x101_64x4d_fpn_20e_coco/best.pth'
model = init_detector(config_file,checkpoint_file,device='cuda:0')

import cv2
import matplotlib.pyplot as plt

img = '/opt/ml/mmdetection/data/simple_test/val_img1.png'

img_arr = cv2.cvtColor(cv2.imread(img),cv2.COLOR_BGR2RGB)
plt.figure(figsize=(12,12))
plt.imshow(img_arr)

img = '/opt/ml/mmdetection/data/simple_test/val_img1.png'
results = inference_detector(model,img)

from mmdet.apis import show_result_pyplot

#show_result_pyplot(model,img,results,palette=([255,0,255],[255,0,255]))
show_result_pyplot(model,img,results)
print("finish")


# %%

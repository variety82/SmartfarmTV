
#%%

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
import numpy as np

#!pip install webcolors
import webcolors

plt.rcParams['axes.grid'] = False

from mmdet.apis import init_detector, inference_detector

config_file = '/opt/ml/mmdetection/mywork/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_20e_coco.py'
checkpoint_file = './work_dirs/cascade_rcnn_x101_64x4d_fpn_20e_coco/best.pth'
model = init_detector(config_file,checkpoint_file,device='cuda:0')

import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('/opt/ml/mmdetection/data/val/images_normal/V006_79_0_00_03_03_12_0_c32_20201117_0498_S01_1.jpg',1)
img2 = cv2.imread('/opt/ml/mmdetection/data/val/images_normal/V006_79_0_00_03_03_12_0_b06_20201109_0149_S01_1.jpg',1)

img1 = cv2.resize(img1,(256,256))
img2 = cv2.resize(img2,(256,256))

upper_img = cv2.hconcat([img1,img2])

img3 = cv2.imread('/opt/ml/mmdetection/data/val/images_normal/V006_79_0_00_03_03_12_0_c32_20201117_0431_S01_1.jpg',1)
img4 = cv2.imread('/opt/ml/mmdetection/data/val/images_normal/V006_79_0_00_03_03_12_0_c32_20201117_1073_S01_1.jpg',1)

img3 = cv2.resize(img3,(256,256))
img4 = cv2.resize(img4,(256,256))

lower_img = cv2.hconcat([img3,img4])

img = cv2.vconcat([upper_img,lower_img])


img_arr = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12,12))
plt.imshow(img_arr)

results = inference_detector(model,img)

from mmdet.apis import show_result_pyplot

show_result_pyplot(model,img,results)
print("finish")


# %%

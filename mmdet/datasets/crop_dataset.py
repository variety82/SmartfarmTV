import copy
import os.path as osp
import cv2

import mmcv
import numpy as np
import pandas as pd

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset

CLASSES = ('detected')
cat2label = {k:i for i, k in enumerate(CLASSES)}

# 반드시 아래 Decorator 설정 할것.@DATASETS.register_module() 설정 시 force=True를 입력하지 않으면 Dataset 재등록 불가. 
@DATASETS.register_module(force=True)
class CropDataset(CustomDataset):
  CLASSES = ('detected')
  
  ##### self.data_root: ./kitti_tiny/ self.ann_file: ./kitti_tiny/train.txt self.img_prefix: ./kitti_tiny/training/image_2
  #### ann_file: ./kitti_tiny/train.txt
  # annotation에 대한 모든 파일명을 가지고 있는 텍스트 파일을 __init__(self, ann_file)로 입력 받고, 이 self.ann_file이 load_annotations()의 인자로 입력
  def load_annotations(self, ann_file):
    print('##### self.data_root:', self.data_root, 'self.ann_file:', self.ann_file, 'self.img_prefix:', self.img_prefix)
    print('#### ann_file:', ann_file)
    
    CLASSES = ('detected')
    cat2label = {'detected': 0}
    image_list = self.ann_file
    data_infos = []
    
    #root = "/Users/somang/Desktop/mymac/boostcamp/final_proj/dataset/Training/label_normal/"
    if "train" in ann_file:
        root = '/opt/ml/mmdetection/data/train/images_normal/'
    else:
        root = '/opt/ml/mmdetection/data/val/images_normal/'
    #img_prefix = "/Users/somang/Desktop/mymac/boostcamp/final_proj/dataset/Training/images_normal"
    for file_name in image_list['0']:
        with open(root+file_name,'r') as j: #annotation 경로 추가
                dic = json.loads(j.read())
                only_img_name = dic['description']['image']
                img_file = '{0:}/{1:}'.format(self.img_prefix, only_img_name)
                image = cv2.imread(img_file)
                height, width = image.shape[:2]
                # 개별 image의 annotation 정보 저장용 Dict 생성. key값 filename 에는 image의 파일명만 들어감(디렉토리는 제외)
                data_info = {'filename': str(only_img_name),'width': width, 'height': height}
                
                anno = dic['annotations']
                points = anno['points']
                bbox_names = 'detected' #detected
                # bbox 좌표를 저장
                bboxes = [points[0]['xtl'],points[0]['ytl'],points[0]['xbr'],points[0]['ybr']]
            
                # 클래스명이 해당 사항이 없는 대상 Filtering out, 'DontCare'sms ignore로 별도 저장.
                gt_bboxes = []
                gt_labels = []
                
                
                gt_bboxes.append(bboxes)
                #gt_labels에는 class id를 입력
                gt_labels.append(cat2label[bbox_names])
                print(bboxes)
                    
                # 개별 image별 annotation 정보를 가지는 Dict 생성. 해당 Dict의 value값은 모두 np.array임. 
                data_anno = {
                    'bboxes': np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
                    'labels': np.array(gt_labels, dtype=np.long)
                }
                data_info.update(ann=data_anno)
                data_infos.append(data_info)
                print(data_infos)

    return data_infos
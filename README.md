## :bulb: Model Installation Guide

### Git Clone
```bash
#Pytorch 설치 필요

git clone https://github.com/boostcampaitech3/final-project-level3-cv-01.git
cd model
```

### 1. MMDetection 
```bash
pip install -U openmim
mim install mmcv-full
cd mmdetection
pip install -v -e 
```

### 2. MMClassfication
```bash
#"1" 의 openmim, mmcv-full 설치필요

cd mmclassification
pip install -v -e 
```
<br>
<br>


## :file_folder: Checkpoint Download


1. [배추 Detection pth]()
2. [해충 Detection pth](https://drive.google.com/file/d/1vVC38mZDHUqYGZEhVeYePIpCbWcwn3Rl/view?usp=sharing)
3. [질병 Classification pth](https://drive.google.com/file/d/1tRxeN1ahd5aGez7EDBYGYN3QykBbXMgy/view?usp=sharing)




<br>
<br>

## :notebook: Inference Usage

### 배추 Detection Model
```bash
```


### 해충 Detection Model
```bash
#pth file 다운 및 config/1.MyConfig/datasets/custom.py 에서 test_pipeline 수정필요

cd mmdetection
python tools/test.py configs/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_1x_coco.py \
{pth_path} --options "jsonfile_prefix={output filename}"
```

### 질병 Classification Model
```bash
#pth file 다운 및 config/1.MyConfig/datasets/custom.py 에서 test_pipeline 수정필요 

cd mmclassification
python tools/test.py configs/1.MyConfig/resnext/my_resnext152-32x4d_8xb32_in1k.py \
{pth_path} --out {output filename} --out-itmes all
```

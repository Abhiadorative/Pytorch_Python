# Deep Sort with PyTorch

![Demo](demo/demo.gif)

This repository contains an implementation of the Deep SORT (Simple Online and Realtime Tracking with a Deep Association Metric) algorithm using PyTorch. Deep SORT extends the SORT algorithm by integrating a CNN model to extract features from detected objects, specifically human parts. The CNN model, used for re-identification (RE-ID), enhances the performance of the tracking system. The original paper utilizes FasterRCNN as the detector.

## Further Improvement Directions
- Train the detector on a specific dataset rather than the default one.
- Retrain the REID model on a pedestrian dataset for better performance.
- Replace the YOLOv3 detector with more advanced ones.

## Dependencies
Ensure you have the following dependencies installed:
- Python 3 (Python 2 compatibility not guaranteed)
- numpy
- scipy
- opencv-python
- sklearn
- torch >= 0.4
- torchvision >= 0.1
- pillow
- vizer
- edict

## Quick Start

### 0. Check Dependencies
Install all required dependencies:
```bash
pip install -r requirements.txt
```

### 1. Clone the Repository
```bash
git clone git@github.com:ZQPei/deep_sort_pytorch.git
```

### 2. Download YOLOv3 Parameters
```bash
cd detector/YOLOv3/weight/
wget https://pjreddie.com/media/files/yolov3.weights
wget https://pjreddie.com/media/files/yolov3-tiny.weights
cd ../../../
```

### 3. Download Deep SORT Parameters
```bash
cd deep_sort/deep/checkpoint
# Download ckpt.t7 from the following link to this folder:
# https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6
cd ../../../
```

### 4. Compile NMS Module
```bash
cd detector/YOLOv3/nms
sh build.sh
cd ../../..
```
*Note: If compilation fails, upgrade PyTorch to >= 1.1 and torchvision to >= 0.3 to avoid issues related to low `gcc` versions or missing libraries.*

### 5. Run the Demo
```bash
# General usage
python yolov3_deepsort.py VIDEO_PATH

# Using yolov3_tiny
python yolov3_deepsort.py VIDEO_PATH --config_detection ./configs/yolov3_tiny.yaml

# Using yolov3 with webcam
python3 yolov3_deepsort.py /dev/video0 --camera 0

# Using yolov3_tiny with webcam
python3 yolov3_deepsort.py /dev/video0 --config_detection ./configs/yolov3_tiny.yaml --camera 0
```
Use the `--display` flag to enable the display. Results will be saved to `./output/results.avi` and `./output/results.txt`.

All files can also be accessed from BaiduDisk!  
[Link](https://pan.baidu.com/s/1YJ1iPpdFTlUyLFoonYvozg)  
Password: fbuw

## Training the RE-ID Model
The original model used in the paper is in `original_model.py`, and its parameters can be found [here](https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6).

To train the model:
1. Download the [Market1501](http://www.liangzheng.com.cn/Project/project_reid.html) or [Mars](http://www.liangzheng.com.cn/Project/project_mars.html) dataset.
2. Use `train.py` to train your model parameters and evaluate them using `test.py` and `evaluate.py`.

![Training](deep_sort/deep/train.jpg)

## Demo Videos and Images
- [Demo Video 1](https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6)
- [Demo Video 2](https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6)

![Demo Image 1](demo/1.jpg)
![Demo Image 2](demo/2.jpg)

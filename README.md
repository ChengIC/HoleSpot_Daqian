# HoleSpot_Daqian

Project HoleSpot for detecting Potholes with computer vision techniques

### Evaluation of Potholes 
Store 2D detection, image classification and 3D reconsturction results into following folder structure. The evaluation pipeline is demonstrated in the jupyter notebook.

## Potholes Localisation in 2D images
### Train 2D object detector 
Download potholes images zip file from: https://drive.google.com/file/d/1C4nMLNE1-rUR4UgYHjCc7CTbIAs3mPmb/view?usp=sharing and unzip into yolov5_src folder for training 
```
python ./yolov5_src/train.py
```
You can add more labelled images and annotations into potholes image folder for better accuracy 

### Inference 2D object detector
```
python python detect.py --weights [your training exp pt file] --source [inferenced video frames] --device 0 --save-txt --save-conf --project [your saved folder]
```

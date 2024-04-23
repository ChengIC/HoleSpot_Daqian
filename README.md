# HoleSpot_Daqian

Project HoleSpot for detecting Potholes with computer vision techniques (Prize Winning: 97,500SAR - around 20,000GBP!)

[Our Demo](https://www.youtube.com/watch?v=zf_WY1KBnTs&t=292s)

![Prize Winning](Prize_winning.jpg)

#### Team -- Daqian
[Ziyi Zhu](https://www.linkedin.com/in/ziyizhu/) 

[Junhui Yang](https://www.linkedin.com/in/junhui-yang/)

[Kaichen Zhou](https://www.linkedin.com/in/kaichen-z-423579139/)

[Ran Cheng](https://www.linkedin.com/in/ran-cheng-9438ic/)

#### Demo
Unzip project folder for scene 1 to understand how pipeline works: https://drive.google.com/file/d/1ERv7pg27XqPUEJk167JbcKGC6bla8ZsP/view?usp=sharing


### Evaluation of Potholes 
Store 2D detection, image classification and 3D reconsturction results into following folder structure. The evaluation pipeline is demonstrated in the jupyter notebook.

``` bash
├── evaluation_pipeline.ipynb
├── potholes_evaluation
│   └── scene1_e8Phytelu4
│       ├── imgs
│       └── results
│           ├── mild
│           ├── moderate
│           ├── results.csv
│           └── severe
├── road_evaluation
│   └── scene1_e8Phytelu4
│       ├── imgs
│       │   ├── most_severe
│       │   └── rep_frames
│       └── summary
│           ├── frames.csv
│           ├── overall_summary.csv
│           ├── potholes.csv
│           └── road_segment.csv
├── threeD_reconstruction
│   └── scene1_e8Phytelu4
│       └── results
│           ├── depth_data
│           ├── depth_imgs
│           ├── pose_data
│           │   ├── pose_x.txt
│           │   └── pose_z.txt
│           └── pose_imgs
├── tree.text
├── twoD_detection
│   └── scene1_e8Phytelu4
│       └── results
│           ├── imgs
│           └── labels
└── uploaded_files

```


## Potholes Localisation in 2D images
### Train 2D object detector 
Download potholes images zip file from: https://drive.google.com/file/d/1C4nMLNE1-rUR4UgYHjCc7CTbIAs3mPmb/view?usp=sharing and unzip into yolov5_src folder for training 
```
python ./yolov5_src/train.py
```
You can add more labelled images and annotations into potholes image folder for better accuracy 

### Inference 2D object detector
```
python detect.py --weights [your training exp pt file] --source [inferenced video frames] --device 0 --save-txt --save-conf --project [your saved folder]
```

## Pothole Severity Classification
All relevant codes are contained in directory `severity_classifier_trainer`. 
### Download the data
You may download the pothole severity raw training data from: https://drive.google.com/file/d/18PsmxDq2wgA0hWQ27UJbVh71CoT1lwQG/view?usp=sharing and the Theme 2 pothole evaluation data from: https://drive.google.com/file/d/1_JnsxcUYa2Iw3G3DBZKWpFZPdpREfnU3/view?usp=sharing. Unzip them into `severity_classifier_trainer/data`. 
### Training data generation
For training data generation, please follow `training_data_creator.ipynb`. 
### Training and inferencing
For one-vs-all classifier method, please follow `one_vs_all_mobilenet_classifier.ipynb`. 
For single classifier method, please follow `single_mobilenet_classifier.ipynb`. 


## Foreground Mask & Trajectory Estimation & 3D Geometry Learning of HoleSpot 

### Generating Mask for Foreground:
```
Python sky_mask.py
```

### Training:
```
sh start2train.sh
# Trained weights will be saved to mono_model/log1/
```

### Evaluating Depth Estimation 
```
sh start2eval.sh
```

### Evaluating Trajectory Estimation 
```
sh start2eval_pose.sh
```

#### Acknowledgement
 Thanks the authors for their works:
 - [monodepth2](https://github.com/nianticlabs/monodepth2)
 - [HRNet](https://github.com/HRNet/HRNet-Semantic-Segmentation)
- [DIFFNET](https://github.com/brandleyzhou/DIFFNet/tree/a4d74f131738bdb1f8feaa52baa58de3697959e7)


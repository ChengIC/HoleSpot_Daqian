# HoleSpot_Daqian

Project HoleSpot for detecting Potholes with computer vision techniques

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
python python detect.py --weights [your training exp pt file] --source [inferenced video frames] --device 0 --save-txt --save-conf --project [your saved folder]
```

## Pothole Severity Classification
All relevant codes are contained in directory `severity_classifier_trainer`. 
### Download the data
You may download the pothole severity raw data from: https://drive.google.com/file/d/18PsmxDq2wgA0hWQ27UJbVh71CoT1lwQG/view?usp=sharing and unzip into `severity_classifier_trainer/data`. 
### Training data generation
For training data generation, please follow `training_data_creator.ipynb`. 
### Training and inferencing
For one-vs-all classifier training, please follow `one_vs_all_mobilenet_classifier.ipynb`. 
For single classifier training, please follow `single_mobilenet_classifier.ipynb`. 

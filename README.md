# Anomaly-Detection-in-Pose-Estimation
Anomaly Detection in Pose Estimation

In this repo a simple method for anomaly detection in predicted poses is presented
## Pose Estimation model
The pose estimation model that is used for extracting the poses is the LightWeight OpenPose

https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch
## Methodology
The methodology is presented in the following steps: 

* Pose estimation using the pre-trained model of LightWeight Open Pose 
(link: https://drive.google.com/file/d/18Ya27IAhILvBHqV_tDp0QjDFvsNNy-hv/view)

* The extracted poses-keypoints are used as input to an anomaly detection Autoencoder model

* According to reconstruction loss the image is characterized as normal or abnormal

## Results

Normal             |  Abnormal | Abnormal |
:-------------------------:|:-------------------------:|:-------------------------: 
![](/images_examples/normal5.png)  |  ![](/images_examples/abnormal6.png) |  ![](/images_examples/abnormal.png)

## Train

## Evaluation

## Inference
For testing the pre-trained anomaly detector

Run `` python autoencoder_results_test_val.py ``

The inference is performed on some testing images in /test_images directory

Alternatively, it can be run on any directory by modifying the respective line in autoencoder_results_test_val.py https://github.com/mthodoris/PoseAnomalyDetect/blob/ed80173f6b69bcb94f7fc331af346d93274a3033/autoencoder_results_test_val.py#L325

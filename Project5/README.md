# Action Recognition in Video

This repo will serve as a playground where I investigate different approaches to solving the problem of action recognition in video.

I use the dataset collected from the public action video datasets, such as ORGBD, MSRDaily and UTKinect. The dataset includes 18 classes, including, reading phone, drinking, using laptop and etc.


##Testing on videos
### Step1: Download the videos with extension .avi into the folder /test/test_videos
### Step2: Download the 3D skeleton data with extension .npy into the folder /test/test_videos

The video data is named as FileName_rgb.avi and the corresponding skeleton data is named as FileName_skeleton.npy.

The skeleton data in each .npy file records a T length list whose each element is a 20*3 array. This represent T frames of the video ,20 joints of skeleton and 3D spatial location.

### Step3: Download the trained model into the folder /model_checkpoints and rename the model as 'ConvLSTM.pth' 

The link of the trained model is [here](https://drive.google.com/open?id=1bPpS4-YTzK8-m3hYs7jtjPOdXBN9YkEj)

### Step4: Run
```
$ python3 test_on_video.py  --video_path test/test_videos \
                            --class_path classes.txt \
                            --checkpoint_model model_checkpoints/ConvLSTM.pth
```

# YOLOv3-Detection
detecting pedestrians using YOLOv3 (supporting images and videos)

This repository is adapteded from 
https://github.com/eriklindernoren/PyTorch-YOLOv3. 

Sometimes there will be FPs,to improve accuracy we can add a post CNN filter following the cropped proposal. More details are in post_cnn_train.py, post_cnn_detect.py and post_filter.py.

I write a video.py for video detection.( detect.py is for image folder)


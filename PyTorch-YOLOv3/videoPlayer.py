from __future__ import division
import numpy as np
import argparse
#import imutils
import cv2

import torchvision as tv

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

ap= argparse.ArgumentParser()

###
ap.add_argument('--config_path', type=str, default='config/yolov3.cfg', help='path to model config file')
ap.add_argument('--weights_path', type=str, default='weights/yolov3.weights', help='path to weights file')
ap.add_argument('--class_path', type=str, default='data/coco.names', help='path to class label file')
ap.add_argument('--conf_thres', type=float, default=0.8, help='object confidence threshold')
ap.add_argument('--nms_thres', type=float, default=0.4, help='iou thresshold for non-maximum suppression')
ap.add_argument('--batch_size', type=int, default=1, help='size of the batches')
ap.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
ap.add_argument('--img_size', type=int, default=416, help='size of each image dimension')
ap.add_argument('--use_cuda', type=bool, default=True, help='whether to use cuda if available')
ap.add_argument("-v", "--video",default="4.mp4",
    help="path to the (optional) video file")
opt=ap.parse_args()
args = vars(ap.parse_args())
# Set up model
model = Darknet(opt.config_path, img_size=opt.img_size)
model.load_weights(opt.weights_path)
cuda = torch.cuda.is_available() and opt.use_cuda
if cuda:
    model.cuda()

model.eval() # Set in evaluation mode


index = 0
if not args.get("video", False):
    camera = cv2.VideoCapture(0)
# otherwise load the video
else:
    camera = cv2.VideoCapture(args["video"])

# Bounding-box colors
#cmap = plt.get_cmap('tab20b')
#colors = [cmap(i) for i in np.linspace(0, 1, 20)]
classes = load_classes(opt.class_path)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

fps = camera.get(cv2.CAP_PROP_FPS)
size = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
#cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
out = cv2.VideoWriter('out.avi',cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)
while True:
    #index = index+1
    # grab the current frame
    (grabbed, frame) = camera.read()
    #print(frame.shape)
    # if we are viewing a video and did not a grab a frame then we have reached
    # the end of the video
    if args.get("video") and not grabbed:
        break

    # resize, convert to grayscale, and then clone it (so we can annotate it)
    # frame = imutils.resize(frame, width=300)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = frame.copy()

    h, w, _ = img.shape
    dim_diff = np.abs(h - w)
    # Upper (left) and lower (right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
    # Add padding
    input_img = np.pad(img, pad, 'constant', constant_values=127.5) / 255.

    # Resize and normalize
    input_img = resize(input_img, (opt.img_size,opt.img_size, 3), mode='reflect')
    #cv2.imshow("image", input_img)
    # Channels-first
    input_img = np.transpose(input_img, (2, 0, 1))
    # As pytorch tensor
    input_img = torch.from_numpy(input_img).float()
    #print(input_img.shape)
    input_img=input_img.view(1,input_img.shape[0],input_img.shape[1],input_img.shape[2])
    #print(input_img)
    input_img = Variable(input_img.type(Tensor))
    #print(input_img)


    # Get detections
    with torch.no_grad():
        detections = model(input_img)
        #print(detections)
       	detections = non_max_suppression(detections, 80, opt.conf_thres, opt.nms_thres)
        #print(detections)
        detections=detections[0]
        #print(detections)
    #print ('\t+ Batch %d, Inference Time: %s' % (batch_i, inference_time))

    #input_img=input_img.data.numpy()
    #input_img=input_img.reshape((input_image.shape[1],input_image.shape[2],input_image.shape[3]))
    # The amount of padding that was added
    pad_x = max(frame.shape[0] - frame.shape[1], 0) * (opt.img_size / max(frame.shape))
    pad_y = max(frame.shape[1] - frame.shape[0], 0) * (opt.img_size / max(frame.shape))
    # Image height and width after padding is removed
    unpad_h = opt.img_size - pad_y
    unpad_w = opt.img_size - pad_x

    # Draw bounding boxes and labels of detections

    #unique_labels = detections[:, -1].cpu().unique()
     #   n_cls_preds = len(unique_labels)
      #  bbox_colors = random.sample(colors, n_cls_preds)

    if detections is not None:
        #print(detections.type)
       	#unique_labels = detections[:, -1].cpu().unique()
        #n_cls_preds = len(unique_labels)
        #bbox_colors = random.sample(colors, n_cls_preds)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            if cls_pred>0:
                continue
            print ('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))

            # Rescale coordinates to original dimensions
            box_h = ((y2 - y1) / unpad_h) * frame.shape[0]
            box_w = ((x2 - x1) / unpad_w) * frame.shape[1]
            y1 = ((y1 - pad_y // 2) / unpad_h) * frame.shape[0]
            x1 = ((x1 - pad_x // 2) / unpad_w) * frame.shape[1]
            print(x1,y1,box_h,box_w)
            #import time
            cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), (0, 255, 0), 2)

            #time.sleep(1)
            #break
            #color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
    #cv2.putText(frameClone, "hello world", (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX,
        #0.45, (0, 0, 255), 2)
    cv2.imshow("video", frame)

    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# clean up
camera.release()
out.release()
cv2.destroyAllWindows()

from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
import cv2
import os
import sys
import time
import datetime
import argparse
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from post_cnn_train import *
import torchvision.models as models


classes=('nonperson','person')
cuda = torch.cuda.is_available()

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def post_filter(image):
    #preprocess
    image = image / 255
    image = cv2.resize(image, (32, 64))
    b, g, r = cv2.split(image)
    image = cv2.merge([r, g, b])

    image = torch.from_numpy(image)
    # print(image)
    image = np.transpose(image, (2, 0, 1))
    print(image.shape)
    image.unsqueeze_(0)
    #load net
    Net = torch.load('post_cnn_weights/net39.pkl')
    Net.load_state_dict(torch.load('post_cnn_weights/net_params39.pkl'))

    image= Variable(image.cuda().float())
    out = Net(image)
    _, predicted = torch.max(out.data, 1)
    return predicted


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

cuda = torch.cuda.is_available()

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


image=cv2.imread('data/samples/1.png')
image= cv2.resize(image, (128,64), interpolation=cv2.INTER_CUBIC)
image=torch.from_numpy(image)
image= np.transpose(image, (2, 0, 1))
image.unsqueeze_(0)



print(image.shape)
Net=torch.load('post_cnn_weights/net69.pkl')
Net.load_state_dict(torch.load('post_cnn_weights/net_params69.pkl'))


print(type(image))
print(image.shape)

image = Variable(image.type(Tensor))
out=Net(image)
print(out)
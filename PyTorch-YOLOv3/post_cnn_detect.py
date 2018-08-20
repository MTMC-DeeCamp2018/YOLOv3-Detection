
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

#
#
#image=cv2.imread('post_train/neg/shadow_0_0.png')
image=cv2.imread('data/samples/shadow_0_6003.png')
#print(image)
#image=cv2.imread('post_train/pos/333.png')
#image=cv2.imread('shadow/532.png')


print(image)
image=image/255
image= cv2.resize(image, (32,64))
b,g,r=cv2.split(image)
image=cv2.merge([r,g,b])

image=torch.from_numpy(image)
#print(image)
image= np.transpose(image, (2, 0, 1))
print(image.shape)
image.unsqueeze_(0)



print(image)
Net=torch.load('post_cnn_weights_64_v2/net39.pkl')
#print(Net)
Net.load_state_dict(torch.load('post_cnn_weights_64_v2/net_params39.pkl'))


#print(type(image))
#print(image.shape)
#
# trainloader = loadtraindata()
# for i, data in enumerate(trainloader, 0):  # 这里我们遇到了第一步中出现的trailoader，代码传入数据
#     if i >0:
#         break
#     # enumerate是python的内置函数，既获得索引也获得数据
#     # get the inputs
#     #print(data)
#     inputs, labels = data  # data是从enumerate返回的data，包含数据和标签信息，分别赋值给inputs和labels
#     inputs.squeeze_(0)
#     inputs = np.transpose(inputs, (1,2, 0))
#     print(inputs.shape)
#     print(inputs.numpy())
#     #cv2.imwrite("1.png",(inputs.squeeze_(0)).numpy())
#     inputs=inputs*255
#     cv2.imwrite("111.png",inputs.numpy())
#
#     inputs = inputs.cuda()
#     labels = labels.cuda()
#     # wrap them in Variable
#     inputs, labels = Variable(inputs), Variable(labels)  # 转换数据格式用Variable
#
#     # optimizer.zero_grad()  # 梯度置零，因为反向传播过程中梯度会累加上一次循环的梯度
#
#     # forward + backward + optimize
#     outputs = Net(inputs.type(Tensor))
#     print(labels)
#     print(inputs)
#     print(outputs)# 把数据输进CNN网络net

image = Variable(image.cuda().float())
#print(image)
out=Net(image)
print(out)
_,predicted=torch.max(out.data,1)
print(classes[predicted])
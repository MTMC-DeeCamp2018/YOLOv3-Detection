#encoding=utf-8
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch import optim

#model:

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(3,32,11)
        self.conv2=nn.Conv2d(32,80,8)
        self.fc1=nn.Linear(80*10*26,50)
        self.fc2=nn.Linear(50,10)
        self.fc3=nn.Linear(10,2)

    def forward(self,x):
        x=F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x=F.max_pool2d(F.relu(self.conv2(x)),2)
        x=x.view(x.size()[0],-1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x




def loadtraindata():
    path = r"/home/shushi16/wyz/PyTorch-YOLOv3/post_train"                                         # 路径
    trainset = torchvision.datasets.ImageFolder(path,
                                                transform=transforms.Compose([
                                                    transforms.Resize((128, 64)),  # 将图片缩放到指定大小（h,w）或者保持长宽比并缩放最短的边到int大小
                                                    transforms.ToTensor()])
                                                )
    print(trainset.class_to_idx)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                              shuffle=True, num_workers=2)
    return trainloader


def trainandsave():
    trainloader = loadtraindata()
    # 神经网络结构
    net = Net()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)  # 学习率为0.001
    criterion = nn.CrossEntropyLoss()# 损失函数也可以自己定义，我们这里用的交叉熵损失函数

    net.cuda()


    # 训练部分
    for epoch in range(100):  # 训练的数据量为5个epoch，每个epoch为一个循环
        # 每个epoch要训练所有的图片，每训练完成200张便打印一下训练的效果（loss值）
        running_loss = 0.0  # 定义一个变量方便我们对loss进行输出
        for i, data in enumerate(trainloader, 0):  # 这里我们遇到了第一步中出现的trailoader，代码传入数据
            # enumerate是python的内置函数，既获得索引也获得数据
            # get the inputs
            inputs, labels = data  # data是从enumerate返回的data，包含数据和标签信息，分别赋值给inputs和labels
            # print(inputs)
            # print(labels)
            inputs = inputs.cuda()
            labels = labels.cuda()
            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)  # 转换数据格式用Variable

            optimizer.zero_grad()  # 梯度置零，因为反向传播过程中梯度会累加上一次循环的梯度

            # forward + backward + optimize
            outputs = net(inputs)  # 把数据输进CNN网络net
            loss = criterion(outputs, labels)  # 计算损失值
            loss.backward()  # loss反向传播
            optimizer.step()  # 反向传播后参数更新
            running_loss += loss.data[0]  # loss累加

            if i % 200 == 199:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))  # 然后再除以200，就得到这两百次的平均损失值
                running_loss = 0.0  # 这一个200次结束后，就把running_loss归零，下一个200次继续使用

        torch.save(net, 'post_cnn_weights2/net%d.pkl'%epoch)  # 保存整个神经网络的结构和模型参数
        torch.save(net.state_dict(), 'post_cnn_weights2/net_params%d.pkl'%epoch)

    print('Finished Training')
    # 保存神经网络


if __name__=='__main__':
    loadtraindata()
    trainandsave()

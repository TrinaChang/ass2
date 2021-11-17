#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables train_val_split,
batch_size as well as the transform function.
You are encouraged to modify these to improve the performance of your model.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

"""
   Answer to Question:

Briefly describe how your program works, and explain any design and training
decisions you made along the way.
"""

############################################################################
######     Specify transform(s) to be applied to the input images     ######
############################################################################
def transform(mode):
    """
    Called when loading the data. Visit this URL for more information:
    https://pytorch.org/vision/stable/transforms.html
    You may specify different transforms for training and testing
    """
    if mode == 'train':
        trainSet = transforms.Compose([
            transforms.Resize(100),
            transforms.RandomCrop(80),
            transforms.ColorJitter(brightness=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.05),
            transforms.ToTensor()
        ])
        # return transforms.ToTensor()
        return trainSet
    elif mode == 'test':
        testSet = transforms.Compose([
            transforms.Resize(100),
            transforms.RandomCrop(80),
            transforms.ColorJitter(brightness=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.05),
            transforms.ToTensor()
        ])
        # return transforms.ToTensor()
        return testSet

class Residual(nn.Module):
    def __init__(self, input, out):
        super(Residual, self).__init__()
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(input, out, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out)

        self.conv2 = nn.Conv2d(out, out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out)

    def forward(self, out):
        out = self.relu(self.bn1(self.conv1(out)))
        out = self.relu(self.bn2(self.conv2(out)))
        return out


############################################################################
######   Define the Module to process the images and produce labels   ######
############################################################################
class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2x =  nn.Sequential(Residual(64, 64), Residual(64, 128))
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3x =  nn.Sequential(Residual(128, 128), Residual(128, 256))
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4x =  nn.Sequential(Residual(256, 256), Residual(256, 256))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d(p=0.5, inplace=True)
        self.fc1 = nn.Linear(256, 8)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.mp(out)
        out = self.conv2x(out)
        out = self.bn2(out)
        out = self.conv3x(out)
        out = self.bn3(out)
        out = self.conv4x(out)
        out = self.avgpool(out)
        out = self.dropout(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return F.log_softmax(out, dim=1)

    # def __init__(self):
    #     super().__init__()
        
    # def forward(self, input):
    #     pass

net = Network()
    
############################################################################
######      Specify the optimizer and loss function                   ######
############################################################################
optimizer = optim.SGD(net.parameters(), lr=0.005, weight_decay=0.0001, momentum=0)

loss_func = nn.CrossEntropyLoss()


############################################################################
######  Custom weight initialization and lr scheduling are optional   ######
############################################################################

# Normally, the default weight initialization and fixed learing rate
# should work fine. But, we have made it possible for you to define
# your own custom weight initialization and lr scheduler, if you wish.
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
    return

scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

############################################################################
#######              Metaparameters and training options              ######
############################################################################
dataset = "./data"
train_val_split = 0.8
batch_size = 64
epochs = 200

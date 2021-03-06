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


Please refer to hw2.PDF attached for model design and training considerations

"""

############################################################################
######     Specify transform(s) to be applied to the input images     ######
############################################################################

def transform(mode):
    """
    Apply the same set of transforms on training and validation data
    """
    # transform on the training set
    if mode == 'train':
        trainSet = transforms.Compose([
            transforms.Resize(100),
            transforms.RandomCrop(80),
            transforms.ColorJitter(brightness=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.05),
            transforms.RandomRotation(degrees=60),
            transforms.ToTensor()
        ])
        return trainSet
    # transform on the testing set
    elif mode == 'test':
        testSet = transforms.Compose([
            transforms.Resize(100),
            transforms.RandomCrop(80),
            transforms.ColorJitter(brightness=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.05),
            transforms.RandomRotation(degrees=60),
            transforms.ToTensor()
        ])
        return testSet


############################################################################
######   Define the Module to process the images and produce labels   ######
############################################################################
"""
For the comments in following sections
"paper" refers to the paper Deep Residual Learning for Image Recognition
Reference: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
"""
class ResidualBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()

        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        # ====== layers ======
        self.conv1 = nn.Conv2d(inplanes, planes, stride=stride, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(planes, planes, stride=1, kernel_size=3, padding=1)

        self.downsample = downsample
        
    def forward(self, x):
        # record the current input
        identity = x

        # The first layer
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        # The second layer
        x = self.conv2(x)  

        # apply skip connection, add identity
        if self.downsample is not None:
            identity = self.downsample(identity)      
        x += identity
        x = self.relu(x)

        return x

class Network(nn.Module):

    def __init__(self):
        super().__init__()

        self.inplanes = 64
        self.bn = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU()

        # ====== pooling layer ======
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=3, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # ====== dropout ======
        self.dropout1 = nn.Dropout2d(p=0.1)
        self.dropout2 = nn.Dropout2d(p=0.15)
        self.dropout3 = nn.Dropout2d(p=0.2)
        self.dropout4 = nn.Dropout2d(p=0.25)

        # ====== layers ======
        # layer conv1 in paper
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        # layer conv2_x in the paper
        self.l1 = self._make_layer(64, 2)
        # layer conv3_x in the paper
        self.l2 = self._make_layer(128, 2, 2)
        # layer conv4_x in the paper    
        self.l3 = self._make_layer(256, 2, 2)
        # layer conv5_x in the paper
        self.l4 = self._make_layer(512, 2, 2)
        self.fc = nn.Linear(512, 8)
    
    def _make_layer(self, planes, blocks, stride=1):
        """ construct stacked blocks """
        layers = []

        # Construct downsampling layer
        downsample = self._make_downsample(planes, stride)

        # stack the residual block with downsample
        layers.append(
            ResidualBlock(self.inplanes, planes, stride=stride, downsample=downsample)
        )

        self.inplanes = planes

        # stack the residual blocks
        for _ in range(1, blocks):
            layers.append(ResidualBlock(planes, planes))

        # return the layer
        return nn.Sequential(*layers)
        
    def _make_downsample(self, planes, stride):
        """ construct downsampling layers """
        # downsample is not needed for the first layer
        if stride == 1 and self.inplanes == planes:
            return None

        # reuturn the downsample
        return nn.Sequential(
            nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes),
        )
    
    def forward(self, input):
        # convolutional layer
        x = self.conv1(input)

        # apply batch normalization/relu/dropout/pooling after the layers
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.maxpool1(x)

        # residual block layers
        x = self.l1(x)
        x = self.l2(x)

        # apply batch normalization/relu/dropout/pooling after the layers
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.maxpool1(x)

        # residual block layers
        x = self.l3(x)
        x = self.l4(x)

        # apply batch normalization/relu/dropout/pooling after the layers
        x = self.relu(x)
        # apply larger dropout rate for layers with more hidden nodes
        if (self.inplanes < 256):
            x = self.dropout3(x)
        else:
            x = self.dropout4(x)
        x = self.avgpool(x)

        # flatten the result and send it to the fully connected layer
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

net = Network()
    
############################################################################
######      Specify the optimizer and loss function                   ######
############################################################################
optimizer = optim.AdamW(net.parameters(), lr=1e-3)

loss_func = nn.CrossEntropyLoss()


############################################################################
######  Custom weight initialization and lr scheduling are optional   ######
############################################################################

# Normally, the default weight initialization and fixed learing rate
# should work fine. But, we have made it possible for you to define
# your own custom weight initialization and lr scheduler, if you wish.

def weights_init(m):
    """
    use the same weight initialization as ResNet
    xavier initialisation can also achieve a similar accuracy as kaiming_normal
    """
    # different initialization for convolutional layer and batch normalization layer
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    return

# learning rate decay with a multiplicator of 0.1 during each milestone interval
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150, 200, 250, 300, 350, 400, 450], gamma=0.1)

############################################################################
#######              Metaparameters and training options              ######
############################################################################
dataset = "./data"
train_val_split = 1
batch_size = 64
epochs = 300
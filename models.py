import torch
import torch.nn as nn
import torchvision
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F

# modelyolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

class BaselineCNN(nn.Module):
    def __init__(self, img_dim=640, backbone=None,num_classes=26, num_kernels=3, input_filter=3, num_filters1=128, num_filters2=64, num_hidden=512, num_hidden2=256, pooling_dim=2, stride=1, padding=1, stridepool=2, paddingpool=0):
        super().__init__()
        flattendim = int((img_dim/stridepool) ** 2) * num_filters2
        
        self.backbone = None
        self.conv1 = nn.Conv2d(input_filter, num_filters1, kernel_size=num_kernels, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(num_filters1, num_filters2, kernel_size=num_kernels, stride=stride, padding=padding)
        self.maxpool = nn.MaxPool2d(kernel_size=pooling_dim, stride=stridepool, padding=paddingpool)
        self.dropout1 = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flattendim, num_hidden)
        self.dropout2 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(num_hidden, num_hidden2)
        self.dropout3 = nn.Dropout(0.25)
        self.classifier = nn.Linear(num_hidden2,num_classes)
        self.softmax = nn.Softmax(1)
        
        # if(backbone):
        #     self.backbone = nn.Sequential(*backbone)
            
    def forward(self, x):
        self.out = self.conv1(x)
        self.out = self.conv2(self.out)
        self.out = self.dropout1(self.maxpool(self.out))
        self.out = self.flatten(self.out)
        self.out = self.dropout2(self.fc1(self.out))
        self.out = self.dropout3(self.fc2(self.out))
        self.out = self.classifier(self.out)
        self.out = self.softmax(self.out)
        
        return self.out
            

class CNN_MNIST(nn.Module):
    def __init__(self):
        super(CNN_MNIST, self).__init__()
        # Input shape (batch_size, 1, 128, 128)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 16 * 16, 1000)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1000, 500)
        self.out = nn.Linear(500, 24) # 24 classes (A to Y without J and Z)

    def forward(self, x):
        # Layer 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Layer 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # Layer 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        # Apply dropout after pooling
        x = self.dropout1(x)
        # Flatten
        x = x.view(-1, 128 * 16 * 16)
        # Dense
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

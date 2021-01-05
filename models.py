## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, 5) 
        self.conv2 = nn.Conv2d(32, 64, 3, padding = 1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding = 1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding = 1)
        self.conv5 = nn.Conv2d(256, 512, 3, padding = 1)
        self.conv6 = nn.Conv2d(512, 1024, 3, padding = 1)
        
        self.batchnorm_1 = nn.BatchNorm2d(64)
        self.batchnorm_2 = nn.BatchNorm2d(256)
        self.batchnorm_3 = nn.BatchNorm2d(1024)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(1024 * 3 * 3, 1000)
        self.fc2 = nn.Linear(1000, 136)
        self.dropout = nn.Dropout(p = 0.4)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        
        x = self.relu(self.conv1(x)) 
        x = self.pool(x) 
        x = self.batchnorm_1(self.relu(self.conv2(x)))
        x = self.pool(x) 
        x = self.relu(self.conv3(x))
        x = self.pool(x) 
        x = self.batchnorm_2(self.relu(self.conv4(x)))
        x = self.pool(x) 
        x = self.relu(self.conv5(x))
        x = self.pool(x)
        x = self.batchnorm_3(self.relu(self.conv6(x)))
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

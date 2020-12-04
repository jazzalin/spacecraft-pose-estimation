# Direct pose regression network
import numpy as np
import json
import os
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torchvision.models as models


# Pre-trained layers
resnet = models.resnet34(pretrained=True)

class MangoNet(nn.Module):
    def __init__(self, criterion=None):
        super(MangoNet, self).__init__()

        # Resnet backbone
        self.resnet = nn.Sequential(*list(resnet.children())[:-2]) 
        self.conv6 = nn.Conv2d(512, 1024, 3, 2)
        self.bn1 = nn.BatchNorm2d(1024)
        
        # Translation branch: t = [tx, ty, tz]
#         self.fc_t1 = nn.Linear(3*3*1024, 1024)
#         self.fc_t2 = nn.Linear(1024, 3)
        self.t_branch = nn.Sequential(nn.Linear(3*3*1024, 1024),
                                      nn.ReLU(inplace=True),
                                      nn.Dropout(),
                                      nn.Linear(1024, 3))



        # Attitude branch: q = [q0, q1, q2, q3]
#         self.fc_att1 = nn.Linear(3*3*1024, 1024)
#         self.fc_att2 = nn.Linear(1024, 4) # unit-quaternions
#         #self.fc_att2 = nn.Linear(1024, 3) # MRP
        self.att_branch = nn.Sequential(nn.Linear(3*3*1024, 1024),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(),
                                        nn.Linear(1024, 4),
                                        nn.Tanh())
    
    def forward(self, input, gt=None):
        # Input size: 256x256
        x = self.resnet(input)
        x = F.relu(self.bn1(self.conv6(x)))
        
        # Reshape after bottleneck
        x_t = x.view(-1, 3*3*1024)
        x_att = x.view(-1, 3*3*1024)

        # T regression
#         x_t = F.relu(self.fc_t1(x_t))
#         x_t = self.fc_t2(x_t)
        x_t = self.t_branch(x_t)

        # q regression
#         x_att = F.relu(self.fc_att1(x_att))
#         x_att = torch.tanh(self.fc_att2(x_att))
        x_att = self.att_branch(x_att)


        return x_t, x_att
        # return x_att
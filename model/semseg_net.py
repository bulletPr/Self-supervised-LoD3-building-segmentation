#
#
#      0=================================0
#      |    Project Name                 |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Implements: Semantic Segmentation Network
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      YUWEI CAO - 2020/11/19 20:42 PM 
#
#


# ----------------------------------------
# import packages
# ----------------------------------------

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F


# ----------------------------------------
# Semantic Segmentation Net Function
# ----------------------------------------

class SemSegNet(nn.Module):
    def __init__(self, num_class, encoder, symmetric_function=1, dropout=False):
        super(SemSegNet, self).__init__()
        self.num_class = num_class
        self.dropout = dropout
        if symmetric_function==1: #512
            self.conv1 = torch.nn.Conv1d(704, 512, 1)
        else:
            self.conv1 = torch.nn.Conv1d(1216, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.num_class, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.dp1 = nn.Dropout(p=0.5)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]

        x = F.leaky_relu(self.bn1(self.conv1(x)),negative_slope=0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)),negative_slope=0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)),negative_slope=0.2)
        if self.dropout:
            x = self.dp1(x)
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.num_class), dim=-1)
        x = x.view(batchsize, n_pts, self.num_class)
        return x
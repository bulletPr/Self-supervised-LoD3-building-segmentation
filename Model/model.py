#
#
#      0=================================0
#      |    Project Name                 |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Implements: knn, graph filter, Foldnet/DGCNN/PointNet Encoder, Foldnet Decoder, DGCNN/PointNet Classifer
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      YUWEI CAO - 2020/10/22 9:29 AM
#
#


# ----------------------------------------
# import packages
# ----------------------------------------

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import itertools
from loss import ChamferLoss, CrossEntropyLoss, ChamferLoss_m


# ----------------------------------------
# KNN
# ----------------------------------------

def knn(x, k):
    batch_size = x.size(0)
    num_points = x.size(2)

    inner = -2*torch.matmul(x.transpose(2,1),x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2,1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1] #(batch_size, num_points, k)

    if idx.get_device() == -1:
        idx_base = torch.arange(0, batch_size).view(-1,1,1)*num_points
    else:
        idx_base = torch.arange(0,batch_size,device=idx.get_device()).view(-1,1,1)*num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    return idx


# ----------------------------------------
# Local Convolution
# ----------------------------------------

def local_cov(pts, idx):
    batch_size = pts.size(0)
    num_points = pts.size(2)
    pts = pts.view(batch_size, -1, num_points)              # (batch_size, 6, num_points)

    _, num_dims, _ = pts.size()

    x = pts.transpose(2, 1).contiguous()                    # (batch_size, num_points, 6)
    x = x.view(batch_size*num_points, -1)[idx, :]           # (batch_size*num_points*2, 6)
    x = x.view(batch_size, num_points, -1, num_dims)        # (batch_size, num_points, k, 6)

    #x = torch.matmul(x[:,:,0].unsqueeze(3), x[:,:,1].unsqueeze(2))  # (batch_size, num_points, 6, 1) * (batch_size,
                                                                    # num_points, 1, 6) -> (batch_size, num_points, 6, 6)
    x = torch.matmul(x[:,:,1:].transpose(3, 2), x[:,:,1:])
    x = x.view(batch_size, num_points, 9).transpose(2, 1)   # (batch_size, 9, num_points)

    x = torch.cat((pts, x), dim=1)                          # (batch_size, 12, num_points)

    return x


# ----------------------------------------
# Local Maxpool
# ----------------------------------------

def local_maxpool(x, idx):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()                      # (batch_size, num_points, num_dims)
    x = x.view(batch_size*num_points, -1)[idx, :]           # (batch_size*n, num_dims) -> (batch_size*n*k, num_dims)
    x = x.view(batch_size, num_points, -1, num_dims)        # (batch_size, num_points, k, num_dims)
    x, _ = torch.max(x, dim=2)                              # (batch_size, num_points, num_dims)

    return x


# ----------------------------------------
# Local Maxpool
# ----------------------------------------

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)      # (batch_size, num_dims, num_points)
    if idx is None:
        idx = knn(x, k=k)                       # (batch_size, num_points, k)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()          # (batch_size, num_points, num_dims)
    feature = x.view(batch_size*num_points, -1)[idx, :]                 # (batch_size*n, num_dims) -> (batch_size*n*k, num_dims)
    feature = feature.view(batch_size, num_points, k, num_dims)         # (batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  # (batch_size, num_points, k, num_dims)

    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2)      # (batch_size, num_points, k, 2*num_dims) -> (batch_size, 2*num_dims, num_points, k)

    return feature                              # (batch_size, 2*num_dims, num_points, k)


# ----------------------------------------
# FoldingNet_Encoder
# ----------------------------------------

class FoldNet_Encoder(nn.Module):
    def __init__(self, args):
        super(FoldNet_Encoder, self).__init__()
        if args.k == None:
            self.k = 16
        else:
            self.k = args.k
        self.n = 2048 # input point cloud size
        self.mlp1 = nn.Sequential(
                nn.Conv1d(12,64,1),
                nn.ReLU(),
                nn.Conv1d(64,64,1),
                nn.ReLU(),
                nn.Conv1d(64,64,1),
                nn.ReLU(),
        )

        self.linear1 = nn.Linear(64,64)
        self.conv1 = nn.Conv1d(64,128,1)
        self.linear2 = nn.Linear(128,128)
        self.conv2 = nn.Conv1d(128,1024,1)
        self.mlp2 = nn.Sequential(
               nn.Conv1d(1024, args.feat_dims, 1),
               nn.ReLU(),
               nn.Conv1d(args.feat_dims, args.feat_dims, 1),
        )

    def graph_layer(self, x, idx):
        x = local_maxpool(x, idx)
        x = self.linear1(x)
        x = x.transpose(2,1)
        x = F.relu(self.conv1(x))
        x = local_maxpool(x, idx)
        x = self.linear2(x)
        x = x.transpose(2,1)
        x = self.conv2(x)
        return x

    def forward(self, pts):
        pts = pts.transpose(2,1) #(batch_size, 3, num_points)
        idx = knn(pts, k=self.k)
        x = local_cov(pts, idx) #(batch_size, 3, num_points) -> (batch_size, 12, num_points)
        x0 = self.mlp1(x) #(batch_size, 64, num_points)
        x = self.graph_layer(x0, idx) #(batch_size,1024, num_points)
        x = torch.max(x, 2, keepdim=True)[0] #(batch_size,1024,1)
        x = self.mlp2(x)                     #(batch_size, feat_dims,1)
        feat = x.transpose(2,1)              #(batch_size,1,feat_dims)

        return feat, x0                      #(batch_size,1,feat_dims)  (batch_size, 64, num_points)


# ----------------------------------------
# FoldingNet_Decoder
# ----------------------------------------

class FoldNet_Decoder(nn.Module):
    def __init__(self, args):
        super(FoldNet_Decoder, self).__init__()
        if args.num_points == 2048:
            self.m = 2025
            self.meshgrid=[[-0.3,0.3,45], [-0.3,0.3,45]]
        elif args.num_points == 4096:
            self.m = 4096
            self.meshgrid=[[-0.3,0.3,64], [-0.3,0.3,64]]
        self.folding1 = nn.Sequential(
                nn.Conv1d(args.feat_dims+2, args.feat_dims, 1),
                nn.ReLU(),
                nn.Conv1d(args.feat_dims, args.feat_dims, 1),
                nn.ReLU(),
                nn.Conv1d(args.feat_dims, 3, 1),
        )

        self.folding2 = nn.Sequential(
                nn.Conv1d(args.feat_dims+3, args.feat_dims, 1),
                nn.ReLU(),
                nn.Conv1d(args.feat_dims, args.feat_dims,1),
                nn.ReLU(),
                nn.Conv1d(args.feat_dims,3,1),
        )


    def build_grid(self, batch_size):
        x = np.linspace(*self.meshgrid[0])
        y = np.linspace(*self.meshgrid[1])
        grid = np.array(list(itertools.product(x, y)))
        grid = np.repeat(grid[np.newaxis, ...], repeats=batch_size, axis=0)
        grid = torch.tensor(grid)
        return grid.float()


    def forward(self, x):
        x = x.transpose(1,2).repeat(1,1,self.m) #(batch_size,feat_dims,num_points)
        grid = self.build_grid(x.shape[0]).transpose(1,2) #(bs, 2, feat_dims)
        if x.get_device() != -1:
            grid = grid.cuda(x.get_device())
        concate1 = torch.cat((x, grid),dim=1) #(bs, feat_dims+2, num_points)
        after_fold1 = self.folding1(concate1) #(bs,3,num_points)
        concate2 = torch.cat((x, after_fold1), dim=1) #(bs, feat_dims+3, num_points)
        after_fold2 = self.folding2(concate2) #(bs, 3, num_points)
        return after_fold2.transpose(1,2)  #(bs, num_points, 3)


# ----------------------------------------
# DGCNN_CLASSIFICATION_Encoder
# ----------------------------------------

class DGCNN_Cls_Encoder(nn.Module):
    def __init__(self, args):
        super(DGCNN_Cls_Encoder, self).__init__()
        if args.k == None:
            self.k = 20
        else:
            self.k = args.k
        self.task = args.task

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.feat_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(3*2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.feat_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        x = x.transpose(2, 1)

        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_graph_feature(x3, k=self.k)     # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(x)                       # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 512, num_points)

        x0 = self.conv5(x)                      # (batch_size, 512, num_points) -> (batch_size, feat_dims, num_points)
        x = x0.max(dim=-1, keepdim=False)[0]    # (batch_size, feat_dims, num_points) -> (batch_size, feat_dims)
        feat = x.unsqueeze(1)                   # (batch_size, feat_dims) -> (batch_size, 1, feat_dims)

        if self.task == 'classify':
            return feat, x0
        elif self.task == 'reconstruct':
            return feat                         # (batch_size, 1, feat_dims)


# ----------------------------------------
# Point_transform_mini_network
# ----------------------------------------

class Point_Transform_Net(nn.Module):
    def __init__(self):
        super(Point_Transform_Net, self).__init__()

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(3*2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, 3*3)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)                       # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 1024, num_points) -> (batch_size, 1024)

        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)     # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)     # (batch_size, 512) -> (batch_size, 256)

        x = self.transform(x)                   # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, 3, 3)            # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x                                # (batch_size, 3, 3)


# ----------------------------------------
# DGCNN_SEGMENTATION_ENCODER
# ----------------------------------------

class DGCNN_Seg_Encoder(nn.Module):
    def __init__(self, args):
        super(DGCNN_Seg_Encoder, self).__init__()
        if args.k == None:
            self.k = 20
        else:
            self.k = args.k
        self.transform_net = Point_Transform_Net()

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(args.feat_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, args.feat_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        x = x.transpose(2, 1)

        batch_size = x.size(0)
        num_points = x.size(2)

        x0 = get_graph_feature(x, k=self.k)     # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        t = self.transform_net(x0)              # (batch_size, 3, 3)
        x = x.transpose(2, 1)                   # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        x = torch.bmm(x, t)                     # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        x = x.transpose(2, 1)                   # (batch_size, num_points, 3) -> (batch_size, 3, num_points)

        x = get_graph_feature(x, k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x4 = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)

        x = self.conv6(x4)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)

        feat = x.unsqueeze(1)                   # (batch_size, num_points) -> (batch_size, 1, emb_dims)

        return feat, x4                             # (batch_size, 1, emb_dims)


# ----------------------------------------
# DGCNN_CLASSIFICATION_Classifier_Net
# ----------------------------------------

class DGCNN_Cls_Classifier(nn.Module):
    def __init__(self, args):
        super(DGCNN_Cls_Classifier, self).__init__()
        if args.dataset == 'arch':
            output_channels = 10
        elif args.dataset == 'shapenetcorev2':
            output_channels = 55
        elif args.dataset == 'shapenetpart':
            output_channels = 16
        elif args.dataset == 'modelnet40':
            output_channels = 40

        self.linear1 = nn.Linear(args.feat_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x


# -----------------------------------------------
# PointNet Encoder
# -----------------------------------------------

class PointNetfeat(nn.Module):
    def __init__(self, global_feat=True):
        super(PointNetfeat, self).__init__()
        self.stn = Point_Transform_Net()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))  # x = batch,1024,n(n=2048)
        x = torch.max(x, 2, keepdim=True)[0]  # x = batch,1024,1
        x = x.view(-1, 1024)  # x = batch,1024
        if self.global_feat:
            return x, trans
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans


class PointNetEncoder(nn.Module):
    def __init__(self):
        super(PointNetEncoder, self).__init__()
        self.feat = PointNetfeat(global_feat=True)
        self.fc1 = nn.Linear(1024, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans = self.feat(x)  # x = batch,1024
        x = F.relu(self.bn1(self.fc1(x)))  # x = batch,1024
        x = self.fc2(x)  # x = batch,1024

        return x, trans


class PointNetCls(nn.Module):
    def __init__(self, output_channels=2):
        super(PointNetCls, self).__init__()
        if args.dataset == 'arch':
            output_channels = 10
        elif args.dataset == 'shapenetcorev2':
            output_channels = 55
        elif args.dataset == 'shapenetpart':
            output_channels = 16
        elif args.dataset == 'modelnet40':
            output_channels = 40

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x


# ----------------------------------------
# Reconstrucion Network
# ----------------------------------------

class DGCNN_FoldNet(nn.Module):
    def __init__(self, args):
        super(DGCNN_FoldNet, self).__init__()
        if args.encoder == 'foldingnet':
            self.encoder = FoldNet_Encoder(args)
        elif args.encoder == 'dgcnn_classification':
            self.encoder = DGCNN_Cls_Encoder(args)
        elif args.encoder == 'dgcnn_segmentation':
            self.encoder = DGCNN_Seg_Encoder(args)
        elif args.encoder == 'pointnet':
            self.encoder = PointNetEncoder(args)
        self.decoder = FoldNet_Decoder(args)
        if args.loss == 'ChamferLoss':
            self.loss = ChamferLoss()
        elif args.loss == 'ChamferLoss_m':
            self.loss = ChamferLoss_m()

    def forward(self, input):
        feature, mid_fea = self.encoder(input)
        output = self.decoder(feature)
        return output, feature, mid_fea

    def get_parameter(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def get_loss(self, input, output):
        #input (bs, 2048, 3)
        #output (bs, 2025,3)
        return self.loss(input, output)


# ----------------------------------------
# Classification Network
# ----------------------------------------

class ClassificationNet(nn.Module):
    def __init__(self, args):
        super(ClassificationNet, self).__init__()
        self.is_eval = args.eval
        if args.encoder == 'foldnet':
            self.encoder = FoldNet_Encoder(args)
        elif args.encoder == 'dgcnn_cls':
            self.encoder = DGCNN_Cls_Encoder(args)
        elif args.encoder == 'dgcnn_seg':
            self.encoder = DGCNN_Seg_Encoder(args)
        elif args.encoder == 'pointnet_cls':
            self.encoder == PointNetEncoder(args)
        if not self.is_eval:
            self.classifier = DGCNN_Cls_Classifier(args)
        self.loss = CrossEntropyLoss()

    def forward(self, input):
        feature, latent = self.encoder(input)
        if not self.is_eval:
            output = self.classifier(latent)
            return output, feature
        else:
            return feature

# ----------------------------------------
# Classification Network
# ----------------------------------------

class PointNetSegmentation(nn.Module):
    def __init__(self, k=2):
        super(PointNetDenseCls, self).__init__()
        if args.dataset == 'arch':
            self.k = 10
        elif args.dataset == 'shapenetcorev2':
            self.k = 55
        elif args.dataset == 'shapenetpart':
            self.k = 16
        elif args.dataset == 'modelnet40':
            self.k = 40
        self.feat = PointNetfeat(global_feat=False)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2, 1).contiguous()
        x = F.log_softmax(x.view(-1, self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x, trans

    def get_parameter(self):
        return list(self.encoder.parameters()) + list(self.classifier.parameters())

    def get_loss(self, input, output):
        # input shape  (batch_size, 2048)
        # output shape (batch_size, 2025)
        return self.loss(input, output)
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
    num_dims = x.size(1)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)      # (batch_size, num_dims, num_points)
    if idx is None:
        if num_dims == 9:
            idx = knn(x[:,6:9], k=k)
        else:
            idx = knn(x, k=k)
    if idx.get_device() == -1:
        idx_base = torch.arange(0, batch_size).view(-1,1,1)*num_points
    else:
        idx_base = torch.arange(0,batch_size,device=idx.get_device()).view(-1,1,1)*num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    x = x.transpose(2, 1).contiguous()          # (batch_size, num_points, num_dims)
    feature = x.view(batch_size*num_points, -1)[idx, :]                 # (batch_size*n, num_dims) -> (batch_size*n*k, num_dims)
    feature = feature.view(batch_size, num_points, k, num_dims)         # (batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  # (batch_size, num_points, k, num_dims)

    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2)      # (batch_size, num_points, k, 2*num_dims) -> (batch_size, 2*num_dims, num_points, k)

    return feature                              # (batch_size, 2*num_dims, num_points, k)


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
        #self.transform_net = Point_Transform_Net()
        self.symmetric_function = args.symmetric_function
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(args.feat_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(args.num_dims*2, 64, kernel_size=1, bias=False),
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
        x = x.transpose(2, 1)  #(batch_size, num_points, num_dims) -> (batch_size, num_dims, num_points)

        batch_size = x.size(0)

        #x = get_graph_feature(x, k=self.k)     # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        #x = x.transpose(2, 1)                   # (batch_size, 9*2, num_points, k) -> (batch_size, num_points, 9,k)
        #x = torch.bmm(x, t)                     # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        #x = x.transpose(2, 1)                   # (batch_size, num_points, 3) -> (batch_size, 3, num_points)

        x = get_graph_feature(x, k=self.k)      # (batch_size, num_dims, num_points) -> (batch_size, num_dims*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, num_dims*2, num_points, k) -> (batch_size, 64, num_points, k)
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
        if self.symmetric_function == 2:
            x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
            x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
            x = torch.cat((x1, x2), 1)           #(batch_size, emb_dims*2)
        else:
            x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        #feat = x.unsqueeze(1)                   # (batch_size, emb_dims) -> (batch_size, 1, emb_dims) or (batch_size, 1, emb_dims*2)

        return x, x4                             # (batch_size, emb_dims*2), (batch_size, 64*3, num_points)



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
                nn.Conv1d(args.feat_dims*args.symmetric_function+2, args.feat_dims, 1),
                nn.ReLU(),
                nn.Conv1d(args.feat_dims, args.feat_dims, 1),
                nn.ReLU(),
                nn.Conv1d(args.feat_dims, 3, 1),
        )

        self.folding2 = nn.Sequential(
                nn.Conv1d(args.feat_dims*args.symmetric_function+3, args.feat_dims, 1),
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
        x = x.unsqueeze(1).transpose(1,2).repeat(1,1,self.m) #(batch_size,feat_dims,num_points)
        grid = self.build_grid(x.shape[0]).transpose(1,2) #(bs, 2, feat_dims)
        if x.get_device() != -1:
            grid = grid.cuda(x.get_device())
        concate1 = torch.cat((x, grid),dim=1) #(bs, feat_dims+2, num_points)
        after_fold1 = self.folding1(concate1) #(bs,3,num_points)
        concate2 = torch.cat((x, after_fold1), dim=1) #(bs, feat_dims+3, num_points)
        after_fold2 = self.folding2(concate2) #(bs, 3, num_points)
        return after_fold2.transpose(1,2)  #(bs, num_points, 3)


# ----------------------------------------
# DGCNN_Segmenter
# ----------------------------------------
class SemSegNet(nn.Module):
    def __init__(self, args):
        super(SemSegNet, self).__init__()
        self.num_class = args.num_class
        self.dropout = args.dropout
        if args.symmetric_function == 1: #512
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

# ----------------------------------------
# DGCNN_ROTAION_CLASSIFICATION_DECODER
# ----------------------------------------
class DGCNN_RT_Cls_Classifier(nn.Module):
    def __init__(self, args):
        super(DGCNN_RT_Cls_Classifier, self).__init__()
        self.output_channels = args.num_angles
        self.linear1 = nn.Linear(args.feat_dims*args.symmetric_function, 512, bias=False)  #(batch_size, 512)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256) #(batch_size, 216)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, self.output_channels) #(batch_size, args.num_angles)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        x = x.contiguous()
        x = F.log_softmax(x.view(-1,self.output_channels), dim=-1)
        x = x.view(batch_size, self.output_channels)
        return x


# ----------------------------------------
# Classification Network
# ----------------------------------------

class ClassificationNet(nn.Module):
    def __init__(self, args):
        super(ClassificationNet, self).__init__()
        self.encoder = DGCNN_Seg_Encoder(args)
        self.classifier = DGCNN_RT_Cls_Classifier(args)
        self.loss = CrossEntropyLoss()

    def forward(self, input):
        feats, _ = self.encoder(input)
        output = self.classifier(feats)
        return output, feats

    def get_loss(self, pred, label):
        # loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=pred, label_smoothing=0.2)
        return self.loss(input, output)


# ----------------------------------------
# Reconstrucion Network
# ----------------------------------------

class ReconstructionNet(nn.Module):
    def __init__(self, args):
        super(ReconstructionNet, self).__init__()
        self.encoder = DGCNN_Seg_Encoder(args)
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
# MultiTask Network
# ----------------------------------------

class MultiTaskNet(nn.Module):
    def __init__(self, args):
        super(MultiTaskNet, self).__init__()
        self.encoder = DGCNN_Seg_Encoder(args)
        self.decoder = FoldNet_Decoder(args)
        if args.rec_loss == 'ChamferLoss':
            self.rec_loss = ChamferLoss()
        elif args.rec_loss == 'ChamferLoss_m':
            self.rec_loss = ChamferLoss_m()
        self.classifer = DGCNN_RT_Cls_Classifier(args)
        self.rot_loss = CrossEntropyLoss()

    def forward(self, input):
        feature, mid_fea = self.encoder(input)
        rec_output = self.decoder(feature)
        rot_output = self.classifer(feature)
        return rec_output, rot_output, feature, mid_fea

    def get_parameter(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters() + self.classifer.parameters())

    def get_loss(self, input, rec_output, label, rot_output):
        #input (bs, 2048, 3)
        #output (bs, 2025,3)
        return self.rec_loss(input, rec_output) + self.rot_loss(rot_output, label)
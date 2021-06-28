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
from . import model
from . import loss


# ----------------------------------------
# Semantic Segmentation Net Function
# ----------------------------------------

class SegModel(nn.Module):
    def __init__(self, args):
        super(SegModel, self).__init__()
        self.args = args
        self.encoder = model.DGCNN_Seg_Encoder(self.args)
        self.segmenter = model.SemSegNet(self.args)
        self.old_lr_encoder = self.args.lr
        self.old_lr_segmenter = self.args.lr
        self.optimizer_encoder = torch.optim.Adam(self.encoder.parameters(),
                                                  lr=self.old_lr_encoder,
                                                  betas=(0.9, 0.999),
                                                  weight_decay=0)
        self.optimizer_segmenter = torch.optim.Adam(self.segmenter.parameters(),
                                                    lr=self.old_lr_segmenter,
                                                    betas=(0.9, 0.999),
                                                    weight_decay=0)
        
    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        _, _, latent_caps, mid_features = self.encoder(x) # (batch_size, emb_dims*2), (batch_size, 64*3, num_points)
        self.feature = torch.cat([latent_caps.view(-1,args.feat_dims*args.symmetric_function,1).repeat(1,1,args.num_points), mid_features],1).cpu().detach().numpy()
            latent_caps = torch.from_numpy(con_code).float()
        self.score_segmenter = self.segmenter(self.feature)
        return self.score_segmenter

    def update_learning_rate(self, ratio):
        # encoder
        lr_encoder = self.old_lr_encoder * ratio
        for param_group in self.optimizer_encoder.param_groups:
            param_group['lr'] = lr_encoder
        print('update encoder learning rate: %f -> %f' % (self.old_lr_encoder, lr_encoder))
        self.old_lr_encoder = lr_encoder

        # segmentation
        lr_segmenter = self.old_lr_segmenter * ratio
        for param_group in self.optimizer_segmenter.param_groups:
            param_group['lr'] = lr_segmenter
        print('update segmenter learning rate: %f -> %f' % (self.old_lr_segmenter, lr_segmenter))
        self.old_lr_segmenter = lr_segmenter

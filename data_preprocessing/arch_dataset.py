
#
#      0=================================0
#      |    Project Name                 |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Implements read ArchDatase
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      YUWEI CAO - 2020/10/2
#
#


# ----------------------------------------
# Import packages and constant
# ----------------------------------------

from torch.utils.data.dataset import Dataset
import os
import numpy as np
import os.path
import torch
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from pc_utils import translate_pointcloud, jitter_pointcloud, rotate_pointcloud
from pc_utils import load_sampled_h5_seg, grouped_shuffle


# ----------------------------------------
# Construct ArchDataset
# ----------------------------------------

class ArchDataset(Dataset):
    def __init__(self, filelist, num_points=2048, num_dims =3, split='train', random_translate=False, random_rotate=False,
            random_jitter=False, group_shuffle=False):
        self.random_translate = random_translate
        self.random_jitter = random_jitter
        self.random_rotate = random_rotate
        self.group_shuffle = group_shuffle
        self.num_points = num_points
        self.num_dims = num_dims
        self.split = split
        
        #define all train/test file
        self.path_h5py_all =[]
        
        log_string("Read datasets by load .h5 files, filelist: " + str(filelist))
        self.path_h5py_all = filelist
        
        self.data, self.seg_labels = load_sampled_h5_seg(self.path_h5py_all, self.num_dims, self.split)
        if self.group_shuffle:
            self.data, self.seg_labels = grouped_shuffle([self.data, self.seg_labels])
        log_string("size of all point_set: [" + str(self.data.shape) + "," + str(self.seg_labels.shape) + "]")
        

    def __getitem__(self, index):
        point_set = self.data[index]
        label = self.seg_labels[index]
        
        # data augument
        if self.random_translate:
            point_set[:,0:3] = translate_pointcloud(point_set[:,0:3])
        if self.random_jitter:
            point_set[:,0:3] = jitter_pointcloud(point_set[:,0:3])
        if self.random_rotate:
            point_set[:,0:3] = rotate_pointcloud(point_set[:,0:3])
            
        #conver numpy array to pytorch Tensor
        point_set = torch.from_numpy(point_set)
        #colors = torch.from_numpy()
        label = torch.from_numpy(np.array([label]).astype(np.int8))
        label = label.squeeze(0)
        return point_set, label


    def __len__(self):
        return self.data.shape[0]


LOG_FOUT = open(os.path.join(ROOT_DIR, 'LOG','dataread_log.txt'), 'w')
def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)
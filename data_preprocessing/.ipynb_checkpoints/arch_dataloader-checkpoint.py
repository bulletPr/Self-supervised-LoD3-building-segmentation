#
#
#      0=================================0
#      |    Project Name                 |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Implements
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      YUWEI CAO - 2020/10/21 5:26 PM 
#
#
import torch
from arch_dataset import ArchDataset
import os.path
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_dataloader(filelist, num_points=2048, num_dims=3, split='train', group_shuffle=False, shuffle=False, random_translate=False, random_rotate=False, random_jitter=False, batch_size=32, num_workers=4, drop_last=False):
    dataset = ArchDataset(
            filelist=filelist,
            num_points=num_points,
            num_dims=num_dims,
            split = split,
            group_shuffle=group_shuffle,
            random_translate=random_translate, 
            random_rotate=random_rotate,
            random_jitter=random_jitter)
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last)

    return dataloader
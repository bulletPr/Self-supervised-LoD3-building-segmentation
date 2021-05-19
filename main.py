#
#
#      0===================================================================0
#      |    SSL Building Point Cloud Feature Learning                 |
#      0===================================================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Implements
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      YUWEI CAO - 2021/5/19 14:07 PM 
#
#
from __future__ import print_function

import os
import sys
import argparse

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'model'))

from train_AE_Rotation import Train_AE_Rotation


def get_parser():
    parser = argparse.ArgumentParser(description='SSL Building Point Cloud Feature Learning')
    parser.add_argument('--experiment_name', type=str, default=None, metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--symmetric_function', type=int, default=2,
                        help='symmetric function')
    parser.add_argument('--task', type=str, default='multitask', metavar='N',
                        choices=['reconstruction', 'rotation_prediction','multitask'],
                        help='task to excution, [reconstruction, rotation_prediction, multitask]')
    parser.add_argument('--feat_dims', type=int, default=512, metavar='N',
                        help='Number of dims for feature ')
    parser.add_argument('--num_dims', type=int, default=3, metavar='N',
                        help='Number of dims for feature ')
    parser.add_argument('--k', type=int, default=None, metavar='N',
                        help='Num of nearest neighbors to use for KNN')
    parser.add_argument('--dataset', type=str, default='arch', metavar='N',
                        choices=['arch','all_arch','shapenetcorev2','modelnet40', 'shapenetpart'],
                        help='Encoder to use, [arch, shapenetcorev2, modelnet40, shapenetpart]')
    parser.add_argument('--split', type=str, default='train', metavar='N',
                        choices=['train','test'],
                        help='train or test')
    parser.add_argument('--use_rotate', action='store_true',
                        help='Rotate the pointcloud before training')
    parser.add_argument('--use_translate', action='store_true',
                        help='Translate the pointcloud before training')
    parser.add_argument('--use_jitter', action='store_true',
                        help='Jitter the pointcloud before training')
    parser.add_argument('--use_shuffle', action='store_true',
                        help='Shuffle the pointcloud before training')
    parser.add_argument('--rec_loss', type=str, default='ChamferLoss', choices=['ChamferLoss_m','ChamferLoss'],
                        help='reconstruction loss')
    parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--workers', type=int, help='Number of data loading workers', default=16)
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='Number of episode to train ')
    parser.add_argument('--snapshot_interval', type=int, default=10, metavar='N',
                        help='Save snapshot interval ')
    parser.add_argument('--gpu_mode', action='store_true',
                        help='Enables CUDA training')
    parser.add_argument('--eval', action='store_true',
                        help='Evaluate the model')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='Num of points to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Path to load model')
    parser.add_argument('--num_workers', type=int, default=0, metavar='N',
                        help='Number of workers to load data')
    parser.add_argument('--num_angles', type=int, default=6, metavar='N',
                        help='Number of rotation angles')
    parser.add_argument('--sample', type=str, default='combined', metavar='N',
                        choices=['combined', 's3dis','semantic3d'],
                        help='the means of sampling ponit cloud')
    parser.add_argument('--block_size', type=str, default='5m', metavar='N',
                        choices=['5m', '1.5m','2.5m'],
                        help='bloc_size of sampling ponit cloud')
    parser.add_argument('--folder', '-f', help='path to data file')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_parser()
    reconstruction = Train_AE_Rotation(args)
    reconstruction.run()
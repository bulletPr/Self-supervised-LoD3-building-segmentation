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
#      YUWEI CAO - 2020/10/28 7:00 PM 
#
#
'''Prepare Filelists for ArCH dataset Segmentation Task.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import random
import argparse
from datetime import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', '-f', help='Path to data folder')
    parser.add_argument('--h5_num', '-d', help='Number of h5 files to be loaded each time', type=int, default=4)
    parser.add_argument('--repeat_num', '-r', help='Number of repeatly using each loaded h5 list', type=int, default=2)

    args = parser.parse_args()
    print(args)

    root = args.folder if args.folder else '../data/arch/'

    splits = ['train', 'test', 'val']
    split_filelists = dict()
    for split in splits:
        split_filelists[split] = ['./%s/%s\n' % (split, filename) for filename in os.listdir(os.path.join(root, split))
                                  if filename.endswith('.h5')]

    train_h5 = split_filelists['train']
    random.shuffle(train_h5)
    train_list = os.path.join(root, 'train_data_files.txt')
    print('{}-Saving {}...'.format(datetime.now(), train_list))
    with open(train_list, 'w') as filelist:
        for filenames in train_h5:
            filelist.write(filenames)

    val_h5 = split_filelists['val']
    val_list = os.path.join(root, 'val_data_files.txt')
    print('{}-Saving {}...'.format(datetime.now(), val_list))
    with open(val_list, 'w') as filelist:
        for filename_h5 in val_h5:
            filelist.write(filename_h5)

    test_h5 = split_filelists['test']
    test_list = os.path.join(root, 'test_data_files.txt')
    print('{}-Saving {}...'.format(datetime.now(), test_list))
    with open(test_list, 'w') as filelist:
        for filename_h5 in test_h5:
            filelist.write(filename_h5)


if __name__ == '__main__':
    main()
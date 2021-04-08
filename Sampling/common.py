#
#
#      0=================================0
#      |    Project Name                 |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Implements Common Functions: scene to blocks, batch_inference
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      YUWEI CAO  -  2020/10/20 11:08 AM
#
#
# -----------------------------------------------------------------------------
# Import packages and constants
# -----------------------------------------------------------------------------

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)
import numpy as np
import argparse
import torch

DATA_PATH = os.path.join(ROOT_DIR, 'data', 'arch')
LOG_PATH = os.path.join(ROOT_DIR, 'LOG')

# -----------------------------------------------------------------------------
# LOG Function
# -----------------------------------------------------------------------------

if not os.path.exists(LOG_PATH): os.mkdir(LOG_PATH)
LOG_FOUT = open(os.path.join(LOG_PATH, 'pointnet_log_blocks.txt'), 'a')
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

# -----------------------------------------------------------------------------
# Random sample data
# -----------------------------------------------------------------------------

def sample_data(data, num_sample):
    """ data is in N x ...
        we want to keep num_samplexC of them.
        if N > num_sample, we will randomly keep num_sample of them.
        if N < num_sample, we will randomly duplicate samples.
    """
    N = data.shape[0]
    if (N == num_sample):
        return data, range(N)
    elif (N > num_sample):
        sample = np.random.choice(N, num_sample)
        return data[sample, ...], sample
    else:
        sample = np.random.choice(N, num_sample-N)
        dup_data = data[sample, ...]
        return np.concatenate([data, dup_data], 0), list(range(N))+list(sample)

def sample_data_label(data, label, num_sample):
    new_data, sample_indices = sample_data(data, num_sample)
    new_label = label[sample_indices]
    return new_data, new_label


# -----------------------------------------------------------------------------
# Split scene into blocks
# -----------------------------------------------------------------------------
def scene2blocks(data, label, num_point, block_size=1.0, stride=1.0, random_sample=False,
        sample_num=None, sample_aug=1):
    """
    Prepare block training data.
    Args:
        data: (N,9) numpy array, 012 are XYZ in meters, 345 are RGB in [0,255]
        label: N size unin8 numpy array from 0-9
        num_point: int, how many points to sample in each block
        block size: float, pysical size of the block in meters
        stride: float, stride for block sweeping
        sampling: bool, if True, we will randomly sample blocks in the room
        sampling_type: string, random, fps or grid
        sample_num: , if sample, how many blocks to sample
            [default: building area]
        sample_aug: if random sample, how much aug
    Returns:
        block_data: K x num_point x 9 np array of XYZRGBNxNyNz
        block_labels: K x num_point x 1 np array of unit8 labels
    """
    assert(stride<=block_size)

    #get the corner location for our sampling blocks
    limit = np.amax(data,0)[0:3]

    #calculate number of blocks and add into block list
    xbeg_list = []
    ybeg_list = []
    if not random_sample:
        num_block_x = int(np.ceil((limit[0] - block_size) / stride)) + 1
        num_block_y = int(np.ceil((limit[1] - block_size) / stride)) + 1
        for i in range(num_block_x):
            for j in range(num_block_y):
                xbeg_list.append(i*stride)
                ybeg_list.append(j*stride)
    else:
        num_block_x = int(np.ceil(limit[0] / block_size))
        num_block_y = int(np.ceil(limit[1] / block_size))
        if sample_num is None:
            sample_num = num_block_x * num_block_y * sample_aug
        for _ in range(sample_num):
            xbeg = np.random.uniform(-block_size, limit[0])
            ybeg = np.random.uniform(-block_size, limit[1])
            xbeg_list.append(xbeg)
            ybeg_list.append(ybeg)
    log_string("block number: " + str(num_block_x)+", " + str(num_block_y))
    #Collect blocks
    block_data_list = []
    block_label_list = []
    idx = 0
    for idx in range(len(xbeg_list)):
       xbeg = xbeg_list[idx]
       ybeg = ybeg_list[idx]
       xcond = (data[:,0]<=xbeg+block_size) & (data[:,0]>=xbeg)
       ycond = (data[:,1]<=ybeg+block_size) & (data[:,1]>=ybeg)
       cond = xcond & ycond
       if np.sum(cond) < 100: # discard block if there are less than 100 pts.
           continue

       block_data = data[cond, :]
       block_label = label[cond]

       # randomly subsample data
       block_data_sampled, block_label_sampled = \
           sample_data_label(block_data, block_label, num_point)
       block_data_list.append(np.expand_dims(block_data_sampled, 0))
       block_label_list.append(np.expand_dims(block_label_sampled, 0))
    log_string("block data list size: " + str(len(block_data_list)))

    return np.concatenate(block_data_list, 0), \
           np.concatenate(block_label_list, 0)

def scene2blocks_plus(data_label, num_point, block_size, stride,
                     random_sample, sample_num, sample_aug):
    """ room2block wit RGB preprocessing.
    """
    data = data_label[:,0:6]
    data[:,3:6] /= 255.0
    label = data_label[:,-1].astype(np.uint8)

    return scene2blocks(data, label, num_point, block_size, stride,
                       random_sample, sample_num, sample_aug)

def scene2blocks_wrapper(data_label_filename, num_point, block_size=1.0, stride=1.0,
                        random_sample=False, sample_num=None, sample_aug=1):
    data_label = np.loadtxt(data_label_filename)
    data_label = data_label[:,0:7]
    xyz_min = np.amin(data_label, axis=0)[0:3]
    data_label[:, 0:3] -= xyz_min
    log_string("input scene: " + data_label_filename + " and shape: " + str(data_label.shape))

    return scene2blocks_plus(data_label, num_point, block_size, stride,
                            random_sample, sample_num, sample_aug)

def scenetoblocks_plus_normalized(data_label, num_point, block_size, stride,
                                sampling, sample_num, sample_aug):
    """ room2block, with RGB preprocessing.
        for each block centralize XYZ, add normalized XYZ as 678 channels
    """
    data = data_label[:,0:6]
    data[:,3:6] /= 255.0
    label = data_label[:,-1].astype(np.uint8)
    max_room_x = max(data[:,0])
    max_room_y = max(data[:,1])
    max_room_z = max(data[:,2])

    data_batch, label_batch = scenetoblocks(data, label, num_point, block_size, stride,
                                          sampling, sample_num, sample_aug)
    new_data_batch = np.zeros((data_batch.shape[0], num_point, 9))
    for b in range(data_batch.shape[0]):
        new_data_batch[b, :, 6] = data_batch[b, :, 0]/max_room_x
        new_data_batch[b, :, 7] = data_batch[b, :, 1]/max_room_y
        new_data_batch[b, :, 8] = data_batch[b, :, 2]/max_room_z
        minx = min(data_batch[b, :, 0])
        miny = min(data_batch[b, :, 1])
        data_batch[b, :, 0] -= (minx+block_size/2)
        data_batch[b, :, 1] -= (miny+block_size/2)
    new_data_batch[:, :, 0:6] = data_batch
    return new_data_batch, label_batch

def scenetoblocks_wrapper_normalized(data_label_filename, num_point, block_size=1.0, stride=1.0,
                                 sampling=False, sample_num=None, sample_aug=1):
    data_label = np.loadtxt(data_label_filename)
    data_label = data_label[:,0:7]
    xyz_min = np.amin(data_label, axis=0)[0:3]
    data_label[:, 0:3] -= xyz_min
    return scenetoblocks_plus_normalized(data_label, num_point, block_size, stride,
                                       sampling, sample_num, sample_aug)

LOG_FOUT = open(os.path.join(ROOT_DIR, 'LOG','splitblock_log.txt'), 'w')
def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)

def room2samples(data, label, sample_num_point):
    """ Prepare whole room samples.
    Args:
        data: N x 6 numpy array, 012 are XYZ in meters, 345 are RGB in [0,1]
            assumes the data is shifted (min point is origin) and
            aligned (aligned with XYZ axis)
        label: N size uint8 numpy array from 0-12
        sample_num_point: int, how many points to sample in each sample
    Returns:
        sample_datas: K x sample_num_point x 9
                     numpy array of XYZRGBX'Y'Z', RGB is in [0,1]
        sample_labels: K x sample_num_point x 1 np array of uint8 labels
    """
    N = data.shape[0]
    order = np.arange(N)
    np.random.shuffle(order)
    data = data[order, :]
    label = label[order]

    batch_num = int(np.ceil(N / float(sample_num_point)))
    sample_datas = np.zeros((batch_num, sample_num_point, 6))
    sample_labels = np.zeros((batch_num, sample_num_point, 1))

    for i in range(batch_num):
        beg_idx = i*sample_num_point
        end_idx = min((i+1)*sample_num_point, N)
        num = end_idx - beg_idx
        sample_datas[i,0:num,:] = data[beg_idx:end_idx, :]
        sample_labels[i,0:num,0] = label[beg_idx:end_idx]
        if num < sample_num_point:
            makeup_indices = np.random.choice(N, sample_num_point - num)
            sample_datas[i,num:,:] = data[makeup_indices, :]
            sample_labels[i,num:,0] = label[makeup_indices]
    return sample_datas, sample_labels

def room2samples_plus_normalized(data_label, num_point):
    """ room2sample, with input filename and RGB preprocessing.
        for each block centralize XYZ, add normalized XYZ as 678 channels
    """
    data = data_label[:,0:6]
    data[:,3:6] /= 255.0
    label = data_label[:,-1].astype(np.uint8)
    max_room_x = max(data[:,0])
    max_room_y = max(data[:,1])
    max_room_z = max(data[:,2])
    #print(max_room_x, max_room_y, max_room_z)

    data_batch, label_batch = room2samples(data, label, num_point)
    new_data_batch = np.zeros((data_batch.shape[0], num_point, 9))
    for b in range(data_batch.shape[0]):
        new_data_batch[b, :, 6] = data_batch[b, :, 0]/max_room_x
        new_data_batch[b, :, 7] = data_batch[b, :, 1]/max_room_y
        new_data_batch[b, :, 8] = data_batch[b, :, 2]/max_room_z
        #minx = min(data_batch[b, :, 0])
        #miny = min(data_batch[b, :, 1])
        #data_batch[b, :, 0] -= (minx+block_size/2)
        #data_batch[b, :, 1] -= (miny+block_size/2)
    new_data_batch[:, :, 0:6] = data_batch
    return new_data_batch, label_batch


def room2samples_wrapper_normalized(data_label_filename, num_point):
    if data_label_filename[-3:] == 'txt':
        data_label = np.loadtxt(data_label_filename)
    elif data_label_filename[-3:] == 'npy':
        data_label = np.load(data_label_filename)
    else:
        print('Unknown file type! exiting.')
        exit()
    return room2samples_plus_normalized(data_label, num_point)

def inferbatch_num(data, label, batch_size):
    file_size = data.shape[0]
    num_batches = file_size // batch_size
    log_string('current_data has ' + str(num_batches) + str(' batches'))
    current_data = []
    current_label = []
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = (batch_idx+1) * batch_size
        current_data.append(data[start_idx:end_idx, :, :])
        current_label.append(label[start_idx:end_idx,:])
        log_string('current data size: ' + str(data[start_idx:end_idx, :,:].shape) + 'current label size: ' +
                str(label[start_idx:end_idx,:].shape))
    current_data = np.array(current_data)
    current_label = np.array(current_label)
    log_string('current data size: ' + str(current_data.shape) + 'current label size: ' + str(current_label.shape))
    return current_data, current_label, num_batches

"""
# test
if __name__ == '__main__':
    building_path = os.path.join(DATA_PATH, '12_KAS_pavillion_1.txt')
    NUM_POINT = 2048
    current_data, current_label = scenetoblocks_wrapper_normalized(building_path, num_point=NUM_POINT)
    print("finish spliting!")
    log_string("splited data size: ")
    log_string("current_data: " + str(current_data.shape)) #(93,2048,9)
    log_string("current_label: " + str(current_label.shape)) #(93,2048)
    batch_size = 8
    batch_data, batch_label, num_batches = inferbatch_num(current_data, current_label, batch_size)
"""

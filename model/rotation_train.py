#
#
#      0=================================0
#      |    Project Name                 |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Implements: Trainer  __init__(), train()
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      YUWEI CAO - 2021/5/20 08:39 AM
#
#


# ----------------------------------------
# Import packages and constant
# ----------------------------------------
import time
import os
import sys
import numpy as np
import shutil
import torch
import argparse
import torch.optim as optim
from torch.autograd import Variable

from tensorboardX import SummaryWriter
from model import MultiTaskNet

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

sys.path.append(os.path.join(ROOT_DIR, 'data_preprocessing'))
#from ArCH import ArchDataset
from arch_dataloader import get_dataloader
from pre_rotation import rotate_pc

DATA_DIR = os.path.join(ROOT_DIR, 'data')


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
parser.add_argument('--rec_loss', type=str, default='ChamferLoss_m', choices=['ChamferLoss_m','ChamferLoss'],
                    help='reconstruction loss')
parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',
                    help='Size of batch)')
parser.add_argument('--workers', type=int, help='Number of data loading workers', default=16)
parser.add_argument('--epochs', type=int, default=250, metavar='N',
                    help='Number of episode to train ')
parser.add_argument('--snapshot_interval', type=int, default=10, metavar='N',
                    help='Save snapshot interval ')
parser.add_argument('--gpu_mode', action='store_true', default=True,
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

DATASET = args.dataset
MAX_EPOCH = args.epochs
BATCH_SIZE = args.batch_size
SNAPSHOT_INTERVAL = args.snapshot_interval
GPU_MODE = args.gpu_mode
MODEL_PATH = args.model_path
NUM_WORKERS = args.num_workers
NUM_ANGLES = args.num_angles

# create output directory and files
file = [f for f in MODEL_PATH.split('/')]
if args.experiment_name != None:
    experiment_id = args.task + args.experiment_name
elif MODEL_PATH != '' and file[-2] == 'models':
    experiment_id = file[-3]
else:
    experiment_id = f"{args.task}_{args.dataset}_rot_angles_{args.num_angles}_sample_{args.sample}_block_size_{args.block_size}_{args.num_points}_point_dims_{args.num_dims}_feat_dims_{args.feat_dims}_batch_{args.batch_size}" 
SAVE_DIR = os.path.join(ROOT_DIR, 'snapshot', experiment_id, 'models/')
TBOARD_DIR = os.path.join(ROOT_DIR, 'tensorboard',experiment_id)
WRITER = SummaryWriter(log_dir = TBOARD_DIR)

#chenck arguments
if MODEL_PATH == '':
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    else:
        choose = input("Remove " + SAVE_DIR + " ? (y/n)")
        if choose == 'y':
            shutil.rmtree(SAVE_DIR)
            os.makedirs(SAVE_DIR)
        else:
            sys.exit(0)
    if not os.path.exists(TBOARD_DIR):
        os.makedirs(TBOARD_DIR)
    else:
        shutil.rmtree(TBOARD_DIR)
        os.makedirs(TBOARD_DIR)

#config logging
log_para_name = f"{args.task}_{args.dataset}_rot_angles_{args.num_angles}_sample_{args.sample}_block_size_{args.block_size}_{args.num_points}_point_dims_{args.num_dims}_feat_dims_{args.feat_dims}_batch_{args.batch_size}" 
LOG_DIR = os.path.join(ROOT_DIR, 'LOG', log_para_name)
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
print("Logging to", LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train_rot.txt'), 'a')
LOG_FOUT.write(str(args)+'\n')

# initial dataset by dataloader
arch_data_dir = args.folder if args.folder else 'arch3_no_others_combined_5m_2048'
filelist = os.path.join(DATA_DIR, arch_data_dir, "train_data_files.txt")

print('-Now loading ArCH dataset...')
TRAIN_LOADER = get_dataloader(filelist=filelist, is_rotated=False, split='train', batch_size=args.batch_size, num_workers=args.workers, num_points=args.num_points, num_dims=args.num_dims, group_shuffle=False, random_rotate = args.use_rotate, random_jitter=args.use_jitter, random_translate=args.use_translate, shuffle=args.use_shuffle, drop_last=True)
print("training set size: "+str(TRAIN_LOADER.dataset.__len__()))

#initial model
MODEL = MultiTaskNet(args)

#load pretrained model
if MODEL_PATH != '':
    load_pretrain(MODEL_PATH)

# load model to gpu
if GPU_MODE:
    MODEL = MODEL.cuda()

# initialize optimizer
PARAMETER = MODEL.parameters()
OPTIMIZER = optim.Adam([{'params': PARAMETER, 'initial_lr': 1e-4}], lr=0.0001*16/BATCH_SIZE, weight_decay=1e-6)
SCHEDULER = optim.lr_scheduler.StepLR(OPTIMIZER, 20, 0.5, MAX_EPOCH)
TRAIN_HIST={
       'loss': [],
        'acc': [],
       'per_epoch_time': [],
       'total_time': []}

def run():
    best_loss = 1000000000

    # start epoch index
    if MODEL_PATH != '':
        start_epoch = MODEL_PATH[-7:-4]
        if start_epoch[0] == '_':
            start_epoch = start_epoch[1:]
        start_epoch=int(start_epoch)
    else:
        start_epoch = 0

    # start training
    print('training start!!!')
    start_time = time.time()
    MODEL.train()
    for epoch in range(start_epoch, MAX_EPOCH):
        log_string('**** EPOCH %03d ****' % (epoch))
        sys.stdout.flush()
        loss = rotate_train_epoch(epoch)
        # save snapeshot
        if (epoch+1) % SNAPSHOT_INTERVAL == 0 or epoch == 0:
            snapshot(epoch+1)
            if loss < best_loss:
                best_loss = loss
                snapshot('best')

        # save tensorboard
        if WRITER:
            WRITER.add_scalar('Train Loss', TRAIN_HIST['loss'][-1], epoch)
            WRITER.add_scalar('Learning Rate', get_lr(), epoch)
            WRITER.add_scalar('Train Accuracy', np.mean(TRAIN_HIST['acc']), epoch)
        log_string("end epoch " + str(epoch) + ", training loss: " + str(TRAIN_HIST['loss'][-1]))
    # finish all epochs
    snapshot(epoch+1)
    if loss < best_loss:
        best_loss = loss
        snapshot('best')
    TRAIN_HIST['total_time'].append(time.time()-start_time)
    print("Avg one epoch time: %.2f, total %d epoches time: %.2f" % (np.mean(TRAIN_HIST['per_epoch_time']),
        MAX_EPOCH, TRAIN_HIST['total_time'][0]))
    print("Training finish!... save training results")


def rotate_train_epoch(epoch):
    epoch_start_time = time.time()
    loss_buf = []
    num_train = len(TRAIN_LOADER.dataset)
    num_batch = int(num_train/BATCH_SIZE)
    log_string("total training nuber: " + str(num_train) + "total batch number: " + str(num_batch) + " .")
    for iter, (pts, _) in enumerate(TRAIN_LOADER):
        pts = Variable(pts)
        rotated_data, rotated_label = rotate_pc(current_data=pts, NUM_CLASSES = NUM_ANGLES)
        rotated_data = torch.from_numpy(rotated_data)
        rotated_label = torch.from_numpy((np.array([rotated_label]).astype(np.int64)))
        if GPU_MODE:
            rotated_data = rotated_data.cuda()
            rotated_label = rotated_label.cuda()
        # forward
        OPTIMIZER.zero_grad()
        #input(bs, 2048, 3), rec_output(bs, 2025,3), rot_ouput(bs, 2048)
        rec_output, rot_output, _ , _ = MODEL(rotated_data)
        rotated_label= rotated_label.view(-1,1)[:,0]
        #print("input: " + pts.shape + ", output shape: " + output.shape)
        loss = MODEL.get_loss(rotated_data, rec_output, rotated_label, rot_output)
        # backward
        loss.backward()
        OPTIMIZER.step()
        loss_buf.append(loss.detach().cpu().numpy())
        #update lr
        rot_output = rot_output.view(-1, NUM_ANGLES)  
        pred_choice = rot_output.data.cpu().max(1)[1]
        correct = pred_choice.eq(rotated_label.data.cpu()).cpu().sum()
        log_string('[%d: %d/%d] | train loss: %f | train accuracy: %f' %(epoch+1, iter+1, num_batch, np.mean(loss_buf), correct.item()/float(BATCH_SIZE*NUM_ANGLES)))
    if OPTIMIZER.param_groups[0]['lr'] > 1e-5:
        SCHEDULER.step()
    if OPTIMIZER.param_groups[0]['lr'] < 1e-5:
        for param_group in OPTIMIZER.param_groups:
            param_group['lr'] = 1e-5
    # finish one epoch
    epoch_time = time.time() - epoch_start_time
    TRAIN_HIST['per_epoch_time'].append(epoch_time)
    TRAIN_HIST['loss'].append(np.mean(loss_buf))
    TRAIN_HIST['acc'].append(np.mean(correct.item()/float(BATCH_SIZE*NUM_ANGLES)))
    log_string(f'Epoch {epoch+1}: Loss {np.mean(loss_buf)}, time {epoch_time:.4f}s')
    return np.mean(loss_buf)


def snapshot(epoch):
    state_dict = MODEL.state_dict()
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for key, val in state_dict.items():
        if key[:6] == 'module':
            name = key[7:]  # remove 'module.'
        else:
            name = key
        new_state_dict[name] = val
    save_dir = os.path.join(SAVE_DIR, DATASET)
    torch.save(new_state_dict, save_dir + "_" + str(epoch) + '.pkl')
    log_string(f"Save model to {save_dir}_{epoch}.pkl")


def load_pretrain(pretrain):
    state_dict = torch.load(pretrain, map_location='cpu')
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for key, val in state_dict.items():
        if key[:6] == 'module':
            name = key[7:]  # remove 'module.'
        else:
            name = key
        new_state_dict[name] = val
    MODEL.load_state_dict(new_state_dict)
    log_string(f"Load model from {pretrain}")


def get_lr(group=0):
    return OPTIMIZER.param_groups[group]['lr']

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)
    

if __name__ == "__main__":
    run()
    LOG_FOUT.close()

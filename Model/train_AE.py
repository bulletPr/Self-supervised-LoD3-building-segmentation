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
#      YUWEI CAO - 2020/10/25 08:54 AM
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
#import argparse
import torch.optim as optim

from tensorboardX import SummaryWriter
from model import DGCNN_FoldNet
from model_feat9 import DGCNN_FoldNet_feat

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

sys.path.append(os.path.join(ROOT_DIR, 'datasets'))
#from ArCH import ArchDataset
from arch_dataloader import get_dataloader
from shapenet_dataloader import get_shapenet_dataloader

sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from pc_utils import is_h5_list, load_seg_list
from net_utils import Logger

DATA_DIR = os.path.join(ROOT_DIR, 'data')


# ----------------------------------------
# Trainer class
# ----------------------------------------

class Train_AE(object):
    def __init__(self, args):
        self.dataset_name = args.dataset
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        #self.data_dir = os.path.join(ROOT_DIR, 'data')
        self.snapshot_interval = args.snapshot_interval
        self.gpu_mode = args.gpu_mode
        self.model_path = args.model_path
        self.split = args.split
        self.num_workers = args.num_workers

        # create output directory and files
        file = [f for f in args.model_path.split('/')]
        if args.experiment_name != None:
            self.experiment_id = 'Reconstruct_' + args.experiment_name
        elif file[-2] == 'models':
            self.experiment_id = file[-3]
        else:
            self.experiment_id = "Reconstruct" + time.strftime('%m%d%H%M%S')
        snapshot_root = 'snapshot/%s' %self.experiment_id
        tensorboard_root = 'tensorboard/%s' %self.experiment_id
        self.save_dir = os.path.join(ROOT_DIR, snapshot_root, 'models/')
        self.tboard_dir = os.path.join(ROOT_DIR, tensorboard_root)

        #chenck arguments
        if self.model_path == '':
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            else:
                choose = input("Remove " + self.save_dir + " ? (y/n)")
                if choose == 'y':
                    shutil.rmtree(self.save_dir)
                    os.makedirs(self.save_dir)
                else:
                    sys.exit(0)
            if not os.path.exists(self.tboard_dir):
                os.makedirs(self.tboard_dir)
            else:
                shutil.rmtree(self.tboard_dir)
                os.makedirs(self.tboard_dir)
        sys.stdout = Logger(os.path.join(ROOT_DIR, 'LOG', 'ae_network_log.txt'))
        self.writer = SummaryWriter(log_dir = self.tboard_dir)
        print(str(args))
        
        
        # initial dataset by dataloader
        print('-Preparing dataset...')
        if self.dataset_name == 'arch':
            # initial dataset filelist
            print('-Preparing dataset file list...')
            if self.split == 'train':
                filelist = os.path.join(DATA_DIR, 'arch_pointcnn_hdf5_4096', "train_data_files.txt")
            else:
                filelist = os.path.join(DATA_DIR, 'arch_pointcnn_hdf5_4096', "test_data_files.txt")
            
            self.is_list_of_h5_list = not is_h5_list(filelist)
            if self.is_list_of_h5_list:
                self.seg_list = load_seg_list(filelist)
                self.seg_list_idx = 0
                filepath = self.seg_list[self.seg_list_idx]
                self.seg_list_idx += 1
            else:
                filepath = filelist
        
            print('-Now loading ArCH dataset...')
            self.train_loader = get_dataloader(filelist=filepath, batch_size=args.batch_size, num_workers=args.workers, group_shuffle=True, 
                                    random_rotate = args.use_rotate, random_jitter=args.use_jitter, random_translate=args.use_translate, shuffle=True)
            print("training set size: ", self.train_loader.dataset.__len__())
        
        elif self.dataset_name == 'all_arch':
            # initial dataset filelist
            if args.no_others:
                arch_data_dir = args.folder if args.folder else 'arch_no_others_pointcnn_hdf5_'+str(args.num_points)
            else:
                arch_data_dir = args.folder if args.folder else 'arch_pointcnn_hdf5_'+str(args.num_points)
            print('-Preparing dataset file list...')
            filelist = os.path.join(DATA_DIR, arch_data_dir, "train_data_files.txt")
        
            print('-Now loading ArCH dataset...')
            self.train_loader = get_dataloader(filelist=filelist, batch_size=args.batch_size, num_workers=args.workers, num_points=args.num_points, group_shuffle=False, 
                                    random_rotate = args.use_rotate, random_jitter=args.use_jitter, random_translate=args.use_translate, shuffle=True, drop_last=True)
            print("training set size: ", self.train_loader.dataset.__len__())
       
        elif self.dataset_name == 'shapenetcorev2':
            print('-Loading ShapeNetCore dataset...')
            self.train_loader = get_shapenet_dataloader(root=DATA_DIR, dataset_name = self.dataset_name, split='train', batch_size=args.batch_size, 
                                    num_workers=args.workers, num_points=args.num_points, shuffle=True, random_translate = args.use_translate, random_rotate = args.use_rotate, random_jitter=args.use_jitter)
            print("training set size: ", self.train_loader.dataset.__len__())
        
        #initial model
        if args.num_dims == 3:
            self.model = DGCNN_FoldNet(args)
        else:
            self.model = DGCNN_FoldNet_feat(args)
        #load pretrained model
        if args.model_path != '':
            self._load_pretrain(args.model_path)

        # load model to gpu
        if self.gpu_mode:
            self.model = self.model.cuda()

        # initialize optimizer
        self.parameter = self.model.parameters()
        self.optimizer = optim.Adam(self.parameter, lr=0.0001*16/args.batch_size, betas=(0.9, 0.999), weight_decay=1e-6)

    def run(self):
        self.train_hist={
               'loss': [],
               'per_epoch_time': [],
               'total_time': []}
        
        best_loss = 1000000000

        # start epoch index
        if self.model_path != '':
            start_epoch = self.model_path[-7:-4]
            if start_epoch[0] == '_':
                start_epoch = start_epoch[1:]
            start_epoch=int(start_epoch)
        else:
            start_epoch = 0

        # start training
        print('training start!!!')
        start_time = time.time()
        self.model.train()
        for epoch in range(start_epoch, self.epochs):
            loss = self.train_epoch(epoch)
            
            # save snapeshot
            if (epoch+1) % self.snapshot_interval == 0 or epoch == 0:
                self._snapshot(epoch+1)
                if loss < best_loss:
                    best_loss = loss
                    self._snapshot('best')

            # save tensorboard
            if self.writer:
                self.writer.add_scalar('Train Loss', self.train_hist['loss'][-1], epoch)
                self.writer.add_scalar('Learning Rate', self._get_lr(), epoch)
            log_string("end epoch " + str(epoch) + ", training loss: " + str(self.train_hist['loss'][-1]))
        # finish all epochs
        self._snapshot(epoch+1)
        if loss < best_loss:
            best_loss = loss
            self._snapshot('best')
        self.train_hist['total_time'].append(time.time()-start_time)
        print("Avg one epoch time: %.2f, total %d epoches time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
            self.epochs, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")


    def train_epoch(self, epoch):
        epoch_start_time = time.time()
        loss_buf = []
        num_train = len(self.train_loader.dataset)
        num_batch = int(num_train/self.batch_size)
        log_string("total training nuber: " + str(num_train) + "total batch number: " + str(num_batch) + " .")
        for iter, (pts, _) in enumerate(self.train_loader):
            log_string("batch idx: " + str(iter) + "/" + str(num_batch) + " in " + str(epoch) + "/" + str(self.epochs) + " epoch...")
            if self.gpu_mode:
                pts = pts.cuda()
            
            start_idx = (self.batch_size * iter) % num_train
            end_idx = min(start_idx + self.batch_size, num_train)
            batch_size_train = end_idx - start_idx
            
            if self.dataset_name == 'arch':
                if start_idx + batch_size_train == num_train:
                    if self.is_list_of_h5_list:
                        filelist_train_prev = self.seg_list[(self.seg_list_idx - 1) % len(self.seg_list)]
                        filelist_train = self.seg_list[self.seg_list_idx % len(self.seg_list)]
                        if filelist_train != filelist_train_prev:
                            self.train_loader = get_dataloader(filelist=filelist_train, batch_size=self.batch_size, num_workers=self.workers)
                            num_train = len(self.train_loader.dataset)
                        self.seg_list_idx += 1
            
            # forward
            self.optimizer.zero_grad()
            #input(bs, 2048, 3), output(bs, 2025,3)
            output, _ , _ = self.model(pts)
            #print("input: " + pts.shape + ", output shape: " + output.shape)
            loss = self.model.get_loss(pts, output)
            # backward
            loss.backward()
            self.optimizer.step()
            loss_buf.append(loss.detach().cpu().numpy())

        # finish one epoch
        epoch_time = time.time() - epoch_start_time
        self.train_hist['per_epoch_time'].append(epoch_time)
        self.train_hist['loss'].append(np.mean(loss_buf))
        print(f'Epoch {epoch+1}: Loss {np.mean(loss_buf)}, time {epoch_time:.4f}s')
        return np.mean(loss_buf)


    def _snapshot(self, epoch):
        state_dict = self.model.state_dict()
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for key, val in state_dict.items():
            if key[:6] == 'module':
                name = key[7:]  # remove 'module.'
            else:
                name = key
            new_state_dict[name] = val
        save_dir = os.path.join(self.save_dir, self.dataset_name)
        torch.save(new_state_dict, save_dir + "_" + str(epoch) + '.pkl')
        print(f"Save model to {save_dir}_{epoch}.pkl")


    def _load_pretrain(self, pretrain):
        state_dict = torch.load(pretrain, map_location='cpu')
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for key, val in state_dict.items():
            if key[:6] == 'module':
                name = key[7:]  # remove 'module.'
            else:
                name = key
            new_state_dict[name] = val
        self.model.load_state_dict(new_state_dict)
        print(f"Load model from {pretrain}")


    def _get_lr(self, group=0):
        return self.optimizer.param_groups[group]['lr']

LOG_FOUT = open(os.path.join(ROOT_DIR, 'LOG','ae_train_log.txt'), 'a')
def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)
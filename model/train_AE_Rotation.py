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
from torch.autograd import Variable

from tensorboardX import SummaryWriter
from model import MultiTaskNet

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

sys.path.append(os.path.join(ROOT_DIR, 'data_preprocessing'))
#from ArCH import ArchDataset
from arch_dataloader import get_dataloader

DATA_DIR = os.path.join(ROOT_DIR, 'data')


# ----------------------------------------
# Trainer class
# ----------------------------------------

class Train_AE_Rotation(object):
    def __init__(self, args):
        self.dataset = args.dataset
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        #self.data_dir = os.path.join(ROOT_DIR, 'data')
        self.snapshot_interval = args.snapshot_interval
        self.gpu_mode = args.gpu_mode
        self.model_path = args.model_path
        self.num_workers = args.num_workers
        self.angle_number = args.num_angles

        # create output directory and files
        file = [f for f in self.model_path.split('/')]
        if args.experiment_name != None:
            self.experiment_id = args.task + args.experiment_name
        elif self.model_path != '' and file[-2] == 'models':
            self.experiment_id = file[-3]
        else:
            self.experiment_id = f"{args.task}_{args.dataset}_rot_angles_{args.num_angles}_sample_{args.sample}_block_size_{args.block_size}_{args.num_points}_point_dims_{args.num_dims}_feat_dims_{args.feat_dims}_batch_{args.batch_size}" 
        self.save_dir = os.path.join(ROOT_DIR, 'snapshot', self.experiment_id, 'models/')
        self.tboard_dir = os.path.join(ROOT_DIR, 'tensorboard',self.experiment_id)
        self.writer = SummaryWriter(log_dir = self.tboard_dir)
        
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
        
        #config logging
        log_para_name = f"{args.task}_{args.dataset}_rot_angles_{args.num_angles}_sample_{args.sample}_block_size_{args.block_size}_{args.num_points}_point_dims_{args.num_dims}_feat_dims_{args.feat_dims}_batch_{args.batch_size}" 
        LOG_DIR = os.path.join(ROOT_DIR, 'LOG', log_para_name)
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)
        print("Logging to", LOG_DIR)
        self.LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train_rot.txt'), 'a')
        self.LOG_FOUT.write(str(args)+'\n')
            
        # initial dataset by dataloader
        arch_data_dir = args.folder if args.folder else f'rotated_{args.num_angles}_angle_{args.sample}_{args.block_size}_{args.num_points}'
        filelist = os.path.join(DATA_DIR, arch_data_dir, "train_data_files.txt")
        
        self._log_string('-Now loading ArCH dataset...')
        self.train_loader = get_dataloader(filelist=filelist, is_rotated=True, split='train', batch_size=args.batch_size, num_workers=args.workers, num_points=args.num_points, num_dims=args.num_dims, group_shuffle=False, random_rotate = args.use_rotate, random_jitter=args.use_jitter, random_translate=args.use_translate, shuffle=args.use_shuffle, drop_last=True)
        self._log_string("training set size: "+str(self.train_loader.dataset.__len__()))

        #initial model
        self.model = MultiTaskNet(args)

        #load pretrained model
        if args.model_path != '':
            self._load_pretrain(args.model_path)

        # load model to gpu
        if self.gpu_mode:
            self.model = self.model.cuda()

        # initialize optimizer
        self.parameter = self.model.parameters()
        self.optimizer = optim.Adam([{'params': self.parameter, 'initial_lr': 1e-4}], lr=0.0001*16/args.batch_size, weight_decay=1e-6)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 20, 0.5, args.epochs)
    
        
    def run(self):
        self.train_hist={
               'loss': [],
                'acc': [],
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
            self._log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
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
                self.writer.add_scalar('Train Accuracy', np.mean(self.train_hist['acc']), epoch)
            self._log_string("end epoch " + str(epoch) + ", training loss: " + str(self.train_hist['loss'][-1]))
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
        #num_train = len(self.train_loader.dataset)
        #num_batch = int(num_train/self.batch_size)
        #self._log_string("total training nuber: " + str(num_train) + "total batch number: " + str(num_batch) + " .")
        for iter, (pts, targets) in enumerate(self.train_loader):
            #self._log_string("batch idx: " + str(iter) + "/" + str(num_batch) + " in " + str(epoch) + "/" + str(self.epochs) + " epoch...")
            pts = Variable(pts)
            targets = targets.long()
            if self.gpu_mode:
                pts = pts.cuda()
                targets = targets.cuda()

            # forward
            self.optimizer.zero_grad()
            #input(bs, 2048, 3), rec_output(bs, 2025,3), rot_ouput(bs, 2048)
            rec_output, rot_output, _ , _ = self.model(pts)
            targets= targets.view(-1,1)[:,0]
            #print("input: " + pts.shape + ", output shape: " + output.shape)
            loss = self.model.get_loss(pts, rec_output, targets, rot_output)
            # backward
            loss.backward()
            self.optimizer.step()
            loss_buf.append(loss.detach().cpu().numpy())
                #update lr
            rot_output = rot_output.view(-1, self.angle_number)  
            pred_choice = rot_output.data.cpu().max(1)[1]
            correct = pred_choice.eq(targets.data.cpu()).cpu().sum()
            self._log_string('[%d: %d] | train loss: %f | train accuracy: %f' %(epoch+1, iter+1, np.mean(loss_buf), correct.item()/float(self.batch_size*self.angle_number)))
        if self.optimizer.param_groups[0]['lr'] > 1e-5:
            self.scheduler.step()
        if self.optimizer.param_groups[0]['lr'] < 1e-5:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = 1e-5
        # finish one epoch
        epoch_time = time.time() - epoch_start_time
        self.train_hist['per_epoch_time'].append(epoch_time)
        self.train_hist['loss'].append(np.mean(loss_buf))
        self.train_hist['acc'].append(np.mean(correct.item()/float(self.batch_size*self.angle_number)))
        self._log_string(f'Epoch {epoch+1}: Loss {np.mean(loss_buf)}, time {epoch_time:.4f}s')
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
        save_dir = os.path.join(self.save_dir, self.dataset)
        torch.save(new_state_dict, save_dir + "_" + str(epoch) + '.pkl')
        self._log_string(f"Save model to {save_dir}_{epoch}.pkl")


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
        self._log_string(f"Load model from {pretrain}")


    def _get_lr(self, group=0):
        return self.optimizer.param_groups[group]['lr']
    
    def _log_string(self, out_str):
        self.LOG_FOUT.write(out_str + '\n')
        self.LOG_FOUT.flush()
        print(out_str)

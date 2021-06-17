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
#      YUWEI CAO - 2021/5/31 09:16 AM
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
import torch.nn.functional as F
import h5py
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from model import MultiTaskNet
from semseg_net import SemSegNet


# ----------------------------------------
# Import initial roots
# ----------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

sys.path.append(os.path.join(ROOT_DIR, 'data_preprocessing'))
import arch_dataloader

DATA_DIR = os.path.join(ROOT_DIR, 'data')    
    
# ----------------------------------------
# Load pretained model
# ----------------------------------------
def load_pretrain(model, pretrain):
    state_dict = torch.load(pretrain, map_location='cpu')
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for key, val in state_dict.items():
        if key[:6] == 'module':
            name = key[7:]  # remove 'module.'
        else:
            name = key
        new_state_dict[name] = val
    model.load_state_dict(new_state_dict)
    print(f"Load model from {pretrain}")
    return model  


# ----------------------------------------
# save model
# ----------------------------------------
def _snapshot(save_dir, model, epoch, opt):
    state_dict = model.state_dict()
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for key, val in state_dict.items():
        if key[:6] == 'module':
            name = key[7:]  # remove 'module.'
        else:
            name = key
        new_state_dict[name] = val
    save_dir = os.path.join(save_dir, opt.dataset)
    torch.save(new_state_dict, save_dir+'_training_data_at_epoch_' + str(epoch) + '.pkl')
    print(f"Save model to {save_dir}+'_training_data_at_epoch_{epoch}.pkl'")


# ----------------------------------------
# Train semantic segmentation network
# ----------------------------------------
def main(opt):
    experiment_id = 'Semantic_segmentation_'+ opt.encoder +'_' +opt.pre_ae_epochs + '_' +   str(opt.feat_dims) + '_' + opt.dataset+'_' + str(opt.percentage)+'_percent'
    LOG_FOUT = open(os.path.join(ROOT_DIR, 'LOG', experiment_id+'_train_log.txt'), 'a')
    def log_string(out_str):
        LOG_FOUT.write(out_str + '\n')
        LOG_FOUT.flush()
        print(out_str)
    
    snapshot_root = 'snapshot/%s' %experiment_id
    tensorboard_root = 'tensorboard/%s' %experiment_id
    heatmap_root = 'heatmap/%s' %experiment_id
    save_dir = os.path.join(ROOT_DIR, snapshot_root, 'models/')
    tboard_dir = os.path.join(ROOT_DIR, tensorboard_root)
    hmap_dir = os.path.join(ROOT_DIR, heatmap_root)
    
    #create folder to save trained models
    if opt.model == '':
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        else:
            choose = input("Remove " + save_dir + " ? (y/n)")
            if choose == 'y':
                shutil.rmtree(save_dir)
                os.makedirs(save_dir)
            else:
                sys.exit(0)
        if not os.path.exists(tboard_dir):
            os.makedirs(tboard_dir)
        else:
            shutil.rmtree(tboard_dir)
            os.makedirs(tboard_dir)
        if not os.path.exists(hmap_dir):
            os.makedirs(hmap_dir)
        else:
            shutil.rmtree(hmap_dir)
            os.makedirs(hmap_dir)
    writer = SummaryWriter(log_dir = tboard_dir)

    #generate part label one-hot correspondence from the catagory:
    if opt.dataset == 'arch':
        if opt.no_others:
            class2label = {"arch":0, "column":1, "moldings":2, "floor":3, "door_window":4, "wall":5, "stairs":6, "vault":7, "roof":8}
        else:
            class2label = {"arch":0, "column":1, "moldings":2, "floor":3, "door_window":4, "wall":5, "stairs":6, "vault":7, "roof":8, "other":9}
        seg_classes = class2label
        seg_label_to_cat = {}
        for i,cat in enumerate(seg_classes.keys()):
            seg_label_to_cat[i] = cat
            
    elif opt.dataset == 'arch_scene_2':
        class2label = {"arch":0, "column":1, "moldings":2, "floor":3, "door_window":4, "wall":5, "stairs":6, "vault":7}
        seg_classes = class2label
        seg_label_to_cat = {}
        for i,cat in enumerate(seg_classes.keys()):
            seg_label_to_cat[i] = cat

    # load the dataset
    log_string('-Preparing dataset...')
    data_resized=False
    #train_path = os.path.join(ROOT_DIR, 'cache', 'latent_' + opt.pre_ae_epochs + '_' + opt.dataset + '_' +str(opt.feature_dims), 'features')
    if opt.dataset == 'arch':
        if opt.no_others:
            NUM_CLASSES = 9
            arch_data_dir = opt.folder if opt.folder else 'arch_no_others_1.0m_pointnet_hdf5_data'
        else:
            NUM_CLASSES = 10
            arch_data_dir = opt.folder if opt.folder else "arch_pointcnn_hdf5_2048"
    
    elif opt.dataset == 'arch_scene_2':
            NUM_CLASSES = 8
            arch_data_dir = opt.folder if opt.folder else "scene_2_1.0m_pointnet_hdf5_data"

    train_filelist = os.path.join(DATA_DIR, arch_data_dir, "train_data_files.txt")
    val_filelist = os.path.join(DATA_DIR, arch_data_dir, "val_data_files.txt")
    
    # load training data
    train_dataset = arch_dataloader.get_dataloader(filelist=train_filelist, is_rotated=False, split='train', batch_size=opt.batch_size, num_workers=4, num_points=opt.num_points, num_dims=opt.num_dims, group_shuffle=False, random_translate=opt.use_translate, shuffle=False, drop_last=True)
    log_string("classifer set size: " + str(train_dataset.dataset.__len__()))
    val_dataset = arch_dataloader.get_dataloader(filelist=val_filelist, is_rotated=False, split='train', num_points=opt.num_points, batch_size=opt.batch_size, num_workers=4, num_dims=opt.num_dims, group_shuffle=False, shuffle=False, random_translate=opt.use_translate, drop_last=False)
    log_string("classifer set size: " + str(val_dataset.dataset.__len__()))

    # load the model for point auto encoder    
    if opt.num_dims == 3:
        ae_net = MultiTaskNet(opt)
    if opt.ae_model != '':
        ae_net = load_pretrain(ae_net, os.path.join(ROOT_DIR, opt.ae_model))
    if opt.gpu_mode:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        ae_net = ae_net.cuda()
    ae_net=ae_net.eval()
 
    #initial segmentation model  
    sem_seg_net = SemSegNet(num_class=opt.n_classes, encoder=opt.encoder, symmetric_function=opt.symmetric_function, dropout=opt.dropout)    
    #load pretrained model
    if opt.model != '':
        sem_seg_net = load_pretrain(sem_seg_net, opt.model)            
    # load model to gpu
    if opt.gpu_mode:
        sem_seg_net = sem_seg_net.cuda()       
    # initialize optimizer
    optimizer = optim.Adam([{'params': sem_seg_net.parameters(), 'initial_lr': 1e-4}], lr=0.01, weight_decay=1e-6) 
    scheduler = optim.lr_scheduler.StepLR(optimizer, 20, 0.5, opt.n_epochs)
        
# start training
    n_batch = 0
    # start epoch index
    if opt.model != '':
        start_epoch = opt.model[-7:-4]
        if start_epoch[0] == '_':
            start_epoch = start_epoch[1:]
        start_epoch=int(start_epoch)
    else:
        start_epoch = 0
    
    log_string('training start!!!')
    start_time = time.time()
    total_time = []
    best_iou = 0
    for epoch in range(start_epoch, opt.n_epochs):
        train_acc_epoch, train_iou_epoch, test_acc_epoch, test_iou_epoch = [], [], [], []
        loss_buf = []
        sem_seg_net=sem_seg_net.train()
        for iter, data in enumerate(train_dataset):
            points, target = data
            # use the pre-trained AE to encode the point cloud into latent features
            points_ = Variable(points)
            target = target.long()
            if opt.gpu_mode:
                points_ = points_.cuda()
            _, _, latent_caps, mid_features = ae_net(points_) # (batch_size, emb_dims*2), (batch_size, 64*3, num_points)
            con_code = torch.cat([latent_caps.view(-1,opt.feat_dims*opt.symmetric_function,1).repeat(1,1,opt.num_points), mid_features],1).cpu().detach().numpy()
            latent_caps = torch.from_numpy(con_code).float()
            
            if(latent_caps.size(0)<opt.batch_size):
                continue
            latent_caps, target = Variable(latent_caps), Variable(target)
            if opt.gpu_mode:
                latent_caps,target = latent_caps.cuda(), target.cuda()                            
    
# forward
            optimizer.zero_grad()
            #latent_caps=latent_caps.transpose(2, 1)# consider the feature vector size as the channel in the network
            output_digit =sem_seg_net(latent_caps)
            output_digit = output_digit.view(-1, opt.n_classes)        
            #batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
    
            target= target.view(-1,1)[:,0]
            train_loss = F.nll_loss(output_digit, target)
            train_loss.backward()
            optimizer.step()

            pred_choice = output_digit.data.cpu().max(1)[1]
            correct = pred_choice.eq(target.data.cpu()).cpu().sum()
            train_acc_epoch.append(correct.item() / float(opt.batch_size*opt.num_points))
            train_iou_epoch.append(correct.item() / float(2*opt.batch_size*opt.num_points-correct.item()))
            n_batch= train_dataset.dataset.__len__() // opt.batch_size
            loss_buf.append(train_loss.detach().cpu().numpy())
            log_string('[%d: %d/%d] | train loss: %f | train accuracy: %f | train iou: %f' %(epoch+1, iter+1, n_batch, np.mean(loss_buf), correct.item()/float(opt.batch_size * opt.num_points),correct.item() / float(2*opt.batch_size*opt.num_points-correct.item())))
        
        #update lr
        if optimizer.param_groups[0]['lr'] > 1e-5:
            scheduler.step()
        if optimizer.param_groups[0]['lr'] < 1e-5:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-5
        
        # save tensorboard
        total_time.append(time.time()-start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(total_time), epoch+1, total_time[0]))
        writer.add_scalar('Train Loss', np.mean(loss_buf), epoch)
        writer.add_scalar('Train Accuracy', np.mean(train_acc_epoch), epoch)
        writer.add_scalar('Train IoU', np.mean(train_iou_epoch), epoch)
        log_string('---- EPOCH %03d TRAIN ----' % (epoch + 1))
        log_string('epoch %d | mean train accuracy: %f | mean train IoU: %f' %(epoch+1, np.mean(train_acc_epoch), np.mean(train_iou_epoch)))
        
        if (epoch+1) % opt.snapshot_interval == 0 or epoch == 0:
            _snapshot(save_dir, sem_seg_net, epoch + 1, opt)

        with torch.no_grad():
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            n_batch= val_dataset.dataset.__len__()
            labelweights = np.zeros(NUM_CLASSES)
            total_seen_class = [0 for _ in range(NUM_CLASSES)]
            total_correct_class = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
            log_string('---- EPOCH %03d EVALUATION ----' % (epoch + 1))
            #sem_seg_net=sem_seg_net.eval()    
            for iter, data in enumerate(val_dataset):
                points, target = data
                # use the pre-trained AE to encode the point cloud into latent capsules
                points_ = Variable(points)
                target = target.long()
                if opt.gpu_mode:
                    points_ = points_.cuda()
                _, _, latent_caps, mid_features = ae_net(points_)
                con_code = torch.cat([latent_caps.view(-1,opt.feat_dims*opt.symmetric_function,1).repeat(1,1,opt.num_points), mid_features],1).cpu().detach().numpy()
                latent_caps = torch.from_numpy(con_code).float()
                if(latent_caps.size(0)<opt.batch_size):
                    continue
                latent_caps, target = Variable(latent_caps), Variable(target)    
                if opt.gpu_mode:
                    latent_caps,target = latent_caps.cuda(), target.cuda()
                batch_label = target.cpu().data.numpy() #(8, 4096)
                #output
                sem_seg_net = sem_seg_net.eval() 
                output=sem_seg_net(latent_caps) #([8, 4096, n_classes])
                #output to prediction
                pred_val = output.contiguous().cpu().data.numpy()
                pred_val = np.argmax(pred_val, 2) #(8, 4096)

                #convert output to calculate loss
                output_digit = output.view(-1, opt.n_classes) #([32768, 10])  
                target= target.view(-1,1)[:,0] #[32768]
                #print("target shape: " + str(target.shape))
                loss = F.nll_loss(output_digit, target)
                loss_sum +=loss
                correct = np.sum((pred_val == batch_label))
                total_correct += correct
                total_seen += (opt.batch_size * opt.num_points)
                tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
                labelweights += tmp

                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum((batch_label == l) )
                    total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l) )
                    total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)) )
            labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6))
            log_string('epoch %d | eval mean loss: %f' % (epoch+1,loss_sum / float(n_batch)))
            log_string('epoch %d | eval point avg class IoU: %f' % (epoch+1, mIoU))                
            log_string('epoch %d | eval point accuracy: %f' % (epoch+1, total_correct / float(total_seen)))
            log_string('epoch %d | eval point avg class acc: %f' % (epoch+1,
                np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))))
            iou_per_class_str = '------- IoU --------\n'
            for l in range(NUM_CLASSES):
                iou_per_class_str += 'epoch %d | class %s weight: %.3f, IoU: %.3f \n' % (epoch+1, 
                    seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])), labelweights[l - 1],
                    total_correct_class[l] / float(total_iou_deno_class[l]))
            log_string(iou_per_class_str)
            writer.add_scalar('Eval Loss', loss_sum / float(n_batch), epoch)
            writer.add_scalar('Eval mIoU', mIoU, epoch)
            writer.add_scalar('Eval Point accuracy', total_correct / float(total_seen), epoch)
            if mIoU >= best_iou:
                best_iou = mIoU
                _snapshot(save_dir, sem_seg_net, 'best', opt)
                log_string('Saving model....')
            log_string('Best mIoU: %f at epoch %d' % (best_iou, epoch+1))
    log_string("Training finish!... save training results")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('--pre_ae_epochs', type=str, default='arch_50', help='choose which pre-trained ae to use')
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
    parser.add_argument('--gpu_mode', action='store_true', help='Enables CUDA training')
    parser.add_argument('--num_points', type=int, default=2048, help='input point set size')
    parser.add_argument('--model', type=str, default='', help='model path')
    parser.add_argument('--dropout', action='store_true',
                        help='Enables dropout when training')
    parser.add_argument('--ae_model', type=str, default='snapshot/multitask_arch_rot_angles_4_sample_combined_block_size_5m_2048_point_dims_3_feat_dims_512_batch_4/models/arch_50.pkl', 
                        help='model path for the pre-trained ae network')
    parser.add_argument('--dataset', type=str, default='arch', help='dataset: s3dis, arch')
    parser.add_argument('--percentage', type=int, default=100, help='training cls with percent of training_data')
    parser.add_argument('--n_classes', type=int, default=10, help='semantic classes in all the catagories')
    parser.add_argument('--encoder', type=str, default='foldingnet', help='encoder use')
    parser.add_argument('--k', type=int, default=None)
    parser.add_argument('--feat_dims', type=int, default=1024)
    parser.add_argument('--num_dims', type=int, default=3, metavar='N',
                        help='Number of dims for feature ')
    parser.add_argument('--rec_loss', type=str, default='ChamferLoss', choices=['ChamferLoss_m','ChamferLoss'],
                        help='reconstruction loss')
    parser.add_argument('--use_translate', action='store_true', help='Enables CUDA training')
    parser.add_argument('--snapshot_interval', type=int, default=1, metavar='N',
                        help='Save snapshot interval ')
    parser.add_argument('--no_others', action='store_true', help='Enables CUDA training')
    parser.add_argument('--folder', '-f', help='path to data file')
    parser.add_argument('--symmetric_function', type=int, default=2,
                        help='symmetric function')
    parser.add_argument('--num_angles', type=int, default=6, metavar='N',
                        help='Number of rotation angles')

    opt = parser.parse_args()
    print(opt)

    main(opt)
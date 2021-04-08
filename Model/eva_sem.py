#
#
#      0=================================0
#      |    Project Name                 |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Implements: save output latent features to .h5 files to train semantic segmentation network
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      YUWEI CAO - 2020/11/22 21:39 PM 
#
#


# ----------------------------------------
# import packages
# ----------------------------------------
import argparse
import torch
import torch.nn.parallel
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import sys
import os
import numpy as np
import statistics
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, '../datasets')))
import arch_dataloader

from model import DGCNN_FoldNet
from model_feat9 import DGCNN_FoldNet_feat
from semseg_net import SemSegNet

#import h5py
import json
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'data')

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


def main(opt):
    USE_CUDA = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    label2color = {
        0: [255, 255, 255],  # white
        1: [0, 0, 255],  # blue
        2: [128, 0, 0],  # maroon
        3: [255, 0, 255],  # fuchisia
        4: [0, 128, 0],  # green
        5: [255, 0, 0],  # red
        6: [128, 0, 128],  # purple
        7: [0, 0, 128],  # navy
        8: [128, 128, 0],  # olive
        9: [128, 128, 128]
    }
    experiment_id = opt.seg_model.split('/')[-3]+'_test'
    n_epoch = opt.seg_model.split('/')[-1][-7:-4]
    LOG_FOUT = open(os.path.join(ROOT_DIR, 'LOG',experiment_id+'_at_epoch_'+n_epoch+'_log.txt'), 'a')
    def log_string(out_str):
        LOG_FOUT.write(out_str + '\n')
        LOG_FOUT.flush()
        print(out_str)
    output_root = 'output/%s' %experiment_id
    save_dir = os.path.join(ROOT_DIR, output_root, 'at_'+n_epoch)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if opt.visual:
        if opt.dataset == 'arch':
            fout = open(os.path.join(save_dir, 'Scene_A_pred.obj'),'w')
            fout_gt = open(os.path.join(save_dir, 'Scene_A_gt.obj'), 'w')
        elif opt.dataset == 'arch_scene_2':
            fout = open(os.path.join(save_dir, 'Scene_2_pred.obj'),'w')
            fout_gt = open(os.path.join(save_dir, 'Scene_2_gt.obj'), 'w')
    
# generate part label one-hot correspondence from the catagory:
    if opt.dataset == 's3dis':
        classes = ['ceiling','floor','wall','beam','column','window','door','table','chair','sofa','bookcase','board','clutter']
        class2label = {cls: i for i,cls in enumerate(classes)}
        seg_classes = class2label
        seg_label_to_cat = {}
        for i,cat in enumerate(seg_classes.keys()):
            seg_label_to_cat[i] = cat
    elif opt.dataset == 'arch':
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

    # load the model for point auto encoder    
    if opt.num_dims == 3:
        ae_net = DGCNN_FoldNet(opt)
    else:
        ae_net = DGCNN_FoldNet_feat(opt)
    if opt.ae_model != '':
        ae_net = load_pretrain(ae_net, os.path.join(ROOT_DIR, opt.ae_model))
    if USE_CUDA:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        ae_net = ae_net.cuda()
    ae_net=ae_net.eval() 
    
# load the model for capsule wised part segmentation
    if opt.feat_dims == 512:      
        sem_seg_net = SemSegNet(num_class=opt.n_classes, encoder=opt.encoder, dropout=opt.dropout, feat_dims=True, with_rgb=False)    
    elif opt.feat_dims == 1024:
        sem_seg_net = SemSegNet(num_class=opt.n_classes, encoder=opt.encoder, dropout=opt.dropout)
    if opt.seg_model != '':
        sem_seg_net=load_pretrain(sem_seg_net, os.path.join(ROOT_DIR,opt.seg_model))
    if USE_CUDA:
        sem_seg_net = sem_seg_net.cuda()
    sem_seg_net = sem_seg_net.eval()    

# load dataset
    if opt.dataset=='s3dis':
        print('-Preparing Loading s3dis test dataset...')
        root = os.path.join(DATA_DIR,'stanford_indoor3d')
        NUM_CLASSES = 13
        NUM_POINT = opt.num_points
        BATCH_SIZE = opt.batch_size
        dataset = S3DISDataset(split='test', data_root=root, num_point=NUM_POINT, rgb=False, test_area=5, block_size=1.0, sample_rate=1.0, transform=None)
        log_string("start loading test data ...")
        dataLoader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=True, worker_init_fn = lambda x: np.random.seed(x+int(time.time())))
        log_string("classifer set size: " + str(dataloader.dataset.__len__()))

    elif opt.dataset == 'arch':
        print('-Preparing Loading ArCH evaluation dataset...')
        if opt.no_others:
            NUM_CLASSES = 9
            arch_dir = opt.folder if opt.folder else 'arch_no_others_1.0m_pointnet_hdf5_data'
        else:
            NUM_CLASSES = 10
            arch_dir = opt.folder if opt.folder else "arch_pointcnn_hdf5_2048"
        log_string('-Now loading test ArCH dataset...')
        filelist = os.path.join(DATA_DIR, arch_dir, "test_data_files.txt")
        # load test data
        dataloader = arch_dataloader.get_dataloader(filelist=filelist, num_dims=opt.num_dims, batch_size=opt.batch_size, num_points = opt.num_points,
                                                num_workers=4, group_shuffle=False,shuffle=False, random_rotate=False, random_jitter=False,random_translate=opt.use_translate, drop_last=False)
        log_string("classifer set size: " + str(dataloader.dataset.__len__()))
    elif opt.dataset == 'arch_scene_2':
        NUM_CLASSES = 8
        print('-Preparing Loading ArCH evaluation dataset...')
        log_string('-Now loading test ArCH dataset...')
        filelist = os.path.join(DATA_DIR, "scene_2_1.0m_pointnet_hdf5_data", "test_data_files.txt")
        # load test data
        dataloader = arch_dataloader.get_dataloader(filelist=filelist, batch_size=opt.batch_size, num_points = opt.num_points,
                                                num_workers=4, group_shuffle=False,shuffle=False, random_rotate=False, random_jitter=False,random_translate=opt.use_translate, drop_last=False)
        log_string("classifer set size: " + str(dataloader.dataset.__len__()))
    correct_sum=0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
    points_collector = []
    pd_labels_collector = []
    gt_labels_collector = []
    for batch_id, data in enumerate(dataloader):
        total_seen_class_tmp = [0 for _ in range(NUM_CLASSES)]
        total_correct_class_tmp = [0 for _ in range(NUM_CLASSES)]
        total_iou_deno_class_tmp = [0 for _ in range(NUM_CLASSES)]
        
        points, target = data        
        if(points.size(0)<opt.batch_size):
            break

        # use the pre-trained AE to encode the point cloud into latent capsules
        points_ = Variable(points)
        target = target.long()
        if USE_CUDA:
            points_ = points_.cuda()

        _, latent_caps, mid_features = ae_net(points_)

        con_code = torch.cat([latent_caps.view(-1,opt.feat_dims,1).repeat(1,1,opt.num_points), mid_features],1).cpu().detach().numpy()
        latent_caps = torch.from_numpy(con_code).float()
        # predict the part class per capsule
        latent_caps, target = Variable(latent_caps), Variable(target)
        if USE_CUDA:
            latent_caps,target = latent_caps.cuda(), target.cuda()
        output = sem_seg_net(latent_caps)        
        output_digit = output.view(-1, opt.n_classes)        
        #batch_label = target.cpu().data.numpy()
    
        target= target.view(-1,1)[:,0]
        pred_choice = output_digit.data.cpu().max(1)[1]
        
        correct = pred_choice.eq(target.data.cpu()).cpu().sum()
        correct_sum=correct_sum+correct.item()
        target = target.cpu().data.numpy()
        pred_choice = pred_choice.cpu().data.numpy()
        # calculate the accuracy with the GT
        for l in range(NUM_CLASSES):
            total_seen_class_tmp[l] += np.sum((target == l))
            total_correct_class_tmp[l] += np.sum((pred_choice == l) & (target == l))
            total_iou_deno_class_tmp[l] += np.sum(((pred_choice == l) | (target == l)))
            total_seen_class[l] += total_seen_class_tmp[l]
            total_correct_class[l] += total_correct_class_tmp[l]
            total_iou_deno_class[l] += total_iou_deno_class_tmp[l]

        iou_map = np.array(total_correct_class_tmp) / (np.array(total_iou_deno_class_tmp, dtype=np.float) + 1e-6)
        #print(iou_map)
        arr = np.array(total_seen_class_tmp)
        tmp_iou = np.mean(iou_map[arr != 0])
        #log_string('Mean IoU of batch %d in Scene_A: %.4f' % (batch_id+1,tmp_iou))
        #log_string('----------------------------')

        points_collector.extend(points.reshape((-1,3)).numpy())
        pd_labels_collector.extend(pred_choice)
        gt_labels_collector.extend(target)
    
    if opt.visual:
        log_string('Writing results...')
        sparse_labels = np.array(pd_labels_collector).astype(int).flatten()
        if opt.dataset == 'arch':
            pd_label_filename = save_dir + '/Scene_A_pd_labels.txt'
            gt_label_filename = save_dir + '/Scene_A_gt_labels.txt'
        elif opt.dataset == 'arch_scene_2':
            pd_label_filename = save_dir + '/Scene_2_pd_labels.txt'
            gt_label_filename = save_dir + '/Scene_2_gt_labels.txt'
        #np.savetxt(pd_label_filename, sparse_labels, fmt='%d', delimiter='\n')
        #log_string("Exported sparse labels to {}".format(pd_label_filename))
        gt_labels = np.array(gt_labels_collector).astype(int).flatten()
        #np.savetxt(gt_label_filename, gt_labels, fmt='%d', delimiter='\n')
        #log_string("Exported sparse labels to {}".format(gt_label_filename))
        #print(points_collector.size())
        sparse_points = np.array(points_collector).reshape((-1, 3))

        for i in range(gt_labels.shape[0]):
            color = label2color[sparse_labels[i]]
            color_gt = label2color[gt_labels[i]]
            if opt.visual:
                fout.write('v %f %f %f %d %d %d\n' % (
                    sparse_points[i, 0], sparse_points[i, 1], sparse_points[i, 2], color[0], color[1],
                    color[2]))
                fout_gt.write(
                    'v %f %f %f %d %d %d\n' % (
                    sparse_points[i, 0], sparse_points[i, 1], sparse_points[i, 2], color_gt[0],
                    color_gt[1], color_gt[2]))
        log_string("Exported sparse pcd to {}".format(save_dir))
    if opt.visual:
        fout.close()
        fout_gt.close()    

    IoU = np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6)
    iou_per_class_str = '------- IoU --------\n'
    for l in range(NUM_CLASSES):
        iou_per_class_str += 'class %s, IoU: %.3f \n' % (
            seg_label_to_cat[l] + ' ' * (12 - len(seg_label_to_cat[l])),
            total_correct_class[l] / float(total_iou_deno_class[l]))
    log_string(iou_per_class_str)
    log_string('test point avg class IoU: %f' % np.mean(IoU))
    log_string('test whole scene point avg class acc: %f' % (
        np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))))
    log_string('test whole scene point accuracy: %f' % (
                np.sum(total_correct_class) / float(np.sum(total_seen_class) + 1e-6)))
    log_string('test mIoU: %f' % np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6))) 
    log_string("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
    parser.add_argument('--encoder', type=str, default='foldingnet', help='encoder use')
    parser.add_argument('--num_points', type=int, default=2048, help='input point set size')
    parser.add_argument('--seg_model', type=str, default='snapshot/Semantic_segmentation_arch_1024_100/models/arch_training_data_at_epoch136.pkl', help='model path for the pre-trained part segmentation network')
    parser.add_argument('--ae_model', type=str, default='snapshot/Reconstruct_shapenet_foldingnet_1024/models/shapenetcorev2_best.pkl', help='model path')
    parser.add_argument('--dataset', type=str, default='arch', help='dataset: arch, arch_scene_2, shapenet_part, shapenet_core13, shapenet_core55, modelent40')
    parser.add_argument('--n_classes', type=int, default=10, help='part classes in all the catagories')
    parser.add_argument('--class_choice', type=str, default=None, help='choose the class to eva')
    parser.add_argument('--k', type=int, default=None)
    parser.add_argument('--feat_dims', type=int, default=1024)
    parser.add_argument('--num_dims', type=int, default=3, metavar='N',
                        help='Number of dims for feature ')
    parser.add_argument('--loss', type=str, default='ChamferLoss_m', choices=['ChamferLoss_m','ChamferLoss'],
                        help='reconstruction loss')
    parser.add_argument('--use_translate', action='store_true', help='Enables CUDA training')
    parser.add_argument('--visual', action='store_true', default=True, help='Whether visualize result [default: False]')
    parser.add_argument('--no_others', action='store_true', help='Enables CUDA training')
    parser.add_argument('--folder', '-f', help='path to data file')
    parser.add_argument('--dropout', action='store_true',
                        help='Enables dropout when training')

    opt = parser.parse_args()
    print(opt)

    USE_CUDA = True
    main(opt)
#
#
#      0=================================0
#      |    Project Name                 |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Implements point clouds augument functions, read/write, draw
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      YUWEI CAO - 2020/10/20 15:40 PM
#
#


# ----------------------------------------
# import packages
# ----------------------------------------
import numpy as np
import h5py
import sys
#from skimage import color

import os.path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pre_hdf5'))
#import common
# ----------------------------------------
# Point cloud argument
# ----------------------------------------

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


def rotate_pointcloud(pointcloud):
    theta = np.pi*2 * np.random.rand()
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    pointcloud[:,[0,2]] = pointcloud[:,[0,2]].dot(rotation_matrix) # random rotation (x,z)
    return pointcloud


'''
def rgb_t_hsv(points_colors):
    points_colors[:,0:3] /= 255.0
    color.rgb2hsv(points_colors)
    return points_colors
'''

def getshiftedpc(point_set):
    xyz_min = np.amin(point_set, axis=0)[0:3]
    return xyz_min


def norm_rgb(points_colors):
    points_colors[:,0:3] /= 255.0
    return points_colors


def shuffle_pointcloud(pointcloud, label):
    N = pointcloud.shape[0]
    order = np.arange(N)
    np.random.shuffle(order)
    pointcloud = pointcloud[order, :]
    lable = label[order]
    return pointcloud, label


def grouped_shuffle(inputs):
    for idx in range(len(inputs) - 1):
        assert (len(inputs[idx]) == len(inputs[idx + 1]))

    shuffle_indices = np.arange(inputs[0].shape[0])
    np.random.shuffle(shuffle_indices)
    outputs = []
    for idx in range(len(inputs)):
        outputs.append(inputs[idx][shuffle_indices, ...])
    return outputs

# ----------------------------------------
# Point cloud IO
# ----------------------------------------

#parse point cloud files
def load_txt(root, path):
    all_data = []
    all_label = []
    for filename in path:
        filename = os.path.join(root, filename)
        print("check filename: " + filename)
        data, label = common.scenetoblocks_wrapper(filename, num_point=2048) 
        all_data.append(data)
        all_label.append(label)
    return all_data, all_label


# ----------------------------------------------------------------
# Following are the helper functions to load save/load HDF5 files
# ----------------------------------------------------------------

# Write numpy array data and label to h5_filename
def save_h5_data_label_normal(h5_filename, data, label, normal, 
        data_dtype='float32', label_dtype='uint8', normal_dtype='float32'):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset(
            'data', data=data,
            compression='gzip', compression_opts=4,
            dtype=data_dtype)
    h5_fout.create_dataset(
            'normal', data=normal,
            compression='gzip', compression_opts=4,
            dtype=normal_dtype)
    h5_fout.create_dataset(
            'label', data=label,
            compression='gzip', compression_opts=1,
            dtype=label_dtype)
    h5_fout.close()


# Write numpy array data and label to h5_filename
def save_h5(h5_filename, data, label, data_dtype='uint8', label_dtype='uint8'):
    h5_fout = h5py.File(h5_filename, 'w')
    h5_fout.create_dataset(
            'data', data=data,
            compression='gzip', compression_opts=4,
            dtype=data_dtype)
    h5_fout.create_dataset(
            'label_seg', data=label,
            compression='gzip', compression_opts=1,
            dtype=label_dtype)
    h5_fout.close()


def load_sampled_h5_seg(filelist, dims=3, split='train'):
    points = []
    labels_seg = []
    folder = os.path.dirname(filelist)
    for line in open(filelist):
        print("Load file: " + str(line))
        data = h5py.File(os.path.join(folder, line.strip()), 'r')
        if split=='test':
            points.append(data['data'][:,:,0:dims+3].astype(np.float32))
        else:
            points.append(data['data'][:,:,3:dims+3].astype(np.float32))
                
        labels_seg.append(data['label_seg'][...].astype(np.int32))
        data.close()

    return (np.concatenate(points, axis=0),
            np.concatenate(labels_seg, axis=0))


def load_rotated_h5(filelist):
    points = []
    labels_angle = []
    folder = os.path.dirname(filelist)
    for line in open(filelist):
        print("Load file: " + str(line))
        data = h5py.File(os.path.join(folder, line.strip()), 'r')
        points.append(data['data'][...].astype(np.float32))
        labels_angle.append(data['label'][...].astype(np.int32))
        data.close()

    return (np.concatenate(points, axis=0),
            np.concatenate(labels_angle, axis=0))


def is_h5_list(filelist):
    return all([line.strip()[-3:] == '.h5' for line in open(filelist)])


def load_seg_list(filelist):
    folder = os.path.dirname(filelist)
    return [os.path.join(folder, line.strip()) for line in open(filelist)]


def load_seg(filelist):
    points = []
    labels = []
    point_nums = []
    labels_seg = []
    indices_split_to_full = []

    folder = os.path.dirname(filelist)
    for line in open(filelist):
        print("Load file: " + str(line))
        data = h5py.File(os.path.join(folder, line.strip()), 'r')
        points.append(data['data'][:,:,0:3].astype(np.float32))
        labels.append(data['label'][...].astype(np.int64))
        point_nums.append(data['data_num'][...].astype(np.int32))
        labels_seg.append(data['label_seg'][...].astype(np.int64))
        if 'indices_split_to_full' in data:
            indices_split_to_full.append(data['indices_split_to_full'][...].astype(np.int64))
        data.close()

    return (np.concatenate(points, axis=0),
            np.concatenate(labels, axis=0),
            np.concatenate(point_nums, axis=0),
            np.concatenate(labels_seg, axis=0),
            np.concatenate(indices_split_to_full, axis=0) if indices_split_to_full else None)


def balance_classes(labels):
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    counts_max = np.amax(counts)
    repeat_num_avg_unique = counts_max / counts
    repeat_num_avg = repeat_num_avg_unique[inverse]
    repeat_num_floor = np.floor(repeat_num_avg)
    repeat_num_probs = repeat_num_avg - repeat_num_floor
    repeat_num = repeat_num_floor + (np.random.rand(repeat_num_probs.shape[0]) < repeat_num_probs)

    return repeat_num.astype(np.int64)


# ----------------------------------------
# Draw Point cloud
# ----------------------------------------

import matplotlib.pyplot as plt
def pyplot_draw_point_cloud(points, output_filename):
    """ points is a Nx3 numpy array """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    #savefig(output_filename)
    

# ----------------------------------------
# Simple Point cloud and Volume Renderers
# ----------------------------------------

def draw_point_cloud(input_points, canvasSize=500, space=200, diameter=25,
                     xrot=0, yrot=0, zrot=0, switch_xyz=[0,1,2], normalize=True):
    """ Render point cloud to image with alpha channel.
        Input:
            points: Nx3 numpy array (+y is up direction)
        Output:
            gray image as numpy array of size canvasSizexcanvasSize
    """
    image = np.zeros((canvasSize, canvasSize))
    if input_points is None or input_points.shape[0] == 0:
        return image

    points = input_points[:, switch_xyz]
    M = euler2mat(zrot, yrot, xrot)
    points = (np.dot(M, points.transpose())).transpose()

    # Normalize the point cloud
    # We normalize scale to fit points in a unit sphere
    if normalize:
        centroid = np.mean(points, axis=0)
        points -= centroid
        furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2,axis=-1)))
        points /= furthest_distance

    # Pre-compute the Gaussian disk
    radius = (diameter-1)/2.0
    disk = np.zeros((diameter, diameter))
    for i in range(diameter):
        for j in range(diameter):
            if (i - radius) * (i-radius) + (j-radius) * (j-radius) <= radius * radius:
                disk[i, j] = np.exp((-(i-radius)**2 - (j-radius)**2)/(radius**2))
    mask = np.argwhere(disk > 0)
    dx = mask[:, 0]
    dy = mask[:, 1]
    dv = disk[disk > 0]
    
    # Order points by z-buffer
    zorder = np.argsort(points[:, 2])
    points = points[zorder, :]
    points[:, 2] = (points[:, 2] - np.min(points[:, 2])) / (np.max(points[:, 2] - np.min(points[:, 2])))
    max_depth = np.max(points[:, 2])
       
    for i in range(points.shape[0]):
        j = points.shape[0] - i - 1
        x = points[j, 0]
        y = points[j, 1]
        xc = canvasSize/2 + (x*space)
        yc = canvasSize/2 + (y*space)
        xc = int(np.round(xc))
        yc = int(np.round(yc))
        
        px = dx + xc
        py = dy + yc
        
        image[px, py] = image[px, py] * 0.7 + dv * (max_depth - points[j, 2]) * 0.3
    
    image = image / np.max(image)
    return image
    
def point_cloud_three_views(points):
    """ input points Nx3 numpy array (+y is up direction).
        return an numpy array gray image of size 500x1500. """ 
    # +y is up direction
    # xrot is azimuth
    # yrot is in-plane
    # zrot is elevation
    img1 = draw_point_cloud(points, zrot=110/180.0*np.pi, xrot=45/180.0*np.pi, yrot=0/180.0*np.pi)
    img2 = draw_point_cloud(points, zrot=70/180.0*np.pi, xrot=135/180.0*np.pi, yrot=0/180.0*np.pi)
    img3 = draw_point_cloud(points, zrot=180.0/180.0*np.pi, xrot=90/180.0*np.pi, yrot=0/180.0*np.pi)
    image_large = np.concatenate([img1, img2, img3], 1)
    return image_large


#from PIL import Image
def point_cloud_three_views_demo():
    """ Demo for draw_point_cloud function """
    points = read_ply('../third_party/mesh_sampling/piano.ply')
    im_array = point_cloud_three_views(points)
    #img = Image.fromarray(np.uint8(im_array*255.0))
    img.save('piano.jpg')
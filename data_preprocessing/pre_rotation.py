#
#
#      0=============================================================0
#      |    Project Name: Self-Supervised LoD3 Building Segmentation              |
#      0=============================================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Implements rotate point cloud
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      YUWEI CAO - 2021/3/29 10:36
#
#


# ----------------------------------------
# Import packages and constant
# ----------------------------------------
import math
import sys
import os.path
import numpy as np
import multiprocessing
import h5py
import argparse
import numpy as np
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pc_utils
sys.path.append(os.path.join(ROOT_DIR,'data_preprocessing'))
import arch_dataloader

def euler2mat(z=0, y=0, x=0):
    ''' Return matrix for rotations around z, y and x axes
    Uses the z, then y, then x convention above
    Parameters
    ----------
    z : scalar
       Rotation angle in radians around z-axis (performed first)
    y : scalar
       Rotation angle in radians around y-axis
    x : scalar
       Rotation angle in radians around x-axis (performed last)
    Returns
    -------
    M : array shape (3,3)
       Rotation matrix giving same rotation as for given angles
    Examples
    --------
    >>> zrot = 1.3 # radians
    >>> yrot = -0.1
    >>> xrot = 0.2
    >>> M = euler2mat(zrot, yrot, xrot)
    >>> M.shape == (3, 3)
    True
    The output rotation matrix is equal to the composition of the
    individual rotations
    >>> M1 = euler2mat(zrot)
    >>> M2 = euler2mat(0, yrot)
    >>> M3 = euler2mat(0, 0, xrot)
    >>> composed_M = np.dot(M3, np.dot(M2, M1))
    >>> np.allclose(M, composed_M)
    True
    You can specify rotations by named arguments
    >>> np.all(M3 == euler2mat(x=xrot))
    True
    When applying M to a vector, the vector should column vector to the
    right of M.  If the right hand side is a 2D array rather than a
    vector, then each column of the 2D array represents a vector.
    >>> vec = np.array([1, 0, 0]).reshape((3,1))
    >>> v2 = np.dot(M, vec)
    >>> vecs = np.array([[1, 0, 0],[0, 1, 0]]).T # giving 3x2 array
    >>> vecs2 = np.dot(M, vecs)
    Rotations are counter-clockwise.
    >>> zred = np.dot(euler2mat(z=np.pi/2), np.eye(3))
    >>> np.allclose(zred, [[0, -1, 0],[1, 0, 0], [0, 0, 1]])
    True
    >>> yred = np.dot(euler2mat(y=np.pi/2), np.eye(3))
    >>> np.allclose(yred, [[0, 0, 1],[0, 1, 0], [-1, 0, 0]])
    True
    >>> xred = np.dot(euler2mat(x=np.pi/2), np.eye(3))
    >>> np.allclose(xred, [[1, 0, 0],[0, 0, -1], [0, 1, 0]])
    True
    Notes
    -----
    The direction of rotation is given by the right-hand rule (orient
    the thumb of the right hand along the axis around which the rotation
    occurs, with the end of the thumb at the positive end of the axis;
    curl your fingers; the direction your fingers curl is the direction
    of rotation).  Therefore, the rotations are counterclockwise if
    looking along the axis of rotation from positive to negative.
    '''
    Ms = []
    if z:
        cosz = math.cos(z)
        sinz = math.sin(z)
        Ms.append(np.array(
                [[cosz, -sinz, 0],
                 [sinz, cosz, 0],
                 [0, 0, 1]]))
    if y:
        cosy = math.cos(y)
        siny = math.sin(y)
        Ms.append(np.array(
                [[cosy, 0, siny],
                 [0, 1, 0],
                 [-siny, 0, cosy]]))
    if x:
        cosx = math.cos(x)
        sinx = math.sin(x)
        Ms.append(np.array(
                [[1, 0, 0],
                 [0, cosx, -sinx],
                 [0, sinx, cosx]]))
    if Ms:
        return reduce(np.dot, Ms[::-1])
    return np.eye(3)


def mat2euler(M, cy_thresh=None):
    ''' Discover Euler angle vector from 3x3 matrix
    Uses the conventions above.
    Parameters
    ----------
    M : array-like, shape (3,3)
    cy_thresh : None or scalar, optional
       threshold below which to give up on straightforward arctan for
       estimating x rotation.  If None (default), estimate from
       precision of input.
    Returns
    -------
    z : scalar
    y : scalar
    x : scalar
       Rotations in radians around z, y, x axes, respectively
    Notes
    -----
    If there was no numerical error, the routine could be derived using
    Sympy expression for z then y then x rotation matrix, which is::
      [                       cos(y)*cos(z),                       -cos(y)*sin(z),         sin(y)],
      [cos(x)*sin(z) + cos(z)*sin(x)*sin(y), cos(x)*cos(z) - sin(x)*sin(y)*sin(z), -cos(y)*sin(x)],
      [sin(x)*sin(z) - cos(x)*cos(z)*sin(y), cos(z)*sin(x) + cos(x)*sin(y)*sin(z),  cos(x)*cos(y)]
    with the obvious derivations for z, y, and x
       z = atan2(-r12, r11)
       y = asin(r13)
       x = atan2(-r23, r33)
    Problems arise when cos(y) is close to zero, because both of::
       z = atan2(cos(y)*sin(z), cos(y)*cos(z))
       x = atan2(cos(y)*sin(x), cos(x)*cos(y))
    will be close to atan2(0, 0), and highly unstable.
    The ``cy`` fix for numerical instability below is from: *Graphics
    Gems IV*, Paul Heckbert (editor), Academic Press, 1994, ISBN:
    0123361559.  Specifically it comes from EulerAngles.c by Ken
    Shoemake, and deals with the case where cos(y) is close to zero:
    See: http://www.graphicsgems.org/
    The code appears to be licensed (from the website) as "can be used
    without restrictions".
    '''
    M = np.asarray(M)
    if cy_thresh is None:
        try:
            cy_thresh = np.finfo(M.dtype).eps * 4
        except ValueError:
            cy_thresh = _FLOAT_EPS_4
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
    # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
    cy = math.sqrt(r33*r33 + r23*r23)
    if cy > cy_thresh: # cos(y) not close to zero, standard form
        z = math.atan2(-r12,  r11) # atan2(cos(y)*sin(z), cos(y)*cos(z))
        y = math.atan2(r13,  cy) # atan2(sin(y), cy)
        x = math.atan2(-r23, r33) # atan2(cos(y)*sin(x), cos(x)*cos(y))
    else: # cos(y) (close to) zero, so x -> 0.0 (see above)
        # so r21 -> sin(z), r22 -> cos(z) and
        z = math.atan2(r21,  r22)
        y = math.atan2(r13,  cy) # atan2(sin(y), cy)
        x = 0.0
    return z, y, x

def apply_rotation(input_points, xrot=0, yrot=0, zrot=0, switch_xyz=[0,1,2], normalize=True):
    # Rotate the point cloud along up direction with certain angle.
    # Rotate in the order of x, y and then z.
    # Input:
    #      BxNx3 array, original batch of point clouds
    #      Ms is B x 3 x 3
    # Return:
    #      BxNx3 array, rotated batch of point clouds
    
    points = input_points[:, switch_xyz]
    Ms=euler2mat(xrot, yrot, zrot)
    rotated_points = (np.dot(Ms, points.transpose())).transpose() # B x N x 3
    
    if normalize:
        centroid = np.mean(rotated_points, axis=0)
        rotated_points -= centroid
        furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2,axis=-1)))
        rotated_points /= furthest_distance
    
    return rotated_points.reshape(batch_data.shape)


def rotate_point_cloud_by_angle_xyz(batch_data, angle_x=0, angle_y=0, angle_z=0):
    """ Rotate the point cloud along up direction with certain angle.
        Rotate in the order of x, y and then z.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = batch_data.reshape((-1, 3))
    
    cosval = np.cos(angle_x)
    sinval = np.sin(angle_x)
    rotation_matrix = np.array([[1, 0, 0],
                                [0, cosval, -sinval],
                                [0, sinval, cosval]])
    rotated_data = np.dot(rotated_data, rotation_matrix)

    cosval = np.cos(angle_y)
    sinval = np.sin(angle_y)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    rotated_data = np.dot(rotated_data, rotation_matrix)

    cosval = np.cos(angle_z)
    sinval = np.sin(angle_z)
    rotation_matrix = np.array([[cosval, -sinval, 0],
                                [sinval, cosval, 0],
                                [0, 0, 1]])
    rotated_data = np.dot(rotated_data, rotation_matrix)
    
    return rotated_data.reshape(batch_data.shape)


def rotate_point_by_label(batch_data, label):
    """ Rotate a batch of points by the label
        Input:
          batch_data: BxNx3 array
          label: B
        Return:
          BxNx3 array
        
    """
    rotate_func = rotate_point_cloud_by_angle_xyz
    batch_size = batch_data.shape[0]
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_size):
        shape_pc = batch_data[k, ...]
        l = label[k]
        if l==0:
            pass
        elif 1<=l<=3:
            shape_pc = rotate_func(shape_pc, angle_x=l*np.pi/2)
        elif 4<=l<=5:
            shape_pc = rotate_func(shape_pc, angle_z=(l*2-7)*np.pi/2)
            
        rotated_data[k, ...] = shape_pc

    return rotated_data


def _rotation_multiprocessing_worker(func, batch_data, label, return_dict, idx, *args):
    result = func(batch_data, label, *args)
    return_dict[idx] = result


def rotation_multiprocessing_wrapper(func, batch_data, label, *args, num_workers=8):
    """
    A wrapper for doing rotation using multiprocessing
    Input:
        func: a function for rotating on batch data, e.g. rotate_point_by_label
        batch_data: B*N*3 numpy array
        label: B length numpy array
    Returns:
        B*N*3 numpy array
    """
    batch_size = batch_data.shape[0] // num_workers
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    for i in range(num_workers):
        start_idx = i * batch_size
        end_idx = (i+1) * batch_size if i < num_workers - 1 else batch_data.shape[0]
        # print(f"[INFO] batch {i}, start index {start_idx}, end index {end_idx}")
        cur_data = batch_data[start_idx: end_idx]
        cur_label = label[start_idx: end_idx]
        p = multiprocessing.Process(target=_rotation_multiprocessing_worker,
                                    args=(func, cur_data, cur_label, return_dict, i, *args))
        p.start()
        jobs.append(p)
    for proc in jobs:
        proc.join()
    result = np.concatenate([return_dict[i] for i in range(num_workers)])
    # print("[INFO] rotated dimension:", result.shape)
    return result


def rotate_pc(current_data, NUM_CLASSES):
    current_data = np.repeat(current_data, NUM_CLASSES, axis=0) # (B,N,3) -> (B*NUM_CLASSES,N,3)
    print(f'Expanded data shape: {current_data.shape}')
    current_label = np.tile(np.arange(NUM_CLASSES), int(current_data.shape[0]//NUM_CLASSES))
    current_data = rotation_multiprocessing_wrapper(rotate_point_by_label, current_data, current_label)
    current_data, current_label = pc_utils.shuffle_pointcloud(current_data, np.squeeze(current_label))
    current_label = np.squeeze(current_label)
    
    return current_data, current_label

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', '-f', help='Path to data folder')
    parser.add_argument('--split', '-s', help='Path to data folder', default='train')
    parser.add_argument('--outpath', '-o', help='Path to output data folder')
    parser.add_argument('--max_point_num', '-m', help='Max point number of each sample', type=int, default=2048)
    parser.add_argument('--batch_size', '-b', help='Batch_size', type=int, default=32)
    parser.add_argument('--NUM_ANGLES', '-a', help='Rotation angle number', type=int, default=4)

    args = parser.parse_args()
    print(args)
    
    #read data
    root = os.path.join(ROOT_DIR, 'data', args.folder) if args.folder else os.path.join(ROOT_DIR, 'data', 'arch3_no_others_combined_5m_4096')
    max_point_num = args.max_point_num
    split= args.split
    train_filelist = os.path.join(root, split + '_data_files.txt')
    train_dataset = arch_dataloader.get_dataloader(filelist=train_filelist, num_points=max_point_num, num_dims=3, split=split, batch_size=args.batch_size, num_workers=8, drop_last=True)
    print("classifer set size: " + str(train_dataset.dataset.__len__()))

    #output settings
    output_dir = os.path.join(ROOT_DIR, 'data', args.outpath) if args.outpath else os.path.join(ROOT_DIR, 'data', 'rotated_%d_angle_%d' % (args.NUM_ANGLES, max_point_num))
    h5_file_size = 2112
    NUM_ANGLES = args.NUM_ANGLES
    batch_size = args.batch_size * args.NUM_ANGLES
    batch_num = h5_file_size / batch_size
    rotated_dataset = np.zeros((h5_file_size, max_point_num, 3))
    rotated_labelset = np.zeros((h5_file_size), dtype=np.int32)
    idx = 0
    idx_h5 = 0
    
    #rotate data 
    for iter, data in enumerate(train_dataset):
        input_points, _ = data
        rotated_data, rotated_label = rotate_pc(current_data=input_points, NUM_CLASSES=args.NUM_ANGLES)
        start = idx*batch_size
        end = (idx+1)*batch_size
        rotated_dataset[start:end, ...] = rotated_data
        rotated_labelset[start:end, ...] = rotated_label
        idx += 1
        if (idx == batch_num):
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            filename_h5 = os.path.join(output_dir, '%d.h5' % idx_h5)
            print('{}-Saving {}...'.format(datetime.now(), filename_h5))
            file = h5py.File(filename_h5, 'w')
            file.create_dataset('data', data=rotated_dataset[0:end, ...])
            file.create_dataset('label', data=rotated_labelset[0:end, ...])
            file.close()
            idx_h5 +=1
            idx = 0

        
if __name__ == '__main__':
    main()
    print('{}-Done.'.format(datetime.now()))
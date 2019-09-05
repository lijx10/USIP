import torch.utils.data as data

import random
import numbers
import os
import os.path
import numpy as np
import struct
import math

import torch
import torchvision
import matplotlib.pyplot as plt
import h5py

from data.augmentation import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from util import vis_tools

def make_dataset_kitti(root, mode, opt):
    if mode == 'train':
        seq_list = list(range(9))
    elif mode == 'test':
        seq_list = [9, 10]
    else:
        raise Exception('Invalid mode.')

    # filter or not
    np_folder = 'np_0.20_20480_r90_sn'
        
    accumulated_sample_num = 0
    sample_num_list = []
    accumulated_sample_num_list = []
    folder_list = []
    for seq in seq_list:
        folder = os.path.join(root, 'data_odometry_velodyne', 'numpy', '%02d'%seq, np_folder)
        folder_list.append(folder)
        
        sample_num = round(len(os.listdir(folder)))
        accumulated_sample_num += sample_num
        sample_num_list.append(sample_num)
        accumulated_sample_num_list.append(round(accumulated_sample_num))
        
    return seq_list, folder_list, sample_num_list, accumulated_sample_num_list


def transform_pc(pc, pose, pose_ref):
    '''
    transform pc to the reference frame
    '''
    pc_coord = pc[:, 0:3]  # Nx3
    pc_coord_homo = np.concatenate((pc_coord, np.ones((pc_coord.shape[0], 1))), axis=1)  # Nx4
    pc_coord_homo_transposed = np.transpose(pc_coord_homo)  # 4xN

    pc_coord_ref_transposed = np.dot(np.dot(np.linalg.inv(pose_ref), pose), pc_coord_homo_transposed)  # 4xN
    pc_coord_ref = np.transpose(pc_coord_ref_transposed)  # Nx4

    if pc.shape[1] == 3:
        pc_ref = pc_coord_ref[:, 0:3]
    else:
        pc_ref = np.concatenate((pc_coord_ref[:, 0:3], pc[:, 3:]), axis=1)

    return pc_ref


class FarthestSampler:
    def __init__(self):
        pass

    def calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=1)

    def sample(self, pts, k):
        farthest_pts = np.zeros((k, 3))
        farthest_pts[0] = pts[np.random.randint(len(pts))]
        distances = self.calc_distances(farthest_pts[0], pts)
        for i in range(1, k):
            farthest_pts[i] = pts[np.argmax(distances)]
            distances = np.minimum(distances, self.calc_distances(farthest_pts[i], pts))
        return farthest_pts


class KittiLoader(data.Dataset):
    def __init__(self, root, mode, opt):
        super(KittiLoader, self).__init__()
        self.root = root
        self.opt = opt
        self.mode = mode

        # farthest point sample
        self.farthest_sampler = FarthestSampler()

        self.seq_list, self.folder_list, self.sample_num_list, self.accumulated_sample_num_list = make_dataset_kitti(root, mode, opt)

    def __len__(self):
        return self.accumulated_sample_num_list[-1]

    def get_instance_unaugmented_np(self, index):
        # determine the sequence
        for i, accumulated_sample_num in enumerate(self.accumulated_sample_num_list):
            if index < accumulated_sample_num:
                break
        folder = self.folder_list[i]
        seq = self.seq_list[i]

        if i == 0:
            index_in_seq = index
        else:
            index_in_seq = index - self.accumulated_sample_num_list[i-1]
        pc_np_file = os.path.join(folder, '%06d.npy' % index_in_seq)
        pose_np_file = os.path.join(self.root, 'poses', '%02d'%seq, '%06d.npz'%index_in_seq)

        pc_np = np.load(pc_np_file)  # Nx8, x, y, z, nx, ny, nz, curvature, reflectance

        # radius threshold
        if self.opt.radius_threshold < 90:
            # camera coordinate
            pc_xz_norm_np = np.linalg.norm(pc_np[:, [0, 2]], axis=1)
            pc_radius_mask_np = pc_xz_norm_np <= self.opt.radius_threshold
            pc_np = pc_np[pc_radius_mask_np, :]

        # random sample
        if pc_np.shape[0] >= self.opt.input_pc_num:
            choice_idx = np.random.choice(pc_np.shape[0], self.opt.input_pc_num, replace=False)
        else:
            fix_idx = np.asarray(range(pc_np.shape[0]))
            while pc_np.shape[0] + fix_idx.shape[0] < self.opt.input_pc_num:
                fix_idx = np.concatenate((fix_idx, np.asarray(range(pc_np.shape[0]))), axis=0)
            random_idx = np.random.choice(pc_np.shape[0], self.opt.input_pc_num - fix_idx.shape[0], replace=False)
            choice_idx = np.concatenate((fix_idx, random_idx), axis=0)
        pc_np = pc_np[choice_idx, :]
        if self.opt.surface_normal_len == 1:
            sn_np = pc_np[:, pc_np.shape[1]-1:]  # Nx4, x, y, z, reflectance
        else:
            sn_np = pc_np[:, 3:3+self.opt.surface_normal_len]  # Nx5, nx, ny, nz, curvature, reflectance, \in [0, 0.99], mean 0.27
        pc_np = pc_np[:, 0:3]  # Nx3

        pose_np = np.load(pose_np_file)['pose']  # 4x4

        # get nodes, perform random sampling to reduce computation cost
        node_np = self.farthest_sampler.sample(pc_np[np.random.choice(pc_np.shape[0], int(self.opt.input_pc_num/3), replace=False)],
                                               self.opt.node_num)

        return pc_np, sn_np, node_np

    def augment(self, data_package_list):
        '''
        apply the same augmentation
        :param data_package_list: [(pc_np, sn_np, node_np), (...), ...]
        :return: augmented_package_list: [(pc_np, sn_np, node_np), (...), ...]
        '''
        B = len(data_package_list)

        # augmentation parameter / data
        # rotation ------
        y_angle = np.random.uniform() * 2 * np.pi
        angles_2d = [0, y_angle, 0]
        angles_3d = np.random.rand(3) * np.pi * 2
        angles_pertb = np.clip(0.06 * np.random.randn(3), -0.18, 0.18)
        # jitter ------
        sigma, clip = 0.04, 0.12
        N, C = data_package_list[0][0].shape
        jitter_pc = np.clip(sigma * np.random.randn(B, N, 3), -1 * clip, clip)
        sigma, clip = 0.01, 0.05
        jitter_sn = np.clip(sigma * np.random.randn(B, N, self.opt.surface_normal_len), -1 * clip, clip)  # nx, ny, nz, curvature, reflectance
        sigma, clip = 0.04, 0.12
        N, C = data_package_list[0][2].shape
        jitter_node = np.clip(sigma * np.random.randn(B, N, 3), -1 * clip, clip)
        # scale ------
        scale = np.random.uniform(low=0.9, high=1.1)
        # shift ------
        shift = np.random.uniform(-1, 1, (1, 3))

        # iterate over the list
        augmented_package_list = []
        for b, data_package in enumerate(data_package_list):
            pc_np, sn_np, node_np = data_package

            # rotation ------
            if self.opt.rot_horizontal:
                pc_np = atomic_rotate(pc_np, angles_2d)
                if self.opt.surface_normal_len >= 3:
                    sn_np[:, 0:3] = atomic_rotate(sn_np[:, 0:3], angles_2d)  # not applicable to reflectance
                node_np = atomic_rotate(node_np, angles_2d)
            if self.opt.rot_3d:
                pc_np = atomic_rotate(pc_np, angles_3d)
                if self.opt.surface_normal_len >= 3:
                    sn_np[:, 0:3] = atomic_rotate(sn_np[:, 0:3], angles_3d)  # not applicable to reflectance
                node_np = atomic_rotate(node_np, angles_3d)
            if self.opt.rot_perturbation:
                pc_np = atomic_rotate(pc_np, angles_pertb)
                if self.opt.surface_normal_len >= 3:
                    sn_np[:, 0:3] = atomic_rotate(sn_np[:, 0:3], angles_pertb)  # not applicable to reflectance
                node_np = atomic_rotate(node_np, angles_pertb)

            # jitter ------
            pc_np += jitter_pc[b]
            sn_np += jitter_sn[b]
            node_np += jitter_node[b]

            # scale
            pc_np = pc_np * scale
            # sn_np = sn_np * scale
            node_np = node_np * scale

            # shift
            if self.opt.translation_perturbation:
                pc_np += shift
                node_np += shift

            augmented_package_list.append([pc_np, sn_np, node_np])

        return augmented_package_list  # [(pc_np, sn_np, node_np), (...), ...]

    def __getitem__(self, index):
        src_pc_np, src_sn_np, src_node_np = self.get_instance_unaugmented_np(index)
        dst_pc_np, dst_sn_np, dst_node_np = self.get_instance_unaugmented_np(index)


        if self.mode == 'train':
            [[src_pc_np, src_sn_np, src_node_np], [dst_pc_np, dst_sn_np, dst_node_np]] = self.augment(
                [[src_pc_np, src_sn_np, src_node_np], [dst_pc_np, dst_sn_np, dst_node_np]])


        src_pc = torch.from_numpy(src_pc_np.transpose().astype(np.float32))  # 3xN
        src_sn = torch.from_numpy(src_sn_np.transpose().astype(np.float32))  # 3xN
        src_node = torch.from_numpy(src_node_np.transpose().astype(np.float32))  # 3xM
        dst_pc = torch.from_numpy(dst_pc_np.transpose().astype(np.float32))  # 3xN
        dst_sn = torch.from_numpy(dst_sn_np.transpose().astype(np.float32))  # 3xN
        dst_node = torch.from_numpy(dst_node_np.transpose().astype(np.float32))  # 3xM

        # === calculate dst data by getting a new node & node_knn_I === begin ===
        if self.opt.rot_3d:
            rot_type = '3d'
        elif self.opt.rot_horizontal:
            rot_type = '2d'
        else:
            rot_type = None
        if self.opt.rot_perturbation:
            rot_perturbation = True
        else:
            rot_perturbation = False
        dst_pc, dst_sn, dst_node, R, scale, shift = transform_pc_pytorch(dst_pc, dst_sn, dst_node,
                                                                         rot_type=rot_type, scale_thre=0, shift_thre=0.5,
                                                                         rot_perturbation=rot_perturbation)

        # debug
        # fig = plt.figure(figsize=(9, 9))
        # ax = Axes3D(fig)
        # ax = vis_tools.plot_pc(src_pc_np, color=[0, 0, 1], ax=ax)
        # ax = vis_tools.plot_pc(dst_pc_np, color=[1, 0, 0], ax=ax)
        # plt.show()

        return src_pc, src_sn, src_node, \
               dst_pc, dst_sn, dst_node, \
               R, scale, shift


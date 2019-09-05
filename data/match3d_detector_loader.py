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

import pickle

from data.augmentation import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from util import vis_tools


def read_txt_as_list(txt_file):
    with open(txt_file, 'r') as f:
        subset = [x.rstrip() for x in f.readlines()]
    return subset


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


def make_3dmatch_dataset(root, mode):
    dataset = []

    if mode == 'train':
        folder_list = read_txt_as_list(os.path.join(root, 'training_list.txt'))
    elif mode == 'test':
        folder_list = read_txt_as_list(os.path.join(root, 'testing_list.txt'))
    else:
        assert False

    # find all the samples
    for folder in folder_list:
        # find subfolder
        things_in_folder = os.listdir(os.path.join(root, folder));
        for subfolder in things_in_folder:
            if not os.path.isdir(os.path.join(root, folder, subfolder)):
                continue

            # find files
            things_in_subfolder = os.listdir(os.path.join(root, folder, subfolder))
            for file in things_in_subfolder:
                npy_file = os.path.join(root, folder, subfolder, file)
                dataset.append(npy_file)

    return dataset


class Match3DDetectorLoader:
    def __init__(self, root, mode, opt):
        super(Match3DDetectorLoader, self).__init__()
        self.root = root
        self.mode = mode
        self.opt = opt

        # farthest point sample
        self.farthest_sampler = FarthestSampler()

        self.dataset = make_3dmatch_dataset(root, mode)

    def __len__(self):
        return len(self.dataset)

    def get_instance_unaugmented_np(self, index):
        pc_np = np.load(os.path.join(self.dataset[index]))

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
        sn_np = pc_np[:, 3:3 + self.opt.surface_normal_len]  # Nx5, nx, ny, nz, curvature, reflectance, \in [0, 0.99], mean 0.27
        pc_np = pc_np[:, 0:3]  # Nx3, x, y, z

        return pc_np, sn_np

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
        angles_pertb = np.clip(0.12 * np.random.randn(3), -0.36, 0.36)
        # jitter ------
        sigma, clip = 0.010, 0.02
        N, C = data_package_list[0][0].shape
        jitter_pc = np.clip(sigma * np.random.randn(B, N, 3), -1 * clip, clip)
        sigma, clip = 0.010, 0.02
        jitter_sn = np.clip(sigma * np.random.randn(B, N, self.opt.surface_normal_len), -1 * clip, clip)  # nx, ny, nz, curvature, reflectance
        sigma, clip = 0.010, 0.02
        N, C = data_package_list[0][2].shape
        jitter_node = np.clip(sigma * np.random.randn(B, N, 3), -1 * clip, clip)
        # scale ------
        scale = np.random.uniform(low=0.8, high=1.2)
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
        # the dataset is already in CAM coordinate
        src_pc_np, src_sn_np = self.get_instance_unaugmented_np(index)
        dst_pc_np, dst_sn_np = self.get_instance_unaugmented_np(index)

        # get nodes, perform random sampling to reduce computation cost
        src_node_np = self.farthest_sampler.sample(
            src_pc_np[np.random.choice(src_pc_np.shape[0], int(self.opt.input_pc_num / 10), replace=False)],
            self.opt.node_num)
        dst_node_np = self.farthest_sampler.sample(
            dst_pc_np[np.random.choice(dst_pc_np.shape[0], int(self.opt.input_pc_num / 10), replace=False)],
            self.opt.node_num)

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
                                                                         rot_type=rot_type, scale_thre=0.1, shift_thre=0.5,
                                                                         rot_perturbation=rot_perturbation)

        # # debug
        # fig = plt.figure(figsize=(9, 9))
        # ax = Axes3D(fig)
        # ax = vis_tools.plot_pc(src_pc_np, color=[0, 0, 1], ax=ax)
        # ax = vis_tools.plot_pc(dst_pc_np, color=[1, 0, 0], ax=ax)
        # plt.show()

        return src_pc, src_sn, src_node, \
               dst_pc, dst_sn, dst_node, \
               R, scale, shift
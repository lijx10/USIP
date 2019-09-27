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


def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


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


class RedwoodLoader(data.Dataset):
    def __init__(self, root, opt):
        super(RedwoodLoader, self).__init__()
        self.root = root
        self.opt = opt

        # farthest point sample
        self.farthest_sampler = FarthestSampler()

        self.redwood_dataset = {'livingroom1': 57, 'livingroom2': 47, 'office1': 53, 'office2': 50}
        self.scene_name_list = ['livingroom1', 'livingroom2', 'office1', 'office2']
        self.scene_frame_num_acc = [57, 104, 157, 207]

    def __len__(self):
        dataset_len = 0
        for folder in os.listdir(self.root):
            dataset_len += len(os.listdir(os.path.join(self.root, folder)))
        assert dataset_len == 207
        return dataset_len

    def get_instance_unaugmented_np(self, index):
        # determine which scene
        for scene_idx in range(len(self.scene_frame_num_acc)):
            if index < self.scene_frame_num_acc[scene_idx]:
                break
        if scene_idx == 0:
            frame_idx = index
        else:
            frame_idx = index - self.scene_frame_num_acc[scene_idx-1]
        pc_np = np.load(os.path.join(self.root, self.scene_name_list[scene_idx], '%d.npy' % frame_idx))

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

        return pc_np, sn_np, scene_idx, frame_idx

    def __getitem__(self, index):
        # the dataset is already in CAM coordinate
        anc_pc_np, anc_sn_np, scene_idx, frame_idx = self.get_instance_unaugmented_np(index)

        # get nodes, perform random sampling to reduce computation cost
        anc_node_np = self.farthest_sampler.sample(
            anc_pc_np[np.random.choice(anc_pc_np.shape[0], int(self.opt.input_pc_num / 2), replace=False)],
            self.opt.node_num)

        # fig = plt.figure()
        # ax = Axes3D(fig)
        # ax.scatter(pos_pc_np[:, 0].tolist(), pos_pc_np[:, 1].tolist(), pos_pc_np[:, 2].tolist(), s=5, c=[1, 0, 0])
        # ax.scatter(anc_pc_np[:, 0].tolist(),
        #            anc_pc_np[:, 1].tolist(),
        #            anc_pc_np[:, 2].tolist(),
        #            s=5, c=[0, 0, 1])
        # axisEqual3D(ax)
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('z')
        # plt.show()

        anc_pc = torch.from_numpy(anc_pc_np.transpose().astype(np.float32))  # 3xN
        anc_sn = torch.from_numpy(anc_sn_np.transpose().astype(np.float32))  # 3xN
        anc_node = torch.from_numpy(anc_node_np.transpose().astype(np.float32))  # 3xM

        return anc_pc, anc_sn, anc_node, \
               scene_idx, frame_idx


if __name__ == '__main__':
    from scenenn import options_detector
    opt = options_detector.Options().parse()  # set CUDA_VISIBLE_DEVICES before import torch

    trainset = RedwoodLoader('/ssd/dataset/redwood/numpy_gt_normal', opt)
    dataset_size = len(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=False,
                                              num_workers=opt.nThreads, drop_last=True, pin_memory=True)
    print('#training point clouds = %d' % len(trainset))


    print('scene %d, frame %d' % (trainset[0][3], trainset[0][4]))
    print('scene %d, frame %d' % (trainset[56][3], trainset[56][4]))
    print('scene %d, frame %d' % (trainset[57][3], trainset[57][4]))
    print('scene %d, frame %d' % (trainset[104][3], trainset[104][4]))
    print('scene %d, frame %d' % (trainset[206][3], trainset[206][4]))

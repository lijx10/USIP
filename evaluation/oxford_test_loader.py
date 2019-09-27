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

import pickle


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


def make_dataset_oxford_test():
    return list(range(828))


class OxfordTestLoader(data.Dataset):
    def __init__(self, root, opt):
        super(OxfordTestLoader, self).__init__()
        self.root = root
        self.opt = opt

        # farthest point sample
        self.farthest_sampler = FarthestSampler()

        self.is_filter_str = '_nofilter'

        self.dataset = make_dataset_oxford_test()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        '''

        :param index:
        :return: anc_pc, anc_sn, anc_node, anc_seq, anc_idx
        '''

        anc_idx = index

        # ===== load numpy array =====
        pc_np = np.load(os.path.join(self.root, 'test_models_20k_np' + self.is_filter_str, '%d.npy' % anc_idx))  # Nx8

        # random choice
        assert self.opt.surface_normal_len == 4
        choice_idx = np.random.choice(pc_np.shape[0], self.opt.input_pc_num, replace=False)
        pc_np = pc_np[choice_idx, :]
        sn_np = pc_np[:,
                3:3 + self.opt.surface_normal_len]  # Nx5, nx, ny, nz, curvature, reflectance, \in [0, 0.99], mean 0.27
        pc_np = pc_np[:, 0:3]  # Nx3

        # get nodes, perform random sampling to reduce computation cost
        node_np = self.farthest_sampler.sample(
            pc_np[np.random.choice(pc_np.shape[0], int(self.opt.input_pc_num / 3), replace=False)],
            self.opt.node_num)

        pc_np, sn_np, node_np = coordinate_ENU_to_cam(pc_np, sn_np, node_np)
        # ===== load numpy array =====

        # convert to torch tensor
        anc_pc = torch.from_numpy(pc_np.transpose().astype(np.float32))  # 3xN
        anc_sn = torch.from_numpy(sn_np.transpose().astype(np.float32))  # 4xN
        anc_node = torch.from_numpy(node_np.transpose().astype(np.float32))  # 3xM

        return anc_pc, anc_sn, anc_node, anc_idx


if __name__ == '__main__':
    import oxford.options_detector
    import oxford.options_descriptor

    opt_detector = oxford.options_detector.Options().parse()

    oxford_testset = OxfordTestLoader('/ssd/dataset/oxford', opt_detector)
    print(len(oxford_testset))
    anc_pc, anc_sn, anc_node, anc_idx = oxford_testset[537]
    print(anc_idx)

    testloader = torch.utils.data.DataLoader(oxford_testset, batch_size=opt_detector.batch_size,
                                             shuffle=False, num_workers=opt_detector.nThreads, pin_memory=False)
    print(len(testloader))
    loader_iter = iter(testloader)
    a = loader_iter.next()
    b = loader_iter.next()
    print(a)
    print(b)
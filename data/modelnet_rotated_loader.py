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



def make_dataset_modelnet40(root):
    dataset = []
    for i in range(2468):
        folder = 'original'
        item = (os.path.join(root, folder, '%d.npy' % i), i, 0)
        dataset.append(item)
    for i in range(2468):
        folder = 'rotated'
        item = (os.path.join(root, folder, '%d.npy' % i), i, 1)
        dataset.append(item)

    return dataset


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


class ModelNet_Rotated_Loader(data.Dataset):
    def __init__(self, root, opt):
        super(ModelNet_Rotated_Loader, self).__init__()
        self.root = root
        self.opt = opt

        self.dataset = make_dataset_modelnet40(self.root)

        # farthest point sample
        self.fathest_sampler = FarthestSampler()

    def __len__(self):
        return len(self.dataset)

    def get_instance_unaugmented_np(self, index):
        pc_np_file, idx, pc_type_id = self.dataset[index]

        data = np.load(pc_np_file)
        data = data[np.random.choice(data.shape[0], self.opt.input_pc_num, replace=False), :]

        pc_np = data[:, 0:3]  # Nx3
        sn_np = data[:, 3:6]  # Nx3

        node_np = self.fathest_sampler.sample(
            pc_np[np.random.choice(pc_np.shape[0], int(self.opt.input_pc_num / 4), replace=False)],
            self.opt.node_num)

        return pc_np, sn_np, node_np, idx, pc_type_id

    def __getitem__(self, index):
        pc_np, sn_np, node_np, idx, pc_type_id = self.get_instance_unaugmented_np(index)

        pc = torch.from_numpy(pc_np.transpose().astype(np.float32))  # 3xN
        sn = torch.from_numpy(sn_np.transpose().astype(np.float32))  # 3xN
        node = torch.from_numpy(node_np.transpose().astype(np.float32))  # 3xM

        return pc, sn, node, idx, pc_type_id

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


def load_kitti_test_gt_txt(txt_root, seq):
    '''

    :param txt_root:
    :param seq
    :return: [{anc_idx: *, pos_idx: *, seq: *}]
    '''
    dataset = []
    with open(os.path.join(txt_root, '%02d'%seq, 'groundtruths.txt'), 'r') as f:
        lines_list = f.readlines()
        for i, line_str in enumerate(lines_list):
            if i == 0:
                # skip the header line
                continue
            line_splitted = line_str.split()
            anc_idx = int(line_splitted[0])
            pos_idx = int(line_splitted[1])

            # search for existence
            anc_idx_is_exist = False
            pos_idx_is_exist = False
            for tmp_data in dataset:
                if tmp_data['anc_idx'] == anc_idx:
                    anc_idx_is_exist = True
                if tmp_data['anc_idx'] == pos_idx:
                    pos_idx_is_exist = True

            if anc_idx_is_exist is False:
                data = {'seq': seq, 'anc_idx': anc_idx, 'pos_idx': pos_idx}
                dataset.append(data)
            if pos_idx_is_exist is False:
                data = {'seq': seq, 'anc_idx': pos_idx, 'pos_idx': anc_idx}
                dataset.append(data)

    return dataset


def make_kitti_test_dataset(txt_root):
    folder_list = os.listdir(txt_root)
    folder_list.sort()
    folder_int_list = [int(x) for x in folder_list]

    dataset = []
    for seq in folder_int_list:
        dataset += (load_kitti_test_gt_txt(txt_root, seq))
    # print(dataset)
    # print(len(dataset))
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


class KittiTestLoader(data.Dataset):
    def __init__(self, txt_root, numpy_root, opt):
        super(KittiTestLoader, self).__init__()
        self.txt_root = txt_root
        self.numpy_root = numpy_root
        self.opt = opt

        # farthest point sample
        self.farthest_sampler = FarthestSampler()

        self.dataset = make_kitti_test_dataset(txt_root)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        '''

        :param index:
        :return: anc_pc, anc_sn, anc_node, anc_seq, anc_idx
        '''

        seq = self.dataset[index]['seq']
        anc_idx = self.dataset[index]['anc_idx']

        # ===== load numpy array =====
        pc_np_file = os.path.join(self.numpy_root, '%02d' % seq, 'np_0.20_20480_r90_sn', '%06d.npy' % anc_idx)

        # random choice
        assert self.opt.surface_normal_len == 4
        pc_np = np.load(pc_np_file)  # Nx8, x, y, z, sn_x, sn_y, sn_z, curvature, reflectance
        choice_idx = np.random.choice(pc_np.shape[0], self.opt.input_pc_num, replace=False)
        pc_np = pc_np[choice_idx, :]
        sn_np = pc_np[:,
                3:3 + self.opt.surface_normal_len]  # Nx5, nx, ny, nz, curvature, reflectance, \in [0, 0.99], mean 0.27
        pc_np = pc_np[:, 0:3]  # Nx3

        # get nodes, perform random sampling to reduce computation cost
        node_np = self.farthest_sampler.sample(
            pc_np[np.random.choice(pc_np.shape[0], int(self.opt.input_pc_num / 4), replace=False)],
            self.opt.node_num)
        # ===== load numpy array =====

        # convert to torch tensor
        anc_pc = torch.from_numpy(pc_np.transpose().astype(np.float32))  # 3xN
        anc_sn = torch.from_numpy(sn_np.transpose().astype(np.float32))  # 4xN
        anc_node = torch.from_numpy(node_np.transpose().astype(np.float32))  # 3xM

        return anc_pc, anc_sn, anc_node, seq, anc_idx


if __name__ == '__main__':
    import kitti.options_detector
    import kitti.options_descriptor

    opt_detector = kitti.options_detector.Options().parse()

    kitti_testset = KittiTestLoader('/ssd/dataset/kitti-reg-test', '/ssd/dataset/odometry/data_odometry_velodyne/numpy', opt_detector)
    print(len(kitti_testset))
    anc_pc, anc_sn, anc_node, seq, anc_idx = kitti_testset[537]
    print(seq)
    print(anc_idx)

    testloader = torch.utils.data.DataLoader(kitti_testset, batch_size=opt_detector.batch_size,
                                             shuffle=False, num_workers=opt_detector.nThreads, pin_memory=False)
    print(len(testloader))
    loader_iter = iter(testloader)
    a = loader_iter.next()
    b = loader_iter.next()
    print(a)
    print(b)



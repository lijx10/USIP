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


# Read numpy array data and label from h5_filename
def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)


def make_dataset_modelnet40_10k(root, mode, opt):
    dataset = []
    rows = round(math.sqrt(opt.node_num))
    cols = rows

    f = open(os.path.join(root, 'modelnet%d_shape_names.txt' % opt.classes))
    shape_list = [str.rstrip() for str in f.readlines()]
    f.close()

    if 'train' == mode:
        f = open(os.path.join(root, 'modelnet%d_train.txt' % opt.classes), 'r')
        lines = [str.rstrip() for str in f.readlines()]
        f.close()
    elif 'test' == mode:
        f = open(os.path.join(root, 'modelnet%d_test.txt' % opt.classes), 'r')
        lines = [str.rstrip() for str in f.readlines()]
        f.close()
    else:
        raise Exception('Network mode error.')

    for i, name in enumerate(lines):
        # locate the folder name
        folder = name[0:-5]
        file_name = name

        # get the label
        label = shape_list.index(folder)

        # som node locations
        som_nodes_folder = '%dx%d_som_nodes' % (rows, cols)

        item = (os.path.join(root, folder, file_name + '.npy'),
                label,
                os.path.join(root, som_nodes_folder, folder, file_name + '.npy'))
        dataset.append(item)

    return dataset


def make_dataset_shrec2016(root, mode, opt):
    rows = round(math.sqrt(opt.node_num))
    cols = rows
    dataset = []

    # load category txt
    f = open(os.path.join(root, 'category.txt'), 'r')
    category_list = [str.rstrip() for str in f.readlines()]
    f.close()

    if 'train' == mode:
        f = open(os.path.join(root, 'train.txt'), 'r')
        lines = [str.rstrip() for str in f.readlines()]
        f.close()
    elif 'val' == mode:
        f = open(os.path.join(root, 'val.txt'), 'r')
        lines = [str.rstrip() for str in f.readlines()]
        f.close()
    elif 'test' == mode:
        f = open(os.path.join(root, 'test.txt'), 'r')
        lines = [str.rstrip() for str in f.readlines()]
        f.close()
    else:
        raise Exception('Network mode error.')

    if 'train' == mode or 'val' == mode:
        for line in lines:
            line_split = [x.strip() for x in line.split(',')]
            name, category = line_split[0], line_split[1]

            npz_file = os.path.join(root, '%dx%d' % (rows, cols), mode, 'model_' + name + '.npz')
            try:
                category = category_list.index(category)
            except ValueError:
                continue

            item = (npz_file, category)
            dataset.append(item)
    elif 'test' == mode:
        for line in lines:
            name, category = line, int(line) % 55
            npz_file = os.path.join(root, '%dx%d' % (rows, cols), mode, 'model_' + name + '.npz')

            item = (npz_file, category)
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


class ModelNet_Shrec_Loader(data.Dataset):
    def __init__(self, root, mode, opt):
        super(ModelNet_Shrec_Loader, self).__init__()
        self.root = root
        self.opt = opt
        self.mode = mode

        if self.opt.dataset == 'modelnet':
            self.dataset = make_dataset_modelnet40_10k(self.root, mode, opt)
        elif self.opt.dataset == 'shrec':
            self.dataset = make_dataset_shrec2016(self.root, mode, opt)
        else:
            raise Exception('Dataset incorrect.')

        # farthest point sample
        self.fathest_sampler = FarthestSampler()

    def __len__(self):
        return len(self.dataset)

    def get_instance_unaugmented_np(self, index):
        if self.opt.dataset == 'modelnet':
            pc_np_file, class_id, som_node_np_file = self.dataset[index]

            data = np.load(pc_np_file)
            data = data[np.random.choice(data.shape[0], self.opt.input_pc_num, replace=False), :]

            pc_np = data[:, 0:3]  # Nx3
            sn_np = data[:, 3:6]  # Nx3

        elif self.opt.dataset == 'shrec':
            npz_file, class_id = self.dataset[index]
            data = np.load(npz_file)

            pc_np = data['pc']
            sn_np = data['sn']

            # random choice
            choice_idx = np.random.choice(pc_np.shape[0], self.opt.input_pc_num, replace=False)
            pc_np = pc_np[choice_idx, :]
            sn_np = sn_np[choice_idx, :]
        else:
            raise Exception('Dataset incorrect.')

        node_np = self.fathest_sampler.sample(
            pc_np[np.random.choice(pc_np.shape[0], int(self.opt.input_pc_num / 4), replace=False)],
            self.opt.node_num)

        return pc_np, sn_np, node_np, class_id

    def augment(self, data_package_list):
        '''
        apply the same augmentation
        :param data_package_list: [(pc_np, sn_np, node_np), (...), ...]
        :return: augmented_package_list: [(pc_np, sn_np, node_np), (...), ...]
        '''
        # augmentation parameter / data
        # rotation ------
        y_angle = np.random.uniform() * 2 * np.pi
        angles_2d = [0, y_angle, 0]
        angles_3d = np.random.rand(3) * np.pi * 2
        angles_pertb = np.clip(0.06 * np.random.randn(3), -0.18, 0.18)
        # jitter ------
        sigma, clip = 0.01, 0.05
        N, C = data_package_list[0][0].shape
        jitter_pc = np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
        jitter_sn = np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
        sigma, clip = 0.04, 0.1
        N, C = data_package_list[0][2].shape
        jitter_node = np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
        # scale ------
        scale = np.random.uniform(low=0.8, high=1.2)
        # shift ------
        shift = np.random.uniform(-0.1, 0.1, (1, 3))

        # iterate over the list
        augmented_package_list = []
        for data_package in data_package_list:
            pc_np, sn_np, node_np = data_package

            # rotation ------
            if self.opt.rot_horizontal:
                pc_np = atomic_rotate(pc_np, angles_2d)
                sn_np = atomic_rotate(sn_np, angles_2d)
                node_np = atomic_rotate(node_np, angles_2d)
            if self.opt.rot_3d:
                pc_np = atomic_rotate(pc_np, angles_3d)
                sn_np = atomic_rotate(sn_np, angles_3d)
                node_np = atomic_rotate(node_np, angles_3d)
            if self.opt.rot_perturbation:
                pc_np = atomic_rotate(pc_np, angles_pertb)
                sn_np = atomic_rotate(sn_np, angles_pertb)
                node_np = atomic_rotate(node_np, angles_pertb)

            # jitter ------
            pc_np += jitter_pc
            sn_np += jitter_sn
            node_np += jitter_node

            # scale
            pc_np = pc_np * scale
            sn_np = sn_np * scale
            node_np = node_np * scale

            # shift
            if self.opt.translation_perturbation:
                pc_np += shift
                node_np += shift

            augmented_package_list.append([pc_np, sn_np, node_np])

        return augmented_package_list  # [(pc_np, sn_np, node_np), (...), ...]

    def __getitem__(self, index):
        src_pc_np, src_sn_np, src_node_np, src_label = self.get_instance_unaugmented_np(index)
        dst_pc_np, dst_sn_np, dst_node_np, dst_label = self.get_instance_unaugmented_np(index)

        # debug
        # dst_pc_np = src_pc_np

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
        dst_pc, dst_sn, dst_node, R, scale, shift = transform_pc_pytorch(dst_pc, dst_sn, dst_node,
                                                                         rot_type=rot_type, shift_thre=0.5)

        return src_pc, src_sn, src_node, \
               dst_pc, dst_sn, dst_node, \
               R, scale, shift


if __name__ == "__main__":
    # dataset = make_dataset_modelnet40('/ssd/dataset/modelnet40_ply_hdf5_2048/', True)
    # print(len(dataset))
    # print(dataset[0])

    class VirtualOpt():
        def __init__(self):
            self.load_all_data = False
            self.input_pc_num = 5000
            self.batch_size = 8
            self.dataset = '10k'
            self.node_num = 64
            self.classes = 10
            self.node_knn_k_1 = 9


    opt = VirtualOpt()
    trainset = ModelNet_Shrec_Loader('/ssd/dataset/modelnet40-normal_numpy/', 'train', opt)
    print('---')
    print(len(trainset))
    print(trainset[0])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=False, num_workers=4)

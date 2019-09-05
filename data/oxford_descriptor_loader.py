import torch.utils.data as data
from torch._six import string_classes, int_classes, FileNotFoundError
import collections

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


def make_dataset_oxford_train(root):
    f = open(os.path.join(root, 'train_relative.txt'), 'r')
    lines_list = f.readlines()

    dataset = []
    for i, line_str in enumerate(lines_list):
        # convert each line to a dict
        line_splitted_list = line_str.split('|')
        try:
            assert len(line_splitted_list) == 3
        except Exception:
            print('Invalid line.')
            print(i)
            print(line_splitted_list)
            continue

        file_name = line_splitted_list[0].strip()
        positive_lines = list(map(int, line_splitted_list[1].split()))
        non_negative_lines = list(map(int, line_splitted_list[2].split()))

        #         print(file_name)
        #         print(positive_lines)
        #         print(non_negative_lines)

        data = {'file': file_name, 'pos_line_list': positive_lines, 'nonneg_line_list': non_negative_lines + positive_lines}
        dataset.append(data)

    f.close()
    return dataset  # [{'file', 'pos_line_list', 'nonneg_line_list'}]


def make_dataset_oxford_test(root):
    with open(os.path.join(root, 'test_models_20k_np_nofilter', 'gt_descriptor_testing.pkl'), 'rb') as f:
        return pickle.load(f)  # [['anc_idx', 'pos_idx_list']]


class OxfordDescriptorLoader(data.Dataset):
    def __init__(self, root, mode, opt):
        super(OxfordDescriptorLoader, self).__init__()
        self.root = root
        self.opt = opt
        self.mode = mode

        self.is_filter_str = '_nofilter'

        # farthest point sample
        self.farthest_sampler = FarthestSampler()

        if mode == 'train':
            self.dataset = make_dataset_oxford_train(root)
        else:
            self.dataset = make_dataset_oxford_test(root)

    def __len__(self):
        return len(self.dataset)

    def get_instance_unaugmented_np(self, index):
        if self.mode == 'train':
            filename = self.dataset[index]['file']
            pc_np = np.load(os.path.join(self.root, 'train_np'+self.is_filter_str, filename[0:-3] + 'npy'))  # Nx8
        else:
            anc_idx = self.dataset[index]['anc_idx']
            pc_np = np.load(os.path.join(self.root, 'test_models_20k_np'+self.is_filter_str, '%d.npy' % anc_idx))  # Nx8

        choice_idx = np.random.choice(pc_np.shape[0], self.opt.input_pc_num, replace=False)
        pc_np = pc_np[choice_idx, :]
        sn_np = pc_np[:,
                3:3 + self.opt.surface_normal_len]  # Nx5, nx, ny, nz, curvature, reflectance, \in [0, 0.99], mean 0.27
        pc_np = pc_np[:, 0:3]  # Nx3, x, y, z

        return pc_np, sn_np  # , keypoints_np, sigmas_np

    def get_instance_pos_unaugmented_np(self, index):
        if self.mode == 'train':
            pos_line_list = self.dataset[index]['pos_line_list']
            rand_idx = random.randint(0, len(pos_line_list)-1)  # a<=x<=b
            pos_line = pos_line_list[rand_idx]
            filename = self.dataset[pos_line]['file']
            pc_np = np.load(os.path.join(self.root, 'train_np'+self.is_filter_str, filename[0:-3] + 'npy'))  # Nx8
        else:
            pos_idx_list = self.dataset[index]['pos_idx_list']
            rand_idx = random.randint(0, len(pos_idx_list) - 1)  # a<=x<=b
            pos_idx = pos_idx_list[rand_idx]
            pc_np = np.load(os.path.join(self.root, 'test_models_20k_np'+self.is_filter_str, '%d.npy' % pos_idx))  # Nx8

        choice_idx = np.random.choice(pc_np.shape[0], self.opt.input_pc_num, replace=False)
        pc_np = pc_np[choice_idx, :]
        sn_np = pc_np[:,
                3:3 + self.opt.surface_normal_len]  # Nx5, nx, ny, nz, curvature, reflectance, \in [0, 0.99], mean 0.27
        pc_np = pc_np[:, 0:3]  # Nx3, x, y, z

        return pc_np, sn_np  # , keypoints_np, sigmas_np

    def get_nonneg_list(self, index):
        assert type(index) is int
        if self.mode == 'train':
            nonneg_line_list = self.dataset[index]['nonneg_line_list']
            return nonneg_line_list
        else:
            pos_idx_list = self.dataset[index]['pos_idx_list']
            return pos_idx_list

    def augment(self, data_package_list):
        '''
        apply the same augmentation
        :param data_package_list: [(pc_np, sn_np, node_np), (...), ...]
        :return: augmented_package_list: [(pc_np, sn_np, node_np), (...), ...]
        '''
        B = len(data_package_list)

        # augmentation parameter / data
        # scale ------
        # scale = np.random.uniform(low=0.9, high=1.1)
        scale = 1.0

        # iterate over the list
        augmented_package_list = []
        for b, data_package in enumerate(data_package_list):
            # rotation ------
            y_angle = np.random.uniform() * 2 * np.pi
            angles_2d = [0, y_angle, 0]
            angles_3d = np.random.rand(3) * np.pi * 2
            angles_pertb = np.clip(0.06 * np.random.randn(3), -0.18, 0.18)

            # jitter ------
            sigma, clip = 0.04, 0.12
            N, C = data_package_list[b][0].shape
            jitter_pc = np.clip(sigma * np.random.randn(N, 3), -1 * clip, clip)
            sigma, clip = 0.01, 0.05
            jitter_sn = np.clip(sigma * np.random.randn(N, self.opt.surface_normal_len), -1 * clip,
                                clip)  # nx, ny, nz, curvature, reflectance
            sigma, clip = 0.04, 0.12
            N, C = data_package_list[b][2].shape
            jitter_node = np.clip(sigma * np.random.randn(N, 3), -1 * clip, clip)

            # shift ------
            shift = np.random.uniform(-1, 1, (1, 3))

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
            pc_np += jitter_pc
            sn_np += jitter_sn
            node_np += jitter_node

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

    def mine_negative_sample(self, index_batch):
        B = index_batch.size()[0]
        neg_idx = torch.zeros(B, dtype=torch.int64)

        if self.mode == 'train':
            # each anchor sample
            for i in range(B):
                index = index_batch[i].item()
                nonneg_line_list = set(self.get_nonneg_list(index))

                # iterate over the batch to find negative candidates
                negative_candidates = []
                for j in range(B):
                    if j == i:
                        continue
                    j_line = index_batch[j].item()
                    if j_line not in nonneg_line_list:
                        negative_candidates.append(j)
                # sample one negative sample
                if len(negative_candidates) == 0:
                    print('!!!!!!!!!!!!Fail to mine negative sample!!!!!!!!!!!!')
                else:
                    rand_idx = random.randint(0, len(negative_candidates)-1)
                    neg_idx[i] = negative_candidates[rand_idx]

        elif self.mode == 'test':
            for i in range(B):
                # each anchor sample
                for i in range(B):
                    index = index_batch[i].item()
                    nonneg_idx_list = set(self.get_nonneg_list(index))

                    # iterate over the batch to find negative candidates
                    negative_candidates = []
                    for j in range(B):
                        if j == i:
                            continue
                        j_anc_idx = self.dataset[index_batch[j].item()]['anc_idx']
                        if j_anc_idx not in nonneg_idx_list:
                            negative_candidates.append(j)
                    # sample one negative sample
                    if len(negative_candidates) == 0:
                        print('!!!!!!!!!!!!Fail to mine negative sample!!!!!!!!!!!!')
                    else:
                        rand_idx = random.randint(0, len(negative_candidates) - 1)
                        neg_idx[i] = negative_candidates[rand_idx]

        else:
            raise Exception('Incorrect dataloader mode')

        return neg_idx

    def __getitem__(self, index):
        anc_pc_np, anc_sn_np = self.get_instance_unaugmented_np(index)
        pos_pc_np, pos_sn_np = self.get_instance_pos_unaugmented_np(index)

        # get nodes, perform random sampling to reduce computation cost
        anc_node_np = self.farthest_sampler.sample(
            anc_pc_np[np.random.choice(anc_pc_np.shape[0], int(self.opt.input_pc_num / 4), replace=False)],
            self.opt.node_num)
        pos_node_np = self.farthest_sampler.sample(
            pos_pc_np[np.random.choice(pos_pc_np.shape[0], int(self.opt.input_pc_num / 4), replace=False)],
            self.opt.node_num)

        anc_pc_np, anc_sn_np, anc_node_np = coordinate_ENU_to_cam(anc_pc_np, anc_sn_np, anc_node_np)
        pos_pc_np, pos_sn_np, pos_node_np = coordinate_ENU_to_cam(pos_pc_np, pos_sn_np, pos_node_np)

        if self.mode == 'train':
            # anc and pos are augmented with different rotations
            [[anc_pc_np, anc_sn_np, anc_node_np], [pos_pc_np, pos_sn_np, pos_node_np]] = self.augment(
                [[anc_pc_np, anc_sn_np, anc_node_np], [pos_pc_np, pos_sn_np, pos_node_np]])

        anc_pc = torch.from_numpy(anc_pc_np.transpose().astype(np.float32))  # 3xN
        anc_sn = torch.from_numpy(anc_sn_np.transpose().astype(np.float32))  # 3xN
        anc_node = torch.from_numpy(anc_node_np.transpose().astype(np.float32))  # 3xM
        pos_pc = torch.from_numpy(pos_pc_np.transpose().astype(np.float32))  # 3xN
        pos_sn = torch.from_numpy(pos_sn_np.transpose().astype(np.float32))  # 3xN
        pos_node = torch.from_numpy(pos_node_np.transpose().astype(np.float32))  # 3xM

        # debug
        # fig = plt.figure(figsize=(9, 9))
        # ax = Axes3D(fig)
        # ax = vis_tools.plot_pc(anc_pc_np, color=[0, 0, 1], ax=ax)
        # ax = vis_tools.plot_pc(pos_pc_np, color=[1, 0, 0], ax=ax)
        # plt.show()

        return anc_pc, anc_sn, anc_node, \
               pos_pc, pos_sn, pos_node, \
               index


if __name__ == '__main__':
    import timeit
    from oxford import options_detector
    opt = options_detector.Options().parse()  # set CUDA_VISIBLE_DEVICES before import torch

    trainset = OxfordDescriptorLoader(opt.dataroot, 'train', opt)
    dataset_size = len(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True,
                                              num_workers=opt.nThreads, drop_last=True, pin_memory=True)
    print('#training point clouds = %d' % len(trainset))

    testset = OxfordDescriptorLoader(opt.dataroot, 'test', opt)
    testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=True,
                                             num_workers=opt.nThreads, pin_memory=True)
    print('#testing point clouds = %d' % len(testset))

    print('training set ----------------')
    t_sum = 0
    for i, data in enumerate(trainloader):
        anc_pc, anc_sn, anc_keypoints, anc_sigmas, \
        pos_pc, pos_sn, pos_keypoints, pos_sigmas, \
        index_batch = data

        start_t = timeit.default_timer()
        neg_idx = trainset.mine_negative_sample(index_batch)
        stop_t = timeit.default_timer()
        t_sum += stop_t - start_t

        if i > 100:
            print('train, average neg_idx time cost: %f' % (t_sum / i))
            break

        # visualize
        # anc_pc_np = anc_pc.transpose(1, 2).numpy()  # BxNx3
        # anc_keypoints_np = anc_keypoints.transpose(1, 2).numpy()  # BxMx3
        # anc_sigmas_np = anc_sigmas.numpy()  # BxM
        #
        # print(np.max(anc_sigmas_np))
        # print(np.min(anc_sigmas_np))
        #
        # anc_sigmas_normalized_np = (1.0 / anc_sigmas_np) / np.max(1.0 / anc_sigmas_np)  # BxM
        #
        # b = 0
        # fig_anc = plt.figure(figsize=(9, 9))
        # ax_src = Axes3D(fig_anc)
        # ax_src.scatter(anc_pc_np[b, :, 0].tolist(), anc_pc_np[b, :, 1].tolist(), anc_pc_np[b, :, 2].tolist(),
        #                s=5,
        #                c=np.repeat(np.asarray([[191 / 255, 191 / 255, 191 / 255]]), anc_pc_np[b].shape[0], axis=0))
        # ax_src.scatter(anc_keypoints_np[b, :, 0].tolist(), anc_keypoints_np[b, :, 1].tolist(),
        #                anc_keypoints_np[b, :, 2].tolist(),
        #                s=8,
        #                c=np.repeat(np.asarray([[1, 0, 0]]), anc_keypoints_np[b].shape[0], axis=0) * np.expand_dims(
        #                    anc_sigmas_normalized_np[b, :], axis=1))
        # axisEqual3D(ax_src)
        # plt.show()

        # verify neg_idx is correct
        B = index_batch.size()[0]
        for i in range(B):
            if index_batch[neg_idx[i].item()].item() in trainset.dataset[index_batch[i].item()]['nonneg_line_list']:
                print('WRONG!!! anc_line %d, neg_idx %d, neg_anc_line %d' % (index_batch[i].item(),
                                                                             neg_idx[i].item(),
                                                                             index_batch[neg_idx[i]].item()))
        # break

    print('testing set ----------------')
    t_sum = 0
    for data in testloader:
        anc_pc, anc_sn, anc_keypoints, anc_sigmas, \
        pos_pc, pos_sn, pos_keypoints, pos_sigmas, \
        index_batch = data

        start_t = timeit.default_timer()
        neg_idx = testset.mine_negative_sample(index_batch)
        stop_t = timeit.default_timer()
        t_sum += stop_t - start_t

        # visualize
        # anc_pc_np = anc_pc.transpose(1, 2).numpy()  # BxNx3
        # anc_keypoints_np = anc_keypoints.transpose(1, 2).numpy()  # BxMx3
        # anc_sigmas_np = anc_sigmas.numpy()  # BxM
        #
        # print(np.max(anc_sigmas_np))
        # print(np.min(anc_sigmas_np))
        #
        # anc_sigmas_normalized_np = (1.0 / anc_sigmas_np) / np.max(1.0 / anc_sigmas_np)  # BxM
        #
        # b = 0
        # fig_anc = plt.figure(figsize=(9, 9))
        # ax_src = Axes3D(fig_anc)
        # ax_src.scatter(anc_pc_np[b, :, 0].tolist(), anc_pc_np[b, :, 1].tolist(), anc_pc_np[b, :, 2].tolist(),
        #                s=5,
        #                c=np.repeat(np.asarray([[191 / 255, 191 / 255, 191 / 255]]), anc_pc_np[b].shape[0], axis=0))
        # ax_src.scatter(anc_keypoints_np[b, :, 0].tolist(), anc_keypoints_np[b, :, 1].tolist(),
        #                anc_keypoints_np[b, :, 2].tolist(),
        #                s=8,
        #                c=np.repeat(np.asarray([[1, 0, 0]]), anc_keypoints_np[b].shape[0], axis=0) * np.expand_dims(
        #                    anc_sigmas_normalized_np[b, :], axis=1))
        # axisEqual3D(ax_src)
        # plt.show()

        # verify neg_idx is correct
        B = index_batch.size()[0]
        for i in range(B):
            if testset.dataset[index_batch[neg_idx[i].item()].item()]['anc_idx'] in testset.dataset[index_batch[i]]['pos_idx_list']:
                print('WRONG!!! anc_idx %d, neg_idx %d, neg_anc_idx %d' % (testset.dataset[index_batch[i].item()]['anc_idx'],
                                                                           neg_idx[i].item(),
                                                                           testset.dataset[index_batch[neg_idx[i].item()].item()]['anc_idx']))
    print('test, average neg_idx time cost: %f' % (t_sum / len(testloader)))
        # break

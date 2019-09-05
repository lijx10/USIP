import torch.utils.data as data

import random
import numbers
import os
import os.path
import numpy as np
import struct
import math
import time

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


class KittiDescriptorLoader(data.Dataset):
    def __init__(self, root, mode, opt):
        super(KittiDescriptorLoader, self).__init__()
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

        # random choice
        pc_np = np.load(pc_np_file)  # Nx4, x, y, z, reflectance
        choice_idx = np.random.choice(pc_np.shape[0], self.opt.input_pc_num, replace=False)
        pc_np = pc_np[choice_idx, :]
        sn_np = pc_np[:, 3:3 + self.opt.surface_normal_len]  # Nx5, nx, ny, nz, curvature, reflectance, \in [0, 0.99], mean 0.27
        pc_np = pc_np[:, 0:3]  # Nx3

        pose_np = np.load(pose_np_file)['pose']  # 4x4

        # get nodes, perform random sampling to reduce computation cost
        sampled_pc_np = pc_np[np.random.choice(pc_np.shape[0], int(self.opt.input_pc_num/4), replace=False)]
        node_np = self.farthest_sampler.sample(sampled_pc_np, self.opt.node_num)

        # # debug
        # _, seq_2, _, pose_np_2 = self.get_seq_pose_by_index(index)
        # print(seq - seq_2)
        # print(pose_np - pose_np_2)

        return pc_np, sn_np, node_np, seq, pose_np

    def get_seq_pose_by_index(self, index):
        # determine the sequence
        for i, accumulated_sample_num in enumerate(self.accumulated_sample_num_list):
            if index < accumulated_sample_num:
                break
        folder = self.folder_list[i]
        seq = self.seq_list[i]

        if i == 0:
            index_in_seq = index
        else:
            index_in_seq = index - self.accumulated_sample_num_list[i - 1]
        pose_np_file = os.path.join(self.root, 'poses', '%02d' % seq, '%06d.npz' % index_in_seq)
        pose_np = np.load(pose_np_file)['pose']

        return i, seq, index_in_seq, pose_np

    def get_nearby_instance_unagumented_np(self, index, positive_radius_threshold):
        accumulated_sample_num_list_i, seq, index_in_seq, pose = self.get_seq_pose_by_index(index)

        # random sample to get nearby scan, within the distance of radius_threshold
        # rotation is not constrained because the scan is 360 degree
        search_interval = int(positive_radius_threshold / 0.8 * 2)  # assume that 0.8 meter between consecutive scans, the multiplier is a safe margin
        search_range_min = max(index_in_seq - search_interval, 0)
        search_range_max = min(index_in_seq + search_interval, self.sample_num_list[accumulated_sample_num_list_i]-1)

        counter = 0
        while True:
            # search_range_min <= nearby_idx_in_seq <= search_range_max
            nearby_idx_in_seq = random.randint(search_range_min, search_range_max)
            nearby_pose_np_file = os.path.join(self.root, 'poses', '%02d' % seq, '%06d.npz' % nearby_idx_in_seq)
            nearby_pose = np.load(nearby_pose_np_file)['pose']

            # verify the relative distance
            # pose_origin = np.linalg.inv(pose)
            # pose_nearby_pose = np.dot(pose_origin, nearby_pose)  # 4x4
            # distance = np.linalg.norm(pose_nearby_pose[0:3, 3])
            # approximate relative distance, to avoid matrix inverse
            distance = np.linalg.norm((nearby_pose - pose)[0:3, 3])
            if distance < positive_radius_threshold:
                break
            else:
                # control the search_range_min and search_range_max to reduce computational cost
                if nearby_idx_in_seq < index_in_seq:
                    search_range_min = nearby_idx_in_seq + 1
                else:
                    search_range_max = nearby_idx_in_seq - 1

            # avoid deadlock, return itself
            counter += 1
            if counter >= search_interval * 3:
                nearby_idx_in_seq = index_in_seq
                nearby_pose = pose
                break

        if accumulated_sample_num_list_i == 0:
            nearby_idx = nearby_idx_in_seq
        else:
            nearby_idx = nearby_idx_in_seq + self.accumulated_sample_num_list[accumulated_sample_num_list_i-1]

        nearby_pc_np, nearby_sn_np, nearby_node_np, _, _ = self.get_instance_unaugmented_np(nearby_idx)

        # debug
        # print(seq-seq2)
        # print(pose2-nearby_pose)

        return nearby_pc_np, nearby_sn_np, nearby_node_np, seq, nearby_pose

    def augment(self, data_package_list):
        '''
        apply the same augmentation
        :param data_package_list: [(pc_np, sn_np, node_np), (...), ...]
        :return: augmented_package_list: [(pc_np, sn_np, node_np), (...), ...]
        '''
        B = len(data_package_list)
        
        # augmentation parameter / data
        # scale ------
        scale = np.random.uniform(low=0.9, high=1.1)
        # scale = 1.0

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
            N, C = data_package_list[0][0].shape
            jitter_pc = np.clip(sigma * np.random.randn(N, 3), -1 * clip, clip)
            sigma, clip = 0.01, 0.05
            jitter_sn = np.clip(sigma * np.random.randn(N, self.opt.surface_normal_len), -1 * clip, clip)  # nx, ny, nz, curvature, reflectance
            sigma, clip = 0.04, 0.12
            N, C = data_package_list[0][2].shape
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

    def mine_negative_sample(self, anc_seq_batch, anc_pose_batch, negative_radius_threshold):
        '''
        find negative samples from the same batch, return the index
        :param anc_seq_batch: B, int TODO: type?
        :param anc_pose_batch: Bx4x4, FloatTensor
        :param negative_radius_threshold: float
        :return: neg_idx, LongTensor, B
        '''
        # debug
        # print(anc_seq_batch)
        # print(anc_pose_batch)

        anc_pose_batch_np = anc_pose_batch.numpy()  # Bx4x4

        B = anc_pose_batch.size()[0]
        neg_idx = torch.zeros(B, dtype=torch.int64)
        # for each anchor, search over all other anchor
        for i in range(B):
            neg_idx_candidate_for_i = []
            for j in range(B):
                if j == i:
                    continue
                else:
                    # 1. in the same sequence?
                    if anc_seq_batch[i] != anc_seq_batch[j]:
                        neg_idx_candidate_for_i.append(j)
                    else:
                        i_origin = np.linalg.inv(anc_pose_batch_np[i])
                        i_j = np.dot(i_origin, anc_pose_batch_np[j])  # 4x4
                        distance = np.linalg.norm(i_j[0:3, 3])
                        if distance > negative_radius_threshold:
                            neg_idx_candidate_for_i.append(j)
            # random sample from neg_idx_candidate_for_i, to get a neg_idx for i
            # TODO: theoretically there is a chance that there is no candidate at all, should be handled.
            if len(neg_idx_candidate_for_i) == 0:
                print('!!!!!!!!!!!!Fail to mine negative sample!!!!!!!!!!!!')
            else:
                neg_idx[i] = neg_idx_candidate_for_i[np.random.randint(len(neg_idx_candidate_for_i))]

        return neg_idx


    def __getitem__(self, index):
        anc_pc_np, anc_sn_np, anc_node_np, anc_seq, anc_pose_np = self.get_instance_unaugmented_np(index)
        pos_pc_np, pos_sn_np, pos_node_np, pos_seq, pos_pose_np = self.get_nearby_instance_unagumented_np(index,
                                                                                                          positive_radius_threshold=self.opt.positive_radius_threshold)

        # debug
        # fig = plt.figure(figsize=(9, 9))
        # ax = Axes3D(fig)
        # ax = vis_tools.plot_pc(src_pc_np, color=[0, 0, 1], ax=ax)
        # ax = vis_tools.plot_pc(dst_pc_np, color=[1, 0, 0], ax=ax)
        # plt.show()

        if self.mode == 'train':
            [[anc_pc_np, anc_sn_np, anc_node_np], [pos_pc_np, pos_sn_np, pos_node_np]] = self.augment(
                [[anc_pc_np, anc_sn_np, anc_node_np], [pos_pc_np, pos_sn_np, pos_node_np]])

        anc_pc = torch.from_numpy(anc_pc_np.transpose().astype(np.float32))  # 3xN
        anc_sn = torch.from_numpy(anc_sn_np.transpose().astype(np.float32))  # 3xN
        anc_node = torch.from_numpy(anc_node_np.transpose().astype(np.float32))  # 3xM
        anc_pose = torch.from_numpy(anc_pose_np.astype(np.float32))  # 4x4
        pos_pc = torch.from_numpy(pos_pc_np.transpose().astype(np.float32))  # 3xN
        pos_sn = torch.from_numpy(pos_sn_np.transpose().astype(np.float32))  # 3xN
        pos_node = torch.from_numpy(pos_node_np.transpose().astype(np.float32))  # 3xM
        pos_pose = torch.from_numpy(pos_pose_np.astype(np.float32))  # 4x4


        return anc_pc, anc_sn, anc_node, anc_seq, anc_pose, \
               pos_pc, pos_sn, pos_node, pos_seq, pos_pose






if __name__ == '__main__':
    from kitti import options_detector
    from kitti import options_descriptor

    print('====== detector ======')
    opt_detector = options_detector.Options().parse()  # set CUDA_VISIBLE_DEVICES before import torch
    print('====== descriptor ======')
    opt_descriptor = options_descriptor.Options().parse()

    trainset = KittiDescriptorLoader(opt_descriptor.dataroot, 'train', opt_descriptor)
    dataset_size = len(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt_descriptor.batch_size, shuffle=True,
                                              num_workers=opt_descriptor.nThreads, drop_last=True, pin_memory=False)
    print('#training point clouds = %d' % len(trainset))

    for epoch in range(1000):
        iter_start_time = time.time()
        print('epoch %d' % epoch)
        for i, data in enumerate(trainloader):
            anc_pc, anc_sn, anc_node, anc_seq, anc_pose, \
            pos_pc, pos_sn, pos_node, pos_seq, pos_pose = data
        t_epoch = (time.time() - iter_start_time)
        print('epoch %d time %f' % (epoch, t_epoch))

    # for i in range(100):
    #     trainset[i]

    print('done')
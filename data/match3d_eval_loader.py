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

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from util import vis_tools


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


def make_dataset_3dmatch_eval(root):
    dataset = []
    scene_list = ['7-scenes-redkitchen', 'sun3d-home_at-home_at_scan1_2013_jan_1',
                  'sun3d-home_md-home_md_scan9_2012_sep_30', 'sun3d-hotel_uc-scan3',
                  'sun3d-hotel_umd-maryland_hotel1', 'sun3d-hotel_umd-maryland_hotel3',
                  'sun3d-mit_76_studyroom-76-1studyroom2', 'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika']
    for scene_idx, scene in enumerate(scene_list):
        npy_folder = os.path.join(root, scene)
        npy_file_list = os.listdir(npy_folder)
        for npy_file in npy_file_list:
            npy_file_path = os.path.join(npy_folder, npy_file)

            npy_file_len = len(npy_file)
            frame_idx_begin = 10
            frame_idx_end = npy_file_len - 4
            frame_idx = int(npy_file[frame_idx_begin:frame_idx_end])
            dataset.append((npy_file_path, scene_idx, frame_idx))

    return dataset


class Match3DEvalLoader(data.Dataset):
    def __init__(self, root, opt):
        super(Match3DEvalLoader, self).__init__()
        self.root = root
        self.opt = opt
        self.dataset = make_dataset_3dmatch_eval(root)

        self.scene_name_list = ['7-scenes-redkitchen', 'sun3d-home_at-home_at_scan1_2013_jan_1',
                           'sun3d-home_md-home_md_scan9_2012_sep_30', 'sun3d-hotel_uc-scan3',
                           'sun3d-hotel_umd-maryland_hotel1', 'sun3d-hotel_umd-maryland_hotel3',
                           'sun3d-mit_76_studyroom-76-1studyroom2', 'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika']

        # farthest point sample
        self.farthest_sampler = FarthestSampler()

    def __len__(self):
        return len(self.dataset)

    def get_instance_unaugmented_np(self, index):
        npy_file_path, scene_idx, frame_idx = self.dataset[index]
        pc_np = np.load(npy_file_path)

        # random sample
        choice_idx = np.random.choice(pc_np.shape[0], self.opt.input_pc_num, replace=False)

        pc_np = pc_np[choice_idx, :]
        sn_np = pc_np[:, 3:3 + self.opt.surface_normal_len]  # Nx5, nx, ny, nz, curvature, reflectance, \in [0, 0.99], mean 0.27
        pc_np = pc_np[:, 0:3]  # Nx3, x, y, z

        return pc_np, sn_np, scene_idx, frame_idx

    def __getitem__(self, index):
        anc_pc_np, anc_sn_np, scene_idx, frame_idx = self.get_instance_unaugmented_np(index)
        # get nodes, perform random sampling to reduce computation cost
        anc_node_np = self.farthest_sampler.sample(
            anc_pc_np[np.random.choice(anc_pc_np.shape[0], int(self.opt.input_pc_num / 2), replace=False)],
            self.opt.node_num)

        anc_pc = torch.from_numpy(anc_pc_np.transpose().astype(np.float32))  # 3xN
        anc_sn = torch.from_numpy(anc_sn_np.transpose().astype(np.float32))  # 3xN
        anc_node = torch.from_numpy(anc_node_np.transpose().astype(np.float32))  # 3xM

        return anc_pc, anc_sn, anc_node, \
               scene_idx, frame_idx



if __name__ == "__main__":
    root = '/ssd/jiaxin/TSF_datasets/3DMatch_eval_npy'
    dataset = make_dataset_3dmatch_eval(root)
    print(len(dataset))


import time
import copy
import numpy as np
import math

from options import Options
opt = Options().parse()  # set CUDA_VISIBLE_DEVICES before import torch

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np

from models.keypoint_detector import Model
from data.modelnet_shrec_loader import ModelNet_Shrec_Loader
from util.visualizer import Visualizer

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


if __name__=='__main__':
    trainset = ModelNet_Shrec_Loader(opt.dataroot, 'train', opt)
    dataset_size = len(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.nThreads, drop_last=True)
    print('#training point clouds = %d' % len(trainset))

    testset = ModelNet_Shrec_Loader(opt.dataroot, 'test', opt)
    testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.nThreads)

    # create model, optionally load pre-trained model
    model = Model(opt)
    model.detector.load_state_dict(torch.load(
        '/ssd/keypoint-pc/modelnet/checkpoints/save/8b025dc-keypc6-1024/gpu1_471_-4.613248_net_detector.pth'))

    visualizer = Visualizer(opt)

    batch_amount = 0
    model.test_loss_average.zero_()
    model.test_chamfer_average.zero_()
    model.test_keypoint_on_pc_average.zero_()
    model.test_chamfer_pure_average.zero_()
    model.test_chamfer_weighted_average.zero_()

    for i, data in enumerate(testloader):
        src_pc, src_sn, src_node, src_node_knn_I, \
        dst_pc, dst_sn, dst_node, dst_node_knn_I, \
        R, scale, shift = data
        model.set_input(src_pc, src_sn, src_node,
                        dst_pc, dst_sn, dst_node,
                        R, scale, shift)
        # model.optimize()
        model.test_model()

        batch_amount += src_pc.size()[0]

        # accumulate loss
        model.test_loss_average += model.loss.detach() * src_pc.size()[0]
        model.test_chamfer_average += model.loss_chamfer.detach() * src_pc.size()[0]
        model.test_keypoint_on_pc_average += (model.loss_keypoint_on_pc_src.detach() + model.loss_keypoint_on_pc_dst.detach()) * src_pc.size()[0]
        model.test_chamfer_pure_average += model.chamfer_criteria.chamfer_pure.detach() * src_pc.size()[0]
        model.test_chamfer_weighted_average += model.chamfer_criteria.chamfer_weighted.detach() * src_pc.size()[0]

        # visualize with numpy & matplotlib
        src_pc_np = src_pc.transpose(1,2).numpy()  # BxNx3
        dst_pc_np = dst_pc.transpose(1,2).numpy()  # BxNx3

        src_node_np = model.src_node_recomputed.detach().cpu().transpose(1,2).numpy()  # BxMx3
        dst_node_np = model.dst_node_recomputed.detach().cpu().transpose(1,2).numpy()  # BxMx3

        src_keypoints_np = model.src_keypoints.detach().cpu().transpose(1,2).numpy()  # BxMx3
        src_sigmas_np = model.src_sigmas.detach().cpu().numpy()  # BxM
        src_sigmas_normalized_np = (1.0/src_sigmas_np) / np.max(1.0/src_sigmas_np)  # BxM
        dst_keypoints_np = model.dst_keypoints.detach().cpu().transpose(1,2).numpy()  # BxMx3
        dst_sigmas_np = model.dst_sigmas.detach().cpu().numpy()  # BxM
        dst_sigmas_normalized_np = (1.0/dst_sigmas_np) / np.max(1.0/dst_sigmas_np)  # BxM

        src_keypoints_transformed_np = model.src_keypoints_transformed.detach().cpu().transpose(1,2).numpy()  # BxMx3

        for b in range(src_pc_np.shape[0]):
            # plot src
            fig_src = plt.figure(figsize=(9, 9))
            ax_src = Axes3D(fig_src)
            ax_src.scatter(src_pc_np[b, :, 0].tolist(), src_pc_np[b, :, 1].tolist(), src_pc_np[b, :, 2].tolist(),
                           s=5,
                           c=np.repeat(np.asarray([[191 / 255, 191 / 255, 191 / 255]]), src_pc_np[b].shape[0], axis=0))
            ax_src.scatter(src_keypoints_np[b, :, 0].tolist(), src_keypoints_np[b, :, 1].tolist(),
                           src_keypoints_np[b, :, 2].tolist(),
                           s=8,
                           c=np.repeat(np.asarray([[1, 0, 0]]), src_node_np[b].shape[0], axis=0) * np.expand_dims(src_sigmas_normalized_np[b, :], axis=1))
            axisEqual3D(ax_src)

            # plot dst with sigma
            # fig_dst = plt.figure(figsize=(9, 9))
            # ax_dst = Axes3D(fig_dst)
            # ax_dst.scatter(dst_pc_np[b, :, 0], dst_pc_np[b, :, 1], dst_pc_np[b, :, 2],
            #                s=5,
            #                c=np.repeat(np.asarray([[191 / 255, 191 / 255, 191 / 255]]), dst_pc_np[b].shape[0], axis=0))
            # ax_dst.scatter(dst_keypoints_np[b, :, 0].tolist(), dst_keypoints_np[b, :, 1].tolist(),
            #                dst_keypoints_np[b, :, 2].tolist(),
            #                s=8,
            #                c=np.repeat(np.asarray([[1, 0, 0]]), dst_node_np[b].shape[0], axis=0) * np.expand_dims(dst_sigmas_normalized_np[b, :], axis=1))
            # ax_dst.scatter(src_keypoints_transformed_np[b, :, 0].tolist(),
            #                src_keypoints_transformed_np[b, :, 1].tolist(),
            #                src_keypoints_transformed_np[b, :, 2].tolist(),
            #                s=8,
            #                c=np.repeat(np.asarray([[0, 0, 1]]), src_keypoints_transformed_np[b].shape[0], axis=0) * np.expand_dims(src_sigmas_normalized_np[b, :], axis=1))
            # axisEqual3D(ax_dst)
            
            print('src_sigma mean: %f, min: %f, max: %f' % (np.mean(src_sigmas_np[b, :]), np.min(src_sigmas_np[b, :]), np.max(src_sigmas_np[b, :])))
            print('dst_sigma mean: %f, min: %f, max: %f' % (np.mean(dst_sigmas_np[b, :]), np.min(dst_sigmas_np[b, :]), np.max(dst_sigmas_np[b, :])))
            print('---')


            plt.show()
            # break

        # break

        # update best loss
        model.test_loss_average /= batch_amount
        model.test_chamfer_average /= batch_amount
        model.test_keypoint_on_pc_average /= batch_amount
        model.test_chamfer_pure_average /= batch_amount
        model.test_chamfer_weighted_average /= batch_amount
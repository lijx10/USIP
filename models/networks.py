import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
import numpy as np
import math
import time

from util import som
from models import operations
from models.layers import *
from util import vis_tools

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import index_max
import ball_query

class RPN_Detector(nn.Module):
    def __init__(self, opt):
        super(RPN_Detector, self).__init__()
        self.opt = opt

        self.C1 = 128
        # first PointNet
        self.first_pointnet = PointNet(3+self.opt.surface_normal_len,
                                       [int(self.C1/2), int(self.C1/2), int(self.C1/2)],
                                       activation=self.opt.activation,
                                       normalization=self.opt.normalization,
                                       momentum=opt.bn_momentum,
                                       bn_momentum_decay_step=opt.bn_momentum_decay_step,
                                       bn_momentum_decay=opt.bn_momentum_decay)

        self.second_pointnet = PointNet(self.C1, [self.C1, self.C1], activation=self.opt.activation,
                                        normalization=self.opt.normalization,
                                        momentum=opt.bn_momentum,
                                        bn_momentum_decay_step=opt.bn_momentum_decay_step,
                                        bn_momentum_decay=opt.bn_momentum_decay)

        assert self.opt.node_knn_k_1 >= 2

        self.C2 = 512
        self.knnlayer_1 = GeneralKNNFusionModule(3 + self.C1, (int(self.C2/2), int(self.C2/2), int(self.C2/2)), (self.C2, self.C2),
                                                 activation=self.opt.activation,
                                                 normalization=self.opt.normalization,
                                                 momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step,
                                                 bn_momentum_decay=opt.bn_momentum_decay)


        # masked max
        # self.masked_max = operations.MaskedMax(self.opt.node_num)

        # --- PointNet to get weight/sigma --- begin --- #
        input_channels = self.C1 + self.C2
        output_channels = 4

        self.mlp1 = EquivariantLayer(input_channels, 512,
                                     activation=self.opt.activation, normalization=self.opt.normalization,
                                     momentum=opt.bn_momentum,
                                     bn_momentum_decay_step=opt.bn_momentum_decay_step,
                                     bn_momentum_decay=opt.bn_momentum_decay)
        self.mlp2 = EquivariantLayer(512, 256,
                                     activation=self.opt.activation, normalization=self.opt.normalization,
                                     momentum=opt.bn_momentum,
                                     bn_momentum_decay_step=opt.bn_momentum_decay_step,
                                     bn_momentum_decay=opt.bn_momentum_decay)
        self.mlp3 = EquivariantLayer(256, output_channels, activation=None, normalization=None)

        self.mlp3.conv.weight.data.normal_(0, 1e-4)
        self.mlp3.conv.bias.data.zero_()
        self.softplus = torch.nn.Softplus()
        # --- PointNet to get weight/sigma --- end --- #

    def forward(self, x, sn, node, is_train=False, epoch=None):
        '''

        :param x: Bx3xN Tensor
        :param sn: Bx3xN Tensor
        :param node: Bx3xM FloatTensor
        :param is_train: determine whether to add noise in KNNModule
        :return:
        '''
        # modify the x according to the nodes, minus the center
        mask, mask_row_max, min_idx = som.query_topk(node, x, node.size()[2],
                                                     k=self.opt.k)  # BxkNxnode_num, Bxnode_num, BxkN
        mask_row_sum = torch.sum(mask, dim=1)  # Bxnode_num
        mask = mask.unsqueeze(1)  # Bx1xkNxnode_num

        # if necessary, stack the x
        x_stack = x.repeat(1, 1, self.opt.k)
        sn_stack = sn.repeat(1, 1, self.opt.k)

        x_stack_data_unsqueeze = x_stack.unsqueeze(3)  # BxCxkNx1
        x_stack_data_masked = x_stack_data_unsqueeze * mask.float()  # BxCxkNxnode_num
        cluster_mean = torch.sum(x_stack_data_masked, dim=2) / (
                mask_row_sum.unsqueeze(1).float() + 1e-5).detach()  # BxCxnode_num
        # cluster_mean = node
        som_node_cluster_mean = cluster_mean

        B, N, kN, M = x.size()[0], x.size()[2], x_stack.size()[2], som_node_cluster_mean.size()[2]

        # assign each point with a center
        node_expanded = som_node_cluster_mean.unsqueeze(2)  # BxCx1xnode_num, som.node is BxCxnode_num
        centers = torch.sum(mask.float() * node_expanded, dim=3).detach()  # BxCxkN

        x_decentered = (x_stack - centers).detach()  # Bx3xkN
        x_augmented = torch.cat((x_decentered, sn_stack), dim=1)  # Bx6xkN

        # go through the first PointNet
        if self.opt.surface_normal_len >= 1:
            first_pn_out = self.first_pointnet(x_augmented, epoch)
        else:
            first_pn_out = self.first_pointnet(x_decentered, epoch)

        # first_gather_index = self.masked_max.compute(first_pn_out, min_idx, mask).detach()  # BxCxM
        with torch.cuda.device(first_pn_out.get_device()):
            first_gather_index = index_max.forward_cuda_shared_mem(first_pn_out.detach(), min_idx.int(), M).detach().long()
        first_pn_out_masked_max = first_pn_out.gather(dim=2,
                                                      index=first_gather_index) * mask_row_max.unsqueeze(1).float()  # BxCxM

        # scatter the masked_max back to the kN points
        scattered_first_masked_max = torch.gather(first_pn_out_masked_max,
                                                  dim=2,
                                                  index=min_idx.unsqueeze(1).expand(B, first_pn_out.size()[1], kN))  # BxCxkN
        first_pn_out_fusion = torch.cat((first_pn_out, scattered_first_masked_max), dim=1)  # Bx2CxkN
        second_pn_out = self.second_pointnet(first_pn_out_fusion, epoch)

        # second_gather_index = self.masked_max.compute(second_pn_out, min_idx, mask).detach()  # BxCxM
        with torch.cuda.device(second_pn_out.get_device()):
            second_gather_index = index_max.forward_cuda_shared_mem(second_pn_out, min_idx.int(), M).detach().long()
        second_pn_out_masked_max = second_pn_out.gather(dim=2,
                                                        index=second_gather_index) * mask_row_max.unsqueeze(1).float()  # BxCxM


        # knn module, knn search on nodes: ----------------------------------
        knn_feature_1 = self.knnlayer_1(query=som_node_cluster_mean,
                                        database=som_node_cluster_mean,
                                        x=second_pn_out_masked_max,
                                        K=self.opt.node_knn_k_1,
                                        epoch=epoch)

        node_feature_aggregated = torch.cat((second_pn_out_masked_max, knn_feature_1), dim=1)  # Bx(C1+C2)xM

        # go through another network to calculate the per-node keypoint & uncertainty sigma
        y = self.mlp1(node_feature_aggregated)
        point_descriptor = self.mlp2(y)
        keypoint_sigma = self.mlp3(point_descriptor)  # Bx(3+1)xkN

        # keypoint = keypoint + node_coordinate
        keypoints = keypoint_sigma[:, 0:3, :] + som_node_cluster_mean  # Bx3xM
        # make sure sigma>=0 by square
        # sigmas = torch.pow(keypoint_sigma[:, 3, :], exponent=2) + self.opt.loss_sigma_lower_bound  # BxM
        sigmas = self.softplus(keypoint_sigma[:, 3, :]) + self.opt.loss_sigma_lower_bound  # BxM

        descriptors = None

        # debug
        # print(keypoints)
        # print(sigmas)

        return som_node_cluster_mean, keypoints, sigmas, descriptors


class RPN_DetectorLite(nn.Module):
    def __init__(self, opt):
        super(RPN_DetectorLite, self).__init__()
        self.opt = opt

        self.C1 = 64
        # first PointNet
        self.first_pointnet = PointNet(3+self.opt.surface_normal_len,
                                       [int(self.C1/2), int(self.C1/2), int(self.C1/2)],
                                       activation=self.opt.activation,
                                       normalization=self.opt.normalization,
                                       momentum=opt.bn_momentum,
                                       bn_momentum_decay_step=opt.bn_momentum_decay_step,
                                       bn_momentum_decay=opt.bn_momentum_decay)

        self.second_pointnet = PointNet(self.C1, [self.C1, self.C1], activation=self.opt.activation,
                                        normalization=self.opt.normalization,
                                        momentum=opt.bn_momentum,
                                        bn_momentum_decay_step=opt.bn_momentum_decay_step,
                                        bn_momentum_decay=opt.bn_momentum_decay)

        assert self.opt.node_knn_k_1 >= 2

        self.C2 = 256
        self.knnlayer_1 = GeneralKNNFusionModule(3 + self.C1, (int(self.C2/2), int(self.C2/2), int(self.C2/2)), (self.C2, self.C2),
                                                 activation=self.opt.activation,
                                                 normalization=self.opt.normalization,
                                                 momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step,
                                                 bn_momentum_decay=opt.bn_momentum_decay)


        # masked max
        # self.masked_max = operations.MaskedMax(self.opt.node_num)

        # --- PointNet to get weight/sigma --- begin --- #
        input_channels = self.C1 + self.C2
        output_channels = 4

        self.mlp1 = EquivariantLayer(input_channels, 512,
                                     activation=self.opt.activation, normalization=self.opt.normalization,
                                     momentum=opt.bn_momentum,
                                     bn_momentum_decay_step=opt.bn_momentum_decay_step,
                                     bn_momentum_decay=opt.bn_momentum_decay)
        self.mlp2 = EquivariantLayer(512, 256,
                                     activation=self.opt.activation, normalization=self.opt.normalization,
                                     momentum=opt.bn_momentum,
                                     bn_momentum_decay_step=opt.bn_momentum_decay_step,
                                     bn_momentum_decay=opt.bn_momentum_decay)
        self.mlp3 = EquivariantLayer(256, output_channels, activation=None, normalization=None)

        self.mlp3.conv.weight.data.normal_(0, 1e-4)
        self.mlp3.conv.bias.data.zero_()
        self.softplus = torch.nn.Softplus()
        # --- PointNet to get weight/sigma --- end --- #

    def forward(self, x, sn, node, is_train=False, epoch=None):
        '''

        :param x: Bx3xN Tensor
        :param sn: Bx3xN Tensor
        :param node: Bx3xM FloatTensor
        :param is_train: determine whether to add noise in KNNModule
        :return:
        '''
        # modify the x according to the nodes, minus the center
        mask, mask_row_max, min_idx = som.query_topk(node, x, node.size()[2],
                                                     k=self.opt.k)  # BxkNxnode_num, Bxnode_num, BxkN
        mask_row_sum = torch.sum(mask, dim=1)  # Bxnode_num
        mask = mask.unsqueeze(1)  # Bx1xkNxnode_num

        # if necessary, stack the x
        x_stack = x.repeat(1, 1, self.opt.k)
        sn_stack = sn.repeat(1, 1, self.opt.k)

        x_stack_data_unsqueeze = x_stack.unsqueeze(3)  # BxCxkNx1
        x_stack_data_masked = x_stack_data_unsqueeze * mask.float()  # BxCxkNxnode_num
        cluster_mean = torch.sum(x_stack_data_masked, dim=2) / (
                mask_row_sum.unsqueeze(1).float() + 1e-5).detach()  # BxCxnode_num
        # cluster_mean = node
        som_node_cluster_mean = cluster_mean

        B, N, kN, M = x.size()[0], x.size()[2], x_stack.size()[2], som_node_cluster_mean.size()[2]

        # assign each point with a center
        node_expanded = som_node_cluster_mean.unsqueeze(2)  # BxCx1xnode_num, som.node is BxCxnode_num
        centers = torch.sum(mask.float() * node_expanded, dim=3).detach()  # BxCxkN

        x_decentered = (x_stack - centers).detach()  # Bx3xkN
        x_augmented = torch.cat((x_decentered, sn_stack), dim=1)  # Bx6xkN

        # go through the first PointNet
        if self.opt.surface_normal_len >= 1:
            first_pn_out = self.first_pointnet(x_augmented, epoch)
        else:
            first_pn_out = self.first_pointnet(x_decentered, epoch)

        # first_gather_index = self.masked_max.compute(first_pn_out, min_idx, mask).detach()  # BxCxM
        with torch.cuda.device(first_pn_out.get_device()):
            first_gather_index = index_max.forward_cuda_shared_mem(first_pn_out.detach(), min_idx.int(), M).detach().long()
        first_pn_out_masked_max = first_pn_out.gather(dim=2,
                                                      index=first_gather_index) * mask_row_max.unsqueeze(1).float()  # BxCxM

        # scatter the masked_max back to the kN points
        scattered_first_masked_max = torch.gather(first_pn_out_masked_max,
                                                  dim=2,
                                                  index=min_idx.unsqueeze(1).expand(B, first_pn_out.size()[1], kN))  # BxCxkN
        first_pn_out_fusion = torch.cat((first_pn_out, scattered_first_masked_max), dim=1)  # Bx2CxkN
        second_pn_out = self.second_pointnet(first_pn_out_fusion, epoch)

        # second_gather_index = self.masked_max.compute(second_pn_out, min_idx, mask).detach()  # BxCxM
        with torch.cuda.device(second_pn_out.get_device()):
            second_gather_index = index_max.forward_cuda_shared_mem(second_pn_out, min_idx.int(), M).detach().long()
        second_pn_out_masked_max = second_pn_out.gather(dim=2,
                                                        index=second_gather_index) * mask_row_max.unsqueeze(1).float()  # BxCxM


        # knn module, knn search on nodes: ----------------------------------
        knn_feature_1 = self.knnlayer_1(query=som_node_cluster_mean,
                                        database=som_node_cluster_mean,
                                        x=second_pn_out_masked_max,
                                        K=self.opt.node_knn_k_1,
                                        epoch=epoch)

        node_feature_aggregated = torch.cat((second_pn_out_masked_max, knn_feature_1), dim=1)  # Bx(C1+C2)xM

        # go through another network to calculate the per-node keypoint & uncertainty sigma
        y = self.mlp1(node_feature_aggregated)
        point_descriptor = self.mlp2(y)
        keypoint_sigma = self.mlp3(point_descriptor)  # Bx(3+1)xkN

        # keypoint = keypoint + node_coordinate
        keypoints = keypoint_sigma[:, 0:3, :] + som_node_cluster_mean  # Bx3xM
        # make sure sigma>=0 by square
        # sigmas = torch.pow(keypoint_sigma[:, 3, :], exponent=2) + self.opt.loss_sigma_lower_bound  # BxM
        sigmas = self.softplus(keypoint_sigma[:, 3, :]) + self.opt.loss_sigma_lower_bound  # BxM

        descriptors = None

        # debug
        # print(keypoints)
        # print(sigmas)

        return som_node_cluster_mean, keypoints, sigmas, descriptors


class DescriptorLiteOld(nn.Module):
    def __init__(self, opt):
        super(DescriptorLiteOld, self).__init__()
        self.opt = opt

        in_channels = 3 + opt.surface_normal_len

        self.conv1 = MyConv2d(in_channels, int(opt.descriptor_len/4), kernel_size=(1, 1), stride=1, padding=0, bias=True,
                              activation=self.opt.activation, normalization=self.opt.normalization,
                              momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step, bn_momentum_decay=opt.bn_momentum_decay)
        self.conv2 = MyConv2d(int(opt.descriptor_len/4), int(opt.descriptor_len/2), kernel_size=(1, 1), stride=1, padding=0, bias=True,
                              activation=self.opt.activation, normalization=self.opt.normalization,
                              momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step, bn_momentum_decay=opt.bn_momentum_decay)
        self.conv3 = MyConv2d(int(opt.descriptor_len/2), opt.descriptor_len, kernel_size=(1, 1), stride=1, padding=0, bias=True,
                              activation=self.opt.activation, normalization=self.opt.normalization,
                              momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step, bn_momentum_decay=opt.bn_momentum_decay)

        self.conv4 = MyConv2d(opt.descriptor_len*2, opt.descriptor_len, kernel_size=(1, 1), stride=1, padding=0, bias=True,
                              activation=self.opt.activation, normalization=self.opt.normalization,
                              momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step, bn_momentum_decay=opt.bn_momentum_decay)
        self.conv5 = MyConv2d(opt.descriptor_len, opt.descriptor_len, kernel_size=(1, 1), stride=1, padding=0, bias=True,
                              activation=None, normalization=None)

    def forward(self, x, sn, keypoints, is_train=False, epoch=None):
        '''

        :param x: Bx3xN
        :param sn: Bx*xN
        :param keypoints: Bx3xM, detected keypoints
        :param is_train: bool
        :param epoch:
        :return:
        '''
        # permute x and sn, because later a ball query will be performed
        B, M, N, K = x.size()[0], keypoints.size()[2], x.size()[2], self.opt.ball_nsamples
        permute_idx = torch.from_numpy(np.random.permutation(N).astype(np.int64)).to(x.device)
        x = x[:, :, permute_idx]
        sn = sn[:, :, permute_idx]

        if self.opt.surface_normal_len <= 0:
            x_aug = x
        else:
            x_aug = torch.cat((x, sn), dim=1)  # Bx4xN

        # node_neighbor_ball_idx = operations.ball_query_wrapper(x, keypoints, self.opt.ball_radius, K)  # BxMxK
        node_expanded = keypoints.unsqueeze(3)  # Bx3xMx1
        pc_expanded = x.unsqueeze(2)  # Bx3x1xN
        node_to_point_dist = torch.norm(node_expanded - pc_expanded, p=2, dim=1, keepdim=False).detach()  # BxMxN
        with torch.cuda.device(node_to_point_dist.get_device()):
            node_neighbor_ball_idx = ball_query.forward_cuda_shared_mem(node_to_point_dist, self.opt.ball_radius, K).long()

        node_neighbor_ball_idx_reshape = node_neighbor_ball_idx.unsqueeze(1).expand(B, x_aug.size()[1], M, K).view(B, x_aug.size()[1], M*K)
        x_aug_ball = torch.gather(x_aug, dim=2, index=node_neighbor_ball_idx_reshape).view(B, x_aug.size()[1], M, K)  # Bx4xMxK

        # visualize to confirm the correctness of ball_query
        # pc_np = np.transpose(x[0].detach().cpu().numpy())  # Nx3
        # x_ball_np = np.transpose(x_aug_ball[0, 0:3, :, :].detach().cpu().numpy(), (1, 2, 0))  # MxKx3
        # ax = vis_tools.plot_pc(pc_np, z_cutoff=50, color='b', size=1)
        # for m in range(x_ball_np.shape[0]):
        #     ax.scatter(x_ball_np[m, :, 0].tolist(), x_ball_np[m, :, 1].tolist(), x_ball_np[m, :, 2].tolist(), s=2, c='r')
        #     plt.show()

        # normalize the ball by subtracting the keypoint location
        x_aug_ball[:, 0:3, :, :] = x_aug_ball[:, 0:3, :, :] - keypoints.unsqueeze(3)  # Bx3xMxK
        x_features = x_aug_ball

        y_first = self.conv3(self.conv2(self.conv1(x_features)))  # BxCxMxK
        y_first_max, _ = torch.max(y_first, dim=3, keepdim=True)  # BxCxMx1
        y_first_max = y_first_max.expand(B, y_first.size()[1], M, K)  # BxCxMxK

        y_second = self.conv5(self.conv4(torch.cat((y_first, y_first_max), dim=1)))  # BxCxMxK

        descriptor, _ = torch.max(y_second, dim=3, keepdim=False)  # BxCxM
        descriptor = descriptor / (torch.norm(descriptor, dim=1, keepdim=True) + 1e-5)

        return descriptor, x_features


class DescriptorLiteOldGlobal(nn.Module):
    def __init__(self, opt):
        super(DescriptorLiteOldGlobal, self).__init__()
        self.opt = opt

        in_channels = 3 + opt.surface_normal_len

        self.conv1 = MyConv2d(in_channels, int(opt.descriptor_len/4), kernel_size=(1, 1), stride=1, padding=0, bias=True,
                              activation=self.opt.activation, normalization=self.opt.normalization,
                              momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step, bn_momentum_decay=opt.bn_momentum_decay)
        self.conv2 = MyConv2d(int(opt.descriptor_len/4), int(opt.descriptor_len/2), kernel_size=(1, 1), stride=1, padding=0, bias=True,
                              activation=self.opt.activation, normalization=self.opt.normalization,
                              momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step, bn_momentum_decay=opt.bn_momentum_decay)
        self.conv3 = MyConv2d(int(opt.descriptor_len/2), opt.descriptor_len, kernel_size=(1, 1), stride=1, padding=0, bias=True,
                              activation=self.opt.activation, normalization=self.opt.normalization,
                              momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step, bn_momentum_decay=opt.bn_momentum_decay)

        self.conv4 = MyConv2d(opt.descriptor_len*2, opt.descriptor_len, kernel_size=(1, 1), stride=1, padding=0, bias=True,
                              activation=self.opt.activation, normalization=self.opt.normalization,
                              momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step, bn_momentum_decay=opt.bn_momentum_decay)
        self.conv5 = MyConv2d(opt.descriptor_len, opt.descriptor_len, kernel_size=(1, 1), stride=1, padding=0, bias=True,
                              activation=None, normalization=None)

        # global context from PPFNet
        self.fc1 = EquivariantLayer(opt.descriptor_len * 2, opt.descriptor_len * 2, activation=self.opt.activation,
                                    normalization=self.opt.normalization,
                                    momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step,
                                    bn_momentum_decay=opt.bn_momentum_decay)
        self.fc2 = EquivariantLayer(opt.descriptor_len * 2, opt.descriptor_len, activation=self.opt.activation,
                                    normalization=self.opt.normalization,
                                    momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step,
                                    bn_momentum_decay=opt.bn_momentum_decay)
        self.fc3 = EquivariantLayer(opt.descriptor_len, self.opt.descriptor_len, activation=None, normalization=None)

    def forward(self, x, sn, keypoints, is_train=False, epoch=None):
        '''

        :param x: Bx3xN
        :param sn: Bx*xN
        :param keypoints: Bx3xM, detected keypoints
        :param is_train: bool
        :param epoch:
        :return:
        '''
        # permute x and sn, because later a ball query will be performed
        B, M, N, K = x.size()[0], keypoints.size()[2], x.size()[2], self.opt.ball_nsamples
        permute_idx = torch.from_numpy(np.random.permutation(N).astype(np.int64)).to(x.device)
        x = x[:, :, permute_idx]
        sn = sn[:, :, permute_idx]

        if self.opt.surface_normal_len <= 0:
            x_aug = x
        else:
            x_aug = torch.cat((x, sn), dim=1)  # Bx4xN

        node_expanded = keypoints.unsqueeze(3)  # Bx3xMx1
        pc_expanded = x.unsqueeze(2)  # Bx3x1xN
        node_to_point_dist = torch.norm(node_expanded - pc_expanded, p=2, dim=1, keepdim=False).detach()  # BxMxN
        with torch.cuda.device(node_to_point_dist.get_device()):
            node_neighbor_ball_idx = operations.ball_query_wrapper(x, keypoints, self.opt.ball_radius, K)  # BxMxK
        node_neighbor_ball_idx_reshape = node_neighbor_ball_idx.unsqueeze(1).expand(B, x_aug.size()[1], M, K).view(B, x_aug.size()[1], M*K)
        x_aug_ball = torch.gather(x_aug, dim=2, index=node_neighbor_ball_idx_reshape).view(B, x_aug.size()[1], M, K)  # Bx4xMxK

        # visualize to confirm the correctness of ball_query
        # pc_np = np.transpose(x[0].detach().cpu().numpy())  # Nx3
        # x_ball_np = np.transpose(x_aug_ball[0, 0:3, :, :].detach().cpu().numpy(), (1, 2, 0))  # MxKx3
        # ax = vis_tools.plot_pc(pc_np, z_cutoff=50, color='b', size=1)
        # for m in range(x_ball_np.shape[0]):
        #     ax.scatter(x_ball_np[m, :, 0].tolist(), x_ball_np[m, :, 1].tolist(), x_ball_np[m, :, 2].tolist(), s=2, c='r')
        #     plt.show()

        # normalize the ball by subtracting the keypoint location
        x_aug_ball[:, 0:3, :, :] = x_aug_ball[:, 0:3, :, :] - keypoints.unsqueeze(3)  # Bx3xMxK
        x_features = x_aug_ball

        y_first = self.conv3(self.conv2(self.conv1(x_features)))  # BxCxMxK
        y_first_max, _ = torch.max(y_first, dim=3, keepdim=True)  # BxCxMx1
        y_first_max = y_first_max.expand(B, y_first.size()[1], M, K)  # BxCxMxK

        y_second = self.conv5(self.conv4(torch.cat((y_first, y_first_max), dim=1)))  # BxCxMxK

        descriptor, _ = torch.max(y_second, dim=3, keepdim=False)  # BxCxM

        # global context from PPFNet
        descriptor_global, _ = torch.max(descriptor, dim=2, keepdim=True)  # BxCx1
        descriptor_fusion = torch.cat((descriptor, descriptor_global.expand(B, self.opt.descriptor_len, M)), dim=1)  # Bx2CxM

        descriptor_final = self.fc3(self.fc2(self.fc1(descriptor_fusion)))

        descriptor_final = descriptor_final / (torch.norm(descriptor_final, dim=1, keepdim=True) + 1e-5)

        return descriptor_final, x_features


class RPN_Detector_KNN(nn.Module):
    def __init__(self, opt):
        super(RPN_Detector_KNN, self).__init__()
        self.opt = opt

        self.C1 = 128
        # first PointNet
        self.conv1 = MyConv2d(3+self.opt.surface_normal_len, int(self.C1/2), kernel_size=(1, 1), stride=1, padding=0,
                              bias=True,
                              activation=self.opt.activation, normalization=self.opt.normalization,
                              momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step,
                              bn_momentum_decay=opt.bn_momentum_decay)
        self.conv2 = MyConv2d(int(self.C1/2), int(self.C1/2), kernel_size=(1, 1), stride=1,
                              padding=0, bias=True,
                              activation=self.opt.activation, normalization=self.opt.normalization,
                              momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step,
                              bn_momentum_decay=opt.bn_momentum_decay)
        self.conv3 = MyConv2d(int(self.C1/2), int(self.C1/2), kernel_size=(1, 1), stride=1, padding=0,
                              bias=True,
                              activation=self.opt.activation, normalization=self.opt.normalization,
                              momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step,
                              bn_momentum_decay=opt.bn_momentum_decay)

        self.conv4 = MyConv2d(int(self.C1), int(self.C1), kernel_size=(1, 1), stride=1,
                              padding=0, bias=True,
                              activation=self.opt.activation, normalization=self.opt.normalization,
                              momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step,
                              bn_momentum_decay=opt.bn_momentum_decay)
        self.conv5 = MyConv2d(int(self.C1), int(self.C1), kernel_size=(1, 1), stride=1, padding=0,
                              bias=True,
                              activation=self.opt.activation, normalization=self.opt.normalization,
                              momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step,
                              bn_momentum_decay=opt.bn_momentum_decay)


        assert self.opt.node_knn_k_1 >= 2

        self.C2 = 512
        self.knnlayer_1 = GeneralKNNFusionModule(3 + self.C1, (int(self.C2/2), int(self.C2/2), int(self.C2/2)), (self.C2, self.C2),
                                                 activation=self.opt.activation,
                                                 normalization=self.opt.normalization,
                                                 momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step,
                                                 bn_momentum_decay=opt.bn_momentum_decay)


        # masked max
        # self.masked_max = operations.MaskedMax(self.opt.node_num)

        # --- PointNet to get weight/sigma --- begin --- #
        input_channels = self.C1 + self.C2
        output_channels = 4

        self.mlp1 = EquivariantLayer(input_channels, 512,
                                     activation=self.opt.activation, normalization=self.opt.normalization,
                                     momentum=opt.bn_momentum,
                                     bn_momentum_decay_step=opt.bn_momentum_decay_step,
                                     bn_momentum_decay=opt.bn_momentum_decay)
        self.mlp2 = EquivariantLayer(512, 256,
                                     activation=self.opt.activation, normalization=self.opt.normalization,
                                     momentum=opt.bn_momentum,
                                     bn_momentum_decay_step=opt.bn_momentum_decay_step,
                                     bn_momentum_decay=opt.bn_momentum_decay)
        self.mlp3 = EquivariantLayer(256, output_channels, activation=None, normalization=None)

        self.mlp3.conv.weight.data.normal_(0, 1e-4)
        self.mlp3.conv.bias.data.zero_()
        self.softplus = torch.nn.Softplus()
        # --- PointNet to get weight/sigma --- end --- #

    def forward(self, x, sn, node, is_train=False, epoch=None):
        '''

        :param x: Bx3xN Tensor
        :param sn: Bx3xN Tensor
        :param node: Bx3xM FloatTensor
        :param is_train: determine whether to add noise in KNNModule
        :return:
        '''
        B, M, N = x.size(0), node.size(2), x.size(2)
        x_aug = torch.cat((x, sn), dim=1)
        C0 = x_aug.size(1)
        k = 64

        node_expanded = node.unsqueeze(3)  # Bx3xMx1
        x_expanded = x.unsqueeze(2)  # Bx3x1xN
        node_x_dist = torch.norm(node_expanded - x_expanded, dim=1, keepdim=False)  # Bx3xMxN -> BxMxN
        _, nn_idx = torch.topk(node_x_dist, k=k, dim=2, largest=False, sorted=False)  # BxMxk
        nn_idx_expanded = nn_idx.unsqueeze(1).expand(B, C0, M, k).view(B, C0, M*k)  # Bx1xMxk -> BxCxMxk -> BxCx(M*k)
        x_aug_knn = torch.gather(x_aug, dim=2, index=nn_idx_expanded).view(B, C0, M, k)  # BxCxN -> BxCxM*k -> BxCxMxk

        # normalize by substracting the node
        x_aug_knn[:, 0:3, :, :] = x_aug_knn[:, 0:3, :, :] - node.unsqueeze(3)  # BxCxMxk

        first_pn_out = self.conv3(self.conv2(self.conv1(x_aug_knn)))  # BxCxMxk
        first_pn_out_max, _ = torch.max(first_pn_out, dim=3, keepdim=True)  # BxCxMx1
        first_pn_out_max = first_pn_out_max.expand(B, first_pn_out_max.size(1), M, k)  # BxCxMxk

        second_pn_out = self.conv5(self.conv4(torch.cat((first_pn_out, first_pn_out_max), dim=1)))  # BxCxMxk
        second_pn_out_max, _ = torch.max(second_pn_out, dim=3, keepdim=False)  # BxCxM

        # knn module, knn search on nodes: ----------------------------------
        knn_feature_1 = self.knnlayer_1(query=node,
                                        database=node,
                                        x=second_pn_out_max,
                                        K=self.opt.node_knn_k_1,
                                        epoch=epoch)

        node_feature_aggregated = torch.cat((second_pn_out_max, knn_feature_1), dim=1)  # Bx(C1+C2)xM

        # go through another network to calculate the per-node keypoint & uncertainty sigma
        y = self.mlp1(node_feature_aggregated)
        point_descriptor = self.mlp2(y)
        keypoint_sigma = self.mlp3(point_descriptor)  # Bx(3+1)xkN

        # keypoint = keypoint + node_coordinate
        keypoints = keypoint_sigma[:, 0:3, :] + node  # Bx3xM
        # make sure sigma>=0 by square
        # sigmas = torch.pow(keypoint_sigma[:, 3, :], exponent=2) + self.opt.loss_sigma_lower_bound  # BxM
        sigmas = self.softplus(keypoint_sigma[:, 3, :]) + self.opt.loss_sigma_lower_bound  # BxM

        descriptors = None

        # debug
        # print(keypoints)
        # print(sigmas)

        return node, keypoints, sigmas, descriptors


class RPN_Detector_Ball(nn.Module):
    def __init__(self, opt):
        super(RPN_Detector_Ball, self).__init__()
        self.opt = opt

        self.C1 = 128
        # first PointNet
        self.conv1 = MyConv2d(3+self.opt.surface_normal_len, int(self.C1/2), kernel_size=(1, 1), stride=1, padding=0,
                              bias=True,
                              activation=self.opt.activation, normalization=self.opt.normalization,
                              momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step,
                              bn_momentum_decay=opt.bn_momentum_decay)
        self.conv2 = MyConv2d(int(self.C1/2), int(self.C1/2), kernel_size=(1, 1), stride=1,
                              padding=0, bias=True,
                              activation=self.opt.activation, normalization=self.opt.normalization,
                              momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step,
                              bn_momentum_decay=opt.bn_momentum_decay)
        self.conv3 = MyConv2d(int(self.C1/2), int(self.C1/2), kernel_size=(1, 1), stride=1, padding=0,
                              bias=True,
                              activation=self.opt.activation, normalization=self.opt.normalization,
                              momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step,
                              bn_momentum_decay=opt.bn_momentum_decay)

        self.conv4 = MyConv2d(int(self.C1), int(self.C1), kernel_size=(1, 1), stride=1,
                              padding=0, bias=True,
                              activation=self.opt.activation, normalization=self.opt.normalization,
                              momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step,
                              bn_momentum_decay=opt.bn_momentum_decay)
        self.conv5 = MyConv2d(int(self.C1), int(self.C1), kernel_size=(1, 1), stride=1, padding=0,
                              bias=True,
                              activation=self.opt.activation, normalization=self.opt.normalization,
                              momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step,
                              bn_momentum_decay=opt.bn_momentum_decay)

        assert self.opt.node_knn_k_1 >= 2

        self.C2 = 512
        self.knnlayer_1 = GeneralKNNFusionModule(3 + self.C1, (int(self.C2/2), int(self.C2/2), int(self.C2/2)), (self.C2, self.C2),
                                                 activation=self.opt.activation,
                                                 normalization=self.opt.normalization,
                                                 momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step,
                                                 bn_momentum_decay=opt.bn_momentum_decay)


        # masked max
        # self.masked_max = operations.MaskedMax(self.opt.node_num)

        # --- PointNet to get weight/sigma --- begin --- #
        input_channels = self.C1 + self.C2
        output_channels = 4

        self.mlp1 = EquivariantLayer(input_channels, 512,
                                     activation=self.opt.activation, normalization=self.opt.normalization,
                                     momentum=opt.bn_momentum,
                                     bn_momentum_decay_step=opt.bn_momentum_decay_step,
                                     bn_momentum_decay=opt.bn_momentum_decay)
        self.mlp2 = EquivariantLayer(512, 256,
                                     activation=self.opt.activation, normalization=self.opt.normalization,
                                     momentum=opt.bn_momentum,
                                     bn_momentum_decay_step=opt.bn_momentum_decay_step,
                                     bn_momentum_decay=opt.bn_momentum_decay)
        self.mlp3 = EquivariantLayer(256, output_channels, activation=None, normalization=None)

        self.mlp3.conv.weight.data.normal_(0, 1e-4)
        self.mlp3.conv.bias.data.zero_()
        self.softplus = torch.nn.Softplus()
        # --- PointNet to get weight/sigma --- end --- #

    def forward(self, x, sn, node, is_train=False, epoch=None):
        '''

        :param x: Bx3xN Tensor
        :param sn: Bx3xN Tensor
        :param node: Bx3xM FloatTensor
        :param is_train: determine whether to add noise in KNNModule
        :return:
        '''
        B, M, N = x.size(0), node.size(2), x.size(2)
        x_aug = torch.cat((x, sn), dim=1)
        C0 = x_aug.size(1)
        k = 64
        radius = 2

        node_expanded = node.unsqueeze(3)  # Bx3xMx1
        x_expanded = x.unsqueeze(2)  # Bx3x1xN
        node_x_dist = torch.norm(node_expanded - x_expanded, dim=1, keepdim=False)  # Bx3xMxN -> BxMxN
        with torch.cuda.device(node_x_dist.get_device()):
            node_neighbor_ball_idx = ball_query.forward_cuda_shared_mem(node_x_dist, radius, k).long()
        node_neighbor_ball_idx_reshape = node_neighbor_ball_idx.unsqueeze(1).expand(B, x_aug.size()[1], M, k).view(B, x_aug.size()[1], M * k)
        x_aug_ball = torch.gather(x_aug, dim=2, index=node_neighbor_ball_idx_reshape).view(B, x_aug.size()[1], M, k)  # BxCxMxK

        # normalize by substracting the node
        x_aug_ball[:, 0:3, :, :] = x_aug_ball[:, 0:3, :, :] - node.unsqueeze(3)  # BxCxMxk

        first_pn_out = self.conv3(self.conv2(self.conv1(x_aug_ball)))  # BxCxMxk
        first_pn_out_max, _ = torch.max(first_pn_out, dim=3, keepdim=True)  # BxCxMx1
        first_pn_out_max = first_pn_out_max.expand(B, first_pn_out_max.size(1), M, k)  # BxCxMxk

        second_pn_out = self.conv5(self.conv4(torch.cat((first_pn_out, first_pn_out_max), dim=1)))  # BxCxMxk
        second_pn_out_max, _ = torch.max(second_pn_out, dim=3, keepdim=False)  # BxCxM

        # knn module, knn search on nodes: ----------------------------------
        knn_feature_1 = self.knnlayer_1(query=node,
                                        database=node,
                                        x=second_pn_out_max,
                                        K=self.opt.node_knn_k_1,
                                        epoch=epoch)

        node_feature_aggregated = torch.cat((second_pn_out_max, knn_feature_1), dim=1)  # Bx(C1+C2)xM

        # go through another network to calculate the per-node keypoint & uncertainty sigma
        y = self.mlp1(node_feature_aggregated)
        point_descriptor = self.mlp2(y)
        keypoint_sigma = self.mlp3(point_descriptor)  # Bx(3+1)xkN

        # keypoint = keypoint + node_coordinate
        keypoints = keypoint_sigma[:, 0:3, :] + node  # Bx3xM
        # make sure sigma>=0 by square
        # sigmas = torch.pow(keypoint_sigma[:, 3, :], exponent=2) + self.opt.loss_sigma_lower_bound  # BxM
        sigmas = self.softplus(keypoint_sigma[:, 3, :]) + self.opt.loss_sigma_lower_bound  # BxM

        descriptors = None

        # debug
        # print(keypoints)
        # print(sigmas)

        return node, keypoints, sigmas, descriptors

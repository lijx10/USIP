import torch
import torch.nn as nn
import numpy as np
import math
from collections import OrderedDict
import os
import random

from models import networks
from models import losses
from data import augmentation


class ModelDescriptor():
    def __init__(self, opt):
        self.opt = opt

        self.descriptor = networks.DescriptorLiteOld(opt)
        self.triplet_criteria = losses.DescPairScanLoss(opt)

        if self.opt.gpu_ids[0] >= 0:
            self.descriptor = self.descriptor.to(opt.device)
            self.triplet_criteria = self.triplet_criteria.to(opt.device)

        # multi-gpu training
        if len(opt.gpu_ids) > 1:
            self.descriptor = nn.DataParallel(self.descriptor, device_ids=opt.gpu_ids)
            self.triplet_criteria = nn.DataParallel(self.triplet_criteria, device_ids=opt.gpu_ids)

        # learning rate_control
        self.old_lr_descriptor = self.opt.lr

        self.optimizer_descriptor = torch.optim.Adam(self.descriptor.parameters(),
                                                     lr=self.old_lr_descriptor,
                                                     betas=(0.9, 0.999),
                                                     weight_decay=0)

        if self.opt.gpu_ids[0] >= 0:
            self.descriptor = self.descriptor.to(self.opt.device)

        # place holder for GPU tensors
        self.anc_pc = torch.FloatTensor(self.opt.batch_size, 3, self.opt.input_pc_num).uniform_()
        self.anc_sn = torch.FloatTensor(self.opt.batch_size, 3, self.opt.input_pc_num).uniform_()
        self.anc_keypoints = torch.FloatTensor(self.opt.batch_size, 3, self.opt.node_num)
        self.anc_sigmas = torch.FloatTensor(self.opt.batch_size, self.opt.node_num)

        self.pos_pc = torch.FloatTensor(self.opt.batch_size, 3, self.opt.input_pc_num).uniform_()
        self.pos_sn = torch.FloatTensor(self.opt.batch_size, 3, self.opt.input_pc_num).uniform_()
        self.pos_keypoints = torch.FloatTensor(self.opt.batch_size, 3, self.opt.node_num)
        self.pos_sigmas = torch.FloatTensor(self.opt.batch_size, self.opt.node_num)

        # neg_idx to build negative samples
        self.neg_idx = torch.LongTensor(self.opt.batch_size)

        # record the test loss and accuracy
        self.test_loss_average = torch.tensor([0], dtype=torch.float32, requires_grad=False)
        self.test_loss_triplet_average = torch.tensor([0], dtype=torch.float32, requires_grad=False)
        self.test_active_percentage_average = torch.tensor([0], dtype=torch.float32, requires_grad=False)

        if self.opt.gpu_ids[0] >= 0:
            self.anc_pc = self.anc_pc.to(self.opt.device)
            self.anc_sn = self.anc_sn.to(self.opt.device)
            self.anc_keypoints = self.anc_keypoints.to(self.opt.device)
            self.anc_sigmas = self.anc_sigmas.to(self.opt.device)

            self.pos_pc = self.pos_pc.to(self.opt.device)
            self.pos_sn = self.pos_sn.to(self.opt.device)
            self.pos_keypoints = self.pos_keypoints.to(self.opt.device)
            self.pos_sigmas = self.pos_sigmas.to(self.opt.device)

            self.neg_idx = self.neg_idx.to(self.opt.device)

            self.test_loss_average = self.test_loss_average.to(self.opt.device)
            self.test_loss_triplet_average = self.test_loss_triplet_average.to(self.opt.device)
            self.test_active_percentage_average = self.test_active_percentage_average.to(self.opt.device)

    def set_input(self,
                  anc_pc, anc_sn, anc_keypoints, anc_sigmas,
                  pos_pc, pos_sn, pos_keypoints, pos_sigmas,
                  neg_idx):
        # self.anc_pc.resize_(anc_pc.size()).copy_(anc_pc).detach()
        # self.anc_pc = anc_pc.to(self.opt.device).detach()
        # self.anc_sn.resize_(anc_sn.size()).copy_(anc_sn).detach()
        # self.anc_keypoints.resize_(anc_keypoints.size()).copy_(anc_keypoints.detach())
        # self.anc_sigmas.resize_(anc_sigmas.size()).copy_(anc_sigmas.detach()).to(self.opt.device)
        #
        # self.pos_pc.resize_(pos_pc.size()).copy_(pos_pc).detach()
        # self.pos_pc = pos_pc.to(self.opt.device).detach()
        # self.pos_sn.resize_(pos_sn.size()).copy_(pos_sn).detach()
        # self.pos_keypoints.resize_(pos_keypoints.size()).copy_(pos_keypoints.detach())
        # self.pos_sigmas.resize_(pos_sigmas.size()).copy_(pos_sigmas.detach())
        #
        # self.neg_idx.resize_(neg_idx.size()).copy_(neg_idx.detach())

        self.anc_pc = anc_pc.float().to(self.opt.device)
        self.anc_sn = anc_sn.float().to(self.opt.device)
        self.anc_keypoints = anc_keypoints.float().to(self.opt.device)
        self.anc_sigmas = anc_sigmas.float().to(self.opt.device)

        self.pos_pc = pos_pc.float().to(self.opt.device)
        self.pos_sn = pos_sn.float().to(self.opt.device)
        self.pos_keypoints = pos_keypoints.float().to(self.opt.device)
        self.pos_sigmas = pos_sigmas.float().to(self.opt.device)

        self.neg_idx = neg_idx.long().to(self.opt.device)

        torch.cuda.synchronize()

    def forward(self, pc, sn, keypoints, is_train=False, epoch=None):
        with torch.cuda.device(pc.get_device()):
            descriptors, x_aug_ball = self.descriptor(pc, sn, keypoints, is_train, epoch)  # Bx1024
            return descriptors, x_aug_ball

    def forward_siamese(self, pc_tuple, sn_tuple, keypoints_tuple, is_train=False, epoch=None):
        size_of_single_chunk = pc_tuple[0].size()[0]
        descriptors, x_aug_ball = self.descriptor(torch.cat(pc_tuple, dim=0),
                                                  torch.cat(sn_tuple, dim=0),
                                                  torch.cat(keypoints_tuple, dim=0),
                                                  is_train, epoch)  # Bx1024

        descriptors_tuple = torch.split(descriptors, split_size_or_sections=size_of_single_chunk, dim=0)
        x_aug_ball_tuple = torch.split(x_aug_ball, split_size_or_sections=size_of_single_chunk, dim=0)

        return descriptors_tuple, x_aug_ball_tuple

    def optimize(self, epoch=None):
        with torch.cuda.device(self.anc_pc.get_device()):
            # random point dropout
            if self.opt.random_pc_dropout_lower_limit < 0.99:
                dropout_keep_ratio = random.uniform(self.opt.random_pc_dropout_lower_limit, 1.0)
                resulting_pc_num = round(dropout_keep_ratio * self.opt.input_pc_num)
                chosen_indices = np.random.choice(self.opt.input_pc_num, resulting_pc_num, replace=False)
                chosen_indices_tensor = torch.from_numpy(chosen_indices).to(self.opt.device)
                self.anc_pc = torch.index_select(self.anc_pc, dim=2, index=chosen_indices_tensor)
                self.anc_sn = torch.index_select(self.anc_sn, dim=2, index=chosen_indices_tensor)
                self.pos_pc = torch.index_select(self.pos_pc, dim=2, index=chosen_indices_tensor)
                self.pos_sn = torch.index_select(self.pos_sn, dim=2, index=chosen_indices_tensor)

            self.descriptor.train()

            (self.anc_descriptors, self.pos_descriptors), (anc_aug_ball, pos_aug_ball) = self.forward_siamese(
                (self.anc_pc, self.pos_pc),
                (self.anc_sn, self.pos_sn),
                (self.anc_keypoints, self.pos_keypoints),
                is_train=True, epoch=epoch)  # BxCxM, Bx4xMxK

            self.descriptor.zero_grad()

            triplet_loss, active_percentage = self.triplet_criteria(self.anc_descriptors, self.pos_descriptors,
                                                                    self.anc_descriptors[self.neg_idx, :, :],
                                                                    self.anc_sigmas)
            self.triplet_loss = torch.mean(triplet_loss)
            self.active_percentage = torch.mean(active_percentage)

            self.loss = self.triplet_loss

            self.loss.backward()

            self.optimizer_descriptor.step()

    def test_model(self):
        self.descriptor.eval()

        (self.anc_descriptors, self.pos_descriptors), (anc_aug_ball, pos_aug_ball) = self.forward_siamese(
            (self.anc_pc, self.pos_pc),
            (self.anc_sn, self.pos_sn),
            (self.anc_keypoints, self.pos_keypoints),
            is_train=False, epoch=None)  # BxCxM, Bx4xMxK

        triplet_loss, active_percentage = self.triplet_criteria(self.anc_descriptors, self.pos_descriptors,
                                                                self.anc_descriptors[self.neg_idx, :, :],
                                                                self.anc_sigmas)
        self.triplet_loss = torch.mean(triplet_loss)
        self.active_percentage = torch.mean(active_percentage)

        self.loss = self.triplet_loss

    def freeze_model(self):
        for p in self.descriptor.parameters():
            p.requires_grad = False

    def run_model(self, pc, sn, keypoints):
        self.descriptor.eval()
        with torch.no_grad():
            descriptors, _ = self.descriptor(pc, sn, keypoints, False, None)  # Bx128
        return descriptors

    def get_negative_samples(self):
        '''
        given neg_idx
        :return: neg_pc, neg_sn, neg_keypoints, neg_sigmas
        '''
        neg_pc = self.anc_pc[self.neg_idx, :, :]  # Bx3xN
        neg_sn = self.anc_sn[self.neg_idx, :, :]  # Bx*xN
        neg_keypoints = self.anc_keypoints[self.neg_idx, :, :]  # Bx3xM
        neg_sigmas = self.anc_sigmas[self.neg_idx, :]  # BxM
        
        return neg_pc, neg_sn, neg_keypoints, neg_sigmas

    @staticmethod
    def build_pc_node_keypoint_visual(pc_np, keypoint_np=None, keypoint_other_np=None, sigmas_np=None,
                                      sigmas_other_np=None):
        pc_color_np = np.repeat(np.expand_dims(np.array([191, 191, 191], dtype=np.int64), axis=0),
                                pc_np.shape[0],
                                axis=0)  # 1x3 -> Nx3
        if keypoint_np is not None:
            keypoint_color_np = np.repeat(np.expand_dims(np.array([255, 0, 0], dtype=np.int64), axis=0),
                                          keypoint_np.shape[0],
                                          axis=0)  # 1x3 -> Kx3
            # consider the sigma
            if sigmas_np is not None:
                sigmas_normalized_np = (1.0 / sigmas_np) / np.max(1.0 / sigmas_np)  # K
                keypoint_color_np = keypoint_color_np * np.expand_dims(sigmas_normalized_np, axis=1)  # Kx3
                keypoint_color_np = keypoint_color_np.astype(np.int32)
        if keypoint_other_np is not None:
            keypoint_other_color_np = np.repeat(np.expand_dims(np.array([0, 0, 255], dtype=np.int64), axis=0),
                                                keypoint_other_np.shape[0],
                                                axis=0)  # 1x3 -> Kx3
            # consider the sigma
            if sigmas_other_np is not None:
                sigmas_other_normalized_np = (1.0 / sigmas_other_np) / np.max(1.0 / sigmas_other_np)  # K
                keypoint_other_color_np = keypoint_other_color_np * np.expand_dims(sigmas_other_normalized_np,
                                                                                   axis=1)  # Kx3
                keypoint_other_color_np = keypoint_other_color_np.astype(np.int32)

        # pc_vis_np = np.concatenate((pc_np), axis=0)
        # pc_vis_color_np = np.concatenate((pc_color_np), axis=0)
        pc_vis_np = pc_np
        pc_vis_color_np = pc_color_np
        if keypoint_np is not None:
            # print(pc_vis_np.shape)
            # print(keypoint_np.shape)
            pc_vis_np = np.concatenate((pc_vis_np, keypoint_np), axis=0)
            pc_vis_color_np = np.concatenate((pc_vis_color_np, keypoint_color_np), axis=0)
        if keypoint_other_np is not None:
            pc_vis_np = np.concatenate((pc_vis_np, keypoint_other_np), axis=0)
            pc_vis_color_np = np.concatenate((pc_vis_color_np, keypoint_other_color_np), axis=0)

        return pc_vis_np, pc_vis_color_np

    # visualization with visdom
    def get_current_visuals(self):
        # build pc & node & keypoint with marker color
        anc_pc_np = self.anc_pc[0].cpu().numpy().transpose()  # Nx3
        anc_keypoint_np = self.anc_keypoints[0].detach().cpu().numpy().transpose()  # Kx3
        anc_sigmas_np = self.anc_sigmas[0].detach().cpu().numpy()  # K

        # rotate to get better visualization if it is 2d rotation augmentation
        if self.opt.rot_3d == False and self.opt.rot_horizontal == True:
            [anc_pc_np, anc_keypoint_np] = augmentation.rotate_point_cloud_list_3d(
                [anc_pc_np, anc_keypoint_np],
                angles=[math.pi / 4, 0, 0])
        anc_data_vis_np, anc_data_vis_color_np = self.build_pc_node_keypoint_visual(anc_pc_np,
                                                                                    anc_keypoint_np,
                                                                                    sigmas_np=anc_sigmas_np)

        pos_pc_np = self.pos_pc[0].cpu().numpy().transpose()  # Nx3
        pos_keypoint_np = self.pos_keypoints[0].detach().cpu().numpy().transpose()  # Kx3
        pos_sigmas_np = self.pos_sigmas[0].detach().cpu().numpy()  # K

        # rotate to get better visualization if it is 2d rotation augmentation
        if self.opt.rot_3d == False and self.opt.rot_horizontal == True:
            [pos_pc_np, pos_keypoint_np] = augmentation.rotate_point_cloud_list_3d(
                [pos_pc_np, pos_keypoint_np],
                angles=[math.pi / 4, 0, 0])
        pos_data_vis_np, pos_data_vis_color_np = self.build_pc_node_keypoint_visual(pos_pc_np,
                                                                                    pos_keypoint_np,
                                                                                    sigmas_np=pos_sigmas_np)

        neg_pc, neg_sn, neg_keypoints, neg_sigmas = self.get_negative_samples()
        neg_pc_np = neg_pc[0].cpu().numpy().transpose()  # Nx3
        neg_keypoint_np = neg_keypoints[0].detach().cpu().numpy().transpose()  # Kx3
        neg_sigmas_np = neg_sigmas[0].detach().cpu().numpy()  # K

        # rotate to get better visualization if it is 2d rotation augmentation
        if self.opt.rot_3d == False and self.opt.rot_horizontal == True:
            [neg_pc_np, neg_keypoint_np] = augmentation.rotate_point_cloud_list_3d(
                [neg_pc_np, neg_keypoint_np],
                angles=[math.pi / 4, 0, 0])
        neg_data_vis_np, neg_data_vis_color_np = self.build_pc_node_keypoint_visual(neg_pc_np,
                                                                                    neg_keypoint_np,
                                                                                    sigmas_np=neg_sigmas_np)

        return OrderedDict([('anc_data_vis', (anc_data_vis_np, anc_data_vis_color_np)),
                            ('pos_data_vis', (pos_data_vis_np, pos_data_vis_color_np)),
                            ('neg_data_vis', (neg_data_vis_np, neg_data_vis_color_np))])


    def get_current_errors(self):
        return OrderedDict([
            ('O_loss', self.loss.item()),
            ('O_triplet', self.triplet_loss.item()),
            ('O_active_perc', self.active_percentage.item()),
            ('E_loss', self.test_loss_average.item()),
            ('E_triplet', self.test_loss_triplet_average.item()),
            ('E_active_perc', self.test_active_percentage_average.item())
        ])

    def save_network(self, network, network_label, epoch_label, gpu_id):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.opt.checkpoints_dir, save_filename)
        torch.save(network.state_dict(), save_path)
        # if gpu_id >= 0 and torch.cuda.is_available():
        #     # torch.cuda.device(gpu_id)
        #     network.to(self.opt.device)

    def update_learning_rate(self, ratio):
        lr_clip = 0.00001

        # descriptor
        lr_descriptor = self.old_lr_descriptor * ratio
        if lr_descriptor < lr_clip:
            lr_descriptor = lr_clip
        for param_group in self.optimizer_descriptor.param_groups:
            param_group['lr'] = lr_descriptor

        print('update descriptor learning rate: %f -> %f' % (self.old_lr_descriptor, lr_descriptor))
        self.old_lr_descriptor = lr_descriptor


class ModelDescriptorIndoor():
    def __init__(self, opt):
        self.opt = opt

        self.descriptor = networks.DescriptorLiteOldGlobal(opt)
        self.triplet_criteria = losses.DescCGFLoss(opt)

        if self.opt.gpu_ids[0] >= 0:
            self.descriptor = self.descriptor.to(opt.device)
            self.triplet_criteria = self.triplet_criteria.to(opt.device)

        # multi-gpu training
        if len(opt.gpu_ids) > 1:
            self.descriptor = nn.DataParallel(self.descriptor, device_ids=opt.gpu_ids)
            self.triplet_criteria = nn.DataParallel(self.triplet_criteria, device_ids=opt.gpu_ids)

        # learning rate_control
        self.old_lr_descriptor = self.opt.lr

        self.optimizer_descriptor = torch.optim.Adam(self.descriptor.parameters(),
                                                     lr=self.old_lr_descriptor,
                                                     betas=(0.9, 0.999),
                                                     weight_decay=0)

        if self.opt.gpu_ids[0] >= 0:
            self.descriptor = self.descriptor.to(self.opt.device)

        # place holder for GPU tensors
        self.anc_pc = torch.FloatTensor(self.opt.batch_size, 3, self.opt.input_pc_num).uniform_()
        self.anc_sn = torch.FloatTensor(self.opt.batch_size, 3, self.opt.input_pc_num).uniform_()
        self.anc_keypoints = torch.FloatTensor(self.opt.batch_size, 3, self.opt.node_num)
        self.anc_sigmas = torch.FloatTensor(self.opt.batch_size, self.opt.node_num)

        self.pos_pc = torch.FloatTensor(self.opt.batch_size, 3, self.opt.input_pc_num).uniform_()
        self.pos_sn = torch.FloatTensor(self.opt.batch_size, 3, self.opt.input_pc_num).uniform_()
        self.pos_keypoints = torch.FloatTensor(self.opt.batch_size, 3, self.opt.node_num)
        self.pos_sigmas = torch.FloatTensor(self.opt.batch_size, self.opt.node_num)

        self.anc_R_pos = torch.zeros((self.opt.batch_size, 3, 3), dtype=torch.float32)
        self.anc_scale_pos = torch.zeros((self.opt.batch_size, 1), dtype=torch.float32)
        self.anc_shift_pos = torch.zeros((self.opt.batch_size, 3, 1), dtype=torch.float32)

        # record the test loss and accuracy
        self.test_loss_average = torch.tensor([0], dtype=torch.float32, requires_grad=False)
        self.test_loss_triplet_average = torch.tensor([0], dtype=torch.float32, requires_grad=False)
        self.test_active_percentage_average = torch.tensor([0], dtype=torch.float32, requires_grad=False)

        if self.opt.gpu_ids[0] >= 0:
            self.anc_pc = self.anc_pc.to(self.opt.device)
            self.anc_sn = self.anc_sn.to(self.opt.device)
            self.anc_keypoints = self.anc_keypoints.to(self.opt.device)
            self.anc_sigmas = self.anc_sigmas.to(self.opt.device)

            self.pos_pc = self.pos_pc.to(self.opt.device)
            self.pos_sn = self.pos_sn.to(self.opt.device)
            self.pos_keypoints = self.pos_keypoints.to(self.opt.device)
            self.pos_sigmas = self.pos_sigmas.to(self.opt.device)

            self.anc_R_pos = self.anc_R_pos.to(self.opt.device)
            self.anc_scale_pos = self.anc_scale_pos.to(self.opt.device)
            self.anc_shift_pos = self.anc_shift_pos.to(self.opt.device)

            self.test_loss_average = self.test_loss_average.to(self.opt.device)
            self.test_loss_triplet_average = self.test_loss_triplet_average.to(self.opt.device)
            self.test_active_percentage_average = self.test_active_percentage_average.to(self.opt.device)

    def set_input(self,
                  anc_pc, anc_sn, anc_keypoints, anc_sigmas,
                  pos_pc, pos_sn, pos_keypoints, pos_sigmas,
                  anc_R_pos, anc_scale_pos, anc_shift_pos):
        self.anc_pc.resize_(anc_pc.size()).copy_(anc_pc).detach()
        self.anc_pc = anc_pc.to(self.opt.device).detach()
        self.anc_sn.resize_(anc_sn.size()).copy_(anc_sn).detach()
        self.anc_keypoints.resize_(anc_keypoints.size()).copy_(anc_keypoints.detach())
        self.anc_sigmas.resize_(anc_sigmas.size()).copy_(anc_sigmas.detach())

        self.pos_pc.resize_(pos_pc.size()).copy_(pos_pc).detach()
        self.pos_pc = pos_pc.to(self.opt.device).detach()
        self.pos_sn.resize_(pos_sn.size()).copy_(pos_sn).detach()
        self.pos_keypoints.resize_(pos_keypoints.size()).copy_(pos_keypoints.detach())
        self.pos_sigmas.resize_(pos_sigmas.size()).copy_(pos_sigmas.detach())

        self.anc_R_pos.resize_(anc_R_pos.size()).copy_(anc_R_pos).detach()
        self.anc_scale_pos.resize_(anc_scale_pos.size()).copy_(anc_scale_pos).detach()
        self.anc_shift_pos.resize_(anc_shift_pos.size()).copy_(anc_shift_pos).detach()

        torch.cuda.synchronize()

    def forward(self, pc, sn, keypoints, is_train=False, epoch=None):
        descriptors, x_aug_ball = self.descriptor(pc, sn, keypoints, is_train, epoch)  # Bx1024
        return descriptors, x_aug_ball

    def forward_siamese(self, pc_tuple, sn_tuple, keypoints_tuple, is_train=False, epoch=None):
        size_of_single_chunk = pc_tuple[0].size()[0]
        descriptors, x_aug_ball = self.descriptor(torch.cat(pc_tuple, dim=0),
                                                  torch.cat(sn_tuple, dim=0),
                                                  torch.cat(keypoints_tuple, dim=0),
                                                  is_train, epoch)  # Bx1024

        descriptors_tuple = torch.split(descriptors, split_size_or_sections=size_of_single_chunk, dim=0)
        x_aug_ball_tuple = torch.split(x_aug_ball, split_size_or_sections=size_of_single_chunk, dim=0)

        return descriptors_tuple, x_aug_ball_tuple

    def optimize(self, epoch=None):
        # random point dropout
        if self.opt.random_pc_dropout_lower_limit < 0.99:
            dropout_keep_ratio = random.uniform(self.opt.random_pc_dropout_lower_limit, 1.0)
            resulting_pc_num = round(dropout_keep_ratio * self.opt.input_pc_num)
            chosen_indices = np.random.choice(self.opt.input_pc_num, resulting_pc_num, replace=False)
            chosen_indices_tensor = torch.from_numpy(chosen_indices).to(self.opt.device)
            self.anc_pc = torch.index_select(self.anc_pc, dim=2, index=chosen_indices_tensor)
            self.anc_sn = torch.index_select(self.anc_sn, dim=2, index=chosen_indices_tensor)
            self.pos_pc = torch.index_select(self.pos_pc, dim=2, index=chosen_indices_tensor)
            self.pos_sn = torch.index_select(self.pos_sn, dim=2, index=chosen_indices_tensor)

        self.descriptor.train()

        (self.anc_descriptors, self.pos_descriptors), (anc_aug_ball, pos_aug_ball) = self.forward_siamese(
            (self.anc_pc, self.pos_pc),
            (self.anc_sn, self.pos_sn),
            (self.anc_keypoints, self.pos_keypoints),
            is_train=True, epoch=epoch)  # BxCxM, Bx4xMxK

        self.descriptor.zero_grad()

        # transform anc to self.anc_keypoints_transformed
        self.anc_keypoints_transformed = torch.matmul(self.anc_R_pos, self.anc_keypoints)  # Bx3x3 * Bx3xM -> Bx3xM
        self.anc_keypoints_transformed = self.anc_keypoints_transformed * self.anc_scale_pos.unsqueeze(1).unsqueeze(2)  # Bx3xM * Bx1x1 -> Bx3xM
        self.anc_keypoints_transformed = self.anc_keypoints_transformed + self.anc_shift_pos  # Bx3xM + Bx3x1 -> Bx3xM

        anc_pc_transformed = torch.matmul(self.anc_R_pos, self.anc_pc)  # Bx3x3 * Bx3xM -> Bx3xM
        anc_pc_transformed = anc_pc_transformed * self.anc_scale_pos.unsqueeze(1).unsqueeze(2)  # Bx3xM * Bx1x1 -> Bx3xM
        anc_pc_transformed = anc_pc_transformed + self.anc_shift_pos  # Bx3xM + Bx3x1 -> Bx3xM

        triplet_loss, active_percentage = self.triplet_criteria(self.anc_keypoints_transformed, self.anc_descriptors,
                                                                self.pos_keypoints, self.pos_descriptors,
                                                                self.anc_sigmas,
                                                                anc_pc_transformed, self.pos_pc)
        self.triplet_loss = torch.mean(triplet_loss)
        self.active_percentage = torch.mean(active_percentage)

        self.loss = self.triplet_loss

        self.loss.backward()

        self.optimizer_descriptor.step()

    def test_model(self):
        self.descriptor.eval()

        (self.anc_descriptors, self.pos_descriptors), (anc_aug_ball, pos_aug_ball) = self.forward_siamese(
            (self.anc_pc, self.pos_pc),
            (self.anc_sn, self.pos_sn),
            (self.anc_keypoints, self.pos_keypoints),
            is_train=False, epoch=None)  # BxCxM, Bx4xMxK

        # transform anc to self.anc_keypoints_transformed
        self.anc_keypoints_transformed = torch.matmul(self.anc_R_pos, self.anc_keypoints)  # Bx3x3 * Bx3xM -> Bx3xM
        self.anc_keypoints_transformed = self.anc_keypoints_transformed * self.anc_scale_pos.unsqueeze(1).unsqueeze(2)  # Bx3xM * Bx1x1 -> Bx3xM
        self.anc_keypoints_transformed = self.anc_keypoints_transformed + self.anc_shift_pos  # Bx3xM + Bx3x1 -> Bx3xM

        triplet_loss, active_percentage = self.triplet_criteria(self.anc_keypoints_transformed, self.anc_descriptors,
                                                                self.pos_keypoints, self.pos_descriptors,
                                                                self.anc_sigmas)
        self.triplet_loss = torch.mean(triplet_loss)
        self.active_percentage = torch.mean(active_percentage)

        self.loss = self.triplet_loss

    def freeze_model(self):
        for p in self.descriptor.parameters():
            p.requires_grad = False

    def run_model(self, pc, sn, keypoints):
        self.descriptor.eval()
        with torch.no_grad():
            descriptors, _ = self.descriptor(pc, sn, keypoints, False, None)  # Bx128
        return descriptors

    @staticmethod
    def build_pc_node_keypoint_visual(pc_np, keypoint_np=None, keypoint_other_np=None, sigmas_np=None,
                                      sigmas_other_np=None):
        pc_color_np = np.repeat(np.expand_dims(np.array([191, 191, 191], dtype=np.int64), axis=0),
                                pc_np.shape[0],
                                axis=0)  # 1x3 -> Nx3
        if keypoint_np is not None:
            keypoint_color_np = np.repeat(np.expand_dims(np.array([255, 0, 0], dtype=np.int64), axis=0),
                                          keypoint_np.shape[0],
                                          axis=0)  # 1x3 -> Kx3
            # consider the sigma
            if sigmas_np is not None:
                sigmas_normalized_np = (1.0 / sigmas_np) / np.max(1.0 / sigmas_np)  # K
                keypoint_color_np = keypoint_color_np * np.expand_dims(sigmas_normalized_np, axis=1)  # Kx3
                keypoint_color_np = keypoint_color_np.astype(np.int32)
        if keypoint_other_np is not None:
            keypoint_other_color_np = np.repeat(np.expand_dims(np.array([0, 0, 255], dtype=np.int64), axis=0),
                                                keypoint_other_np.shape[0],
                                                axis=0)  # 1x3 -> Kx3
            # consider the sigma
            if sigmas_other_np is not None:
                sigmas_other_normalized_np = (1.0 / sigmas_other_np) / np.max(1.0 / sigmas_other_np)  # K
                keypoint_other_color_np = keypoint_other_color_np * np.expand_dims(sigmas_other_normalized_np,
                                                                                   axis=1)  # Kx3
                keypoint_other_color_np = keypoint_other_color_np.astype(np.int32)

        # pc_vis_np = np.concatenate((pc_np), axis=0)
        # pc_vis_color_np = np.concatenate((pc_color_np), axis=0)
        pc_vis_np = pc_np
        pc_vis_color_np = pc_color_np
        if keypoint_np is not None:
            # print(pc_vis_np.shape)
            # print(keypoint_np.shape)
            pc_vis_np = np.concatenate((pc_vis_np, keypoint_np), axis=0)
            pc_vis_color_np = np.concatenate((pc_vis_color_np, keypoint_color_np), axis=0)
        if keypoint_other_np is not None:
            pc_vis_np = np.concatenate((pc_vis_np, keypoint_other_np), axis=0)
            pc_vis_color_np = np.concatenate((pc_vis_color_np, keypoint_other_color_np), axis=0)

        return pc_vis_np, pc_vis_color_np

    # visualization with visdom
    def get_current_visuals(self):
        # build pc & node & keypoint with marker color
        anc_pc_np = self.anc_pc[0].cpu().numpy().transpose()  # Nx3
        anc_keypoint_np = self.anc_keypoints[0].detach().cpu().numpy().transpose()  # Kx3
        anc_sigmas_np = self.anc_sigmas[0].detach().cpu().numpy()  # K

        # rotate to get better visualization if it is 2d rotation augmentation
        # if self.opt.rot_3d == False and self.opt.rot_horizontal == True:
        #     [anc_pc_np, anc_keypoint_np] = augmentation.rotate_point_cloud_list_3d(
        #         [anc_pc_np, anc_keypoint_np],
        #         angles=[math.pi / 4, 0, 0])
        anc_data_vis_np, anc_data_vis_color_np = self.build_pc_node_keypoint_visual(anc_pc_np,
                                                                                    anc_keypoint_np,
                                                                                    sigmas_np=anc_sigmas_np)

        pos_pc_np = self.pos_pc[0].cpu().numpy().transpose()  # Nx3
        pos_keypoint_np = self.pos_keypoints[0].detach().cpu().numpy().transpose()  # Kx3
        pos_sigmas_np = self.pos_sigmas[0].detach().cpu().numpy()  # K

        # rotate to get better visualization if it is 2d rotation augmentation
        # if self.opt.rot_3d == False and self.opt.rot_horizontal == True:
        #     [pos_pc_np, pos_keypoint_np] = augmentation.rotate_point_cloud_list_3d(
        #         [pos_pc_np, pos_keypoint_np],
        #         angles=[math.pi / 4, 0, 0])
        pos_data_vis_np, pos_data_vis_color_np = self.build_pc_node_keypoint_visual(pos_pc_np,
                                                                                    pos_keypoint_np,
                                                                                    sigmas_np=pos_sigmas_np)

        return OrderedDict([('anc_data_vis', (anc_data_vis_np, anc_data_vis_color_np)),
                            ('pos_data_vis', (pos_data_vis_np, pos_data_vis_color_np))])

    def get_current_errors(self):
        return OrderedDict([
            ('O_loss', self.loss.item()),
            ('O_triplet', self.triplet_loss.item()),
            ('O_active_perc', self.active_percentage.item()),
            ('E_loss', self.test_loss_average.item()),
            ('E_triplet', self.test_loss_triplet_average.item()),
            ('E_active_perc', self.test_active_percentage_average.item())
        ])

    def save_network(self, network, network_label, epoch_label, gpu_id):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.opt.checkpoints_dir, save_filename)
        torch.save(network.state_dict(), save_path)
        # if gpu_id >= 0 and torch.cuda.is_available():
        #     # torch.cuda.device(gpu_id)
        #     network.to(self.opt.device)

    def update_learning_rate(self, ratio):
        lr_clip = 0.00001

        # descriptor
        lr_descriptor = self.old_lr_descriptor * ratio
        if lr_descriptor < lr_clip:
            lr_descriptor = lr_clip
        for param_group in self.optimizer_descriptor.param_groups:
            param_group['lr'] = lr_descriptor

        print('update descriptor learning rate: %f -> %f' % (self.old_lr_descriptor, lr_descriptor))
        self.old_lr_descriptor = lr_descriptor
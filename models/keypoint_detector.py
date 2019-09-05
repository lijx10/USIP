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
from models import operations


class ModelDetector():
    def __init__(self, opt):
        self.opt = opt

        if opt.scene == 'indoor':
            self.detector = networks.RPN_DetectorLite(opt)
        else:
            self.detector = networks.RPN_Detector(opt)
            # self.detector = networks.RPN_Detector_KNN(opt)
            # self.detector = networks.RPN_Detector_Ball(opt)
        self.chamfer_criteria = losses.ChamferLoss_Brute(opt)
        # self.chamfer_criteria = losses.ChamferLoss_Brute_NoSigma(opt)
        self.keypoint_on_pc_criteria = losses.KeypointOnPCLoss(opt)

        if self.opt.gpu_ids[0] >= 0:
            self.detector = self.detector.to(self.opt.device)
            self.chamfer_criteria = self.chamfer_criteria.to(self.opt.device)
            self.keypoint_on_pc_criteria = self.keypoint_on_pc_criteria.to(self.opt.device)

        # multi-gpu training
        if len(opt.gpu_ids) > 1:
            self.detector = nn.DataParallel(self.detector, device_ids=opt.gpu_ids)
            self.keypoint_on_pc_criteria = nn.DataParallel(self.keypoint_on_pc_criteria, device_ids=opt.gpu_ids)

        # learning rate_control
        self.old_lr_detector = self.opt.lr

        self.optimizer_detector = torch.optim.Adam(self.detector.parameters(),
                                                   lr=self.old_lr_detector,
                                                   betas=(0.9, 0.999),
                                                   weight_decay=0)

        if self.opt.gpu_ids[0] >= 0:
            self.detector = self.detector.to(self.opt.device)

        # place holder for GPU tensors
        self.src_pc = torch.FloatTensor(self.opt.batch_size, 3, self.opt.input_pc_num).uniform_()
        self.src_sn = torch.FloatTensor(self.opt.batch_size, 3, self.opt.input_pc_num).uniform_()
        self.src_label = torch.LongTensor(self.opt.batch_size).fill_(1)
        self.src_node = torch.FloatTensor(self.opt.batch_size, 3, self.opt.node_num)

        self.dst_pc = torch.FloatTensor(self.opt.batch_size, 3, self.opt.input_pc_num).uniform_()
        self.dst_sn = torch.FloatTensor(self.opt.batch_size, 3, self.opt.input_pc_num).uniform_()
        self.dst_label = torch.LongTensor(self.opt.batch_size).fill_(1)
        self.dst_node = torch.FloatTensor(self.opt.batch_size, 3, self.opt.node_num)

        self.src_R_dst = torch.zeros((self.opt.batch_size, 3, 3), dtype=torch.float32)
        self.src_scale_dst = torch.zeros((self.opt.batch_size, 1), dtype=torch.float32)
        self.src_shift_dst = torch.zeros((self.opt.batch_size, 3, 1), dtype=torch.float32)

        # record the test loss and accuracy
        self.test_chamfer_average = torch.tensor([0], dtype=torch.float32, requires_grad=False)
        self.test_loss_average = torch.tensor([0], dtype=torch.float32, requires_grad=False)
        self.test_keypoint_on_pc_average = torch.tensor([0], dtype=torch.float32, requires_grad=False)
        # loss that do not involve in optimization
        self.chamfer_pure = torch.tensor([0], dtype=torch.float32, requires_grad=False)
        self.test_chamfer_pure_average = torch.tensor([0], dtype=torch.float32, requires_grad=False)
        self.chamfer_weighted = torch.tensor([0], dtype=torch.float32, requires_grad=False)
        self.test_chamfer_weighted_average = torch.tensor([0], dtype=torch.float32, requires_grad=False)

        if self.opt.gpu_ids[0] >= 0:
            self.src_pc = self.src_pc.to(self.opt.device)
            self.src_sn = self.src_sn.to(self.opt.device)
            self.src_node = self.src_node.to(self.opt.device)
            self.src_label = self.src_label.to(self.opt.device)

            self.dst_pc = self.dst_pc.to(self.opt.device)
            self.dst_sn = self.dst_sn.to(self.opt.device)
            self.dst_node = self.dst_node.to(self.opt.device)
            self.dst_label = self.dst_label.to(self.opt.device)

            self.src_R_dst = self.src_R_dst.to(self.opt.device)
            self.src_scale_dst = self.src_scale_dst.to(self.opt.device)
            self.src_shift_dst = self.src_shift_dst.to(self.opt.device)

            self.test_chamfer_average = self.test_chamfer_average.to(self.opt.device)
            self.test_loss_average = self.test_loss_average.to(self.opt.device)
            self.test_keypoint_on_pc_average = self.test_keypoint_on_pc_average.to(self.opt.device)

            self.chamfer_pure = self.chamfer_pure.to(self.opt.device)
            self.test_chamfer_pure_average = self.test_chamfer_pure_average.to(self.opt.device)
            self.chamfer_weighted = self.chamfer_weighted.to(self.opt.device)
            self.test_chamfer_weighted_average = self.test_chamfer_weighted_average.to(self.opt.device)

        # pre-cautions for shared memory usage
        if opt.batch_size * 2 / len(opt.gpu_ids) > operations.CUDA_SHARED_MEM_DIM_X or opt.node_num > operations.CUDA_SHARED_MEM_DIM_Y:
            print('--- WARNING: batch_size or node_num larger than pre-defined cuda shared memory array size. '
                  'Please modify CUDA_SHARED_MEM_DIM_X and CUDA_SHARED_MEM_DIM_Y in models/operations.py')

    def set_input(self, 
                  src_pc, src_sn, src_node,
                  dst_pc, dst_sn, dst_node,
                  src_R_dst, src_scale_dst, src_shift_dst):
        # self.src_pc.resize_(src_pc.size()).copy_(src_pc).detach()
        # self.src_pc = src_pc.to(self.opt.device).detach()
        # self.src_sn.resize_(src_sn.size()).copy_(src_sn).detach()
        # self.src_node.resize_(src_node.size()).copy_(src_node).detach()
        #
        # self.dst_pc.resize_(dst_pc.size()).copy_(dst_pc).detach()
        # self.dst_pc = dst_pc.to(self.opt.device).detach()
        # self.dst_sn.resize_(dst_sn.size()).copy_(dst_sn).detach()
        # self.dst_node.resize_(dst_node.size()).copy_(dst_node).detach()
        #
        # self.src_R_dst.resize_(src_R_dst.size()).copy_(src_R_dst).detach()
        # self.src_scale_dst.resize_(src_scale_dst.size()).copy_(src_scale_dst).detach()
        # self.src_shift_dst.resize_(src_shift_dst.size()).copy_(src_shift_dst).detach()

        self.src_pc = src_pc.float().to(self.opt.device).detach()
        self.src_sn = src_sn.float().to(self.opt.device).detach()
        self.src_node = src_node.float().to(self.opt.device).detach()

        self.dst_pc = dst_pc.float().to(self.opt.device).detach()
        self.dst_sn = dst_sn.float().to(self.opt.device).detach()
        self.dst_node = dst_node.float().to(self.opt.device).detach()

        self.src_R_dst = src_R_dst.float().to(self.opt.device).detach()
        self.src_scale_dst = src_scale_dst.float().to(self.opt.device).detach()
        self.src_shift_dst = src_shift_dst.float().to(self.opt.device).detach()
        
        torch.cuda.synchronize()

    def forward(self, pc, sn, node, is_train=False, epoch=None):
        with torch.cuda.device(pc.get_device()):
            node_recomputed, keypoints, sigmas, descriptors = self.detector(pc, sn, node, is_train, epoch)  # Bx1024
            return node_recomputed, keypoints, sigmas, descriptors

    def forward_siamese(self, pc_tuple, sn_tuple, node_tuple, is_train=False, epoch=None):
        size_of_single_chunk = pc_tuple[0].size()[0]
        node_recomputed, keypoints, sigmas, descriptors = self.detector(torch.cat(pc_tuple, dim=0),
                                                                        torch.cat(sn_tuple, dim=0),
                                                                        torch.cat(node_tuple, dim=0),
                                                                        is_train, epoch)  # Bx1024
        node_recomputed_tuple = torch.split(node_recomputed, split_size_or_sections=size_of_single_chunk, dim=0)
        keypoints_tuple = torch.split(keypoints, split_size_or_sections=size_of_single_chunk, dim=0)
        sigmas_tuple = torch.split(sigmas, split_size_or_sections=size_of_single_chunk, dim=0)

        if descriptors is not None:
            descriptors_tuple = torch.split(descriptors, split_size_or_sections=size_of_single_chunk, dim=0)
        else:
            descriptors_tuple = (None, None)

        return node_recomputed_tuple, keypoints_tuple, sigmas_tuple, descriptors_tuple

    def optimize(self, epoch=None):
        with torch.cuda.device(self.src_pc.get_device()):
            # random point dropout
            if self.opt.random_pc_dropout_lower_limit < 0.99:
                dropout_keep_ratio = random.uniform(self.opt.random_pc_dropout_lower_limit, 1.0)
                resulting_pc_num = round(dropout_keep_ratio*self.opt.input_pc_num)
                chosen_indices = np.random.choice(self.opt.input_pc_num, resulting_pc_num, replace=False)
                chosen_indices_tensor = torch.from_numpy(chosen_indices).to(self.opt.device)
                self.src_pc = torch.index_select(self.src_pc, dim=2, index=chosen_indices_tensor)
                self.src_sn = torch.index_select(self.src_sn, dim=2, index=chosen_indices_tensor)
                self.dst_pc = torch.index_select(self.dst_pc, dim=2, index=chosen_indices_tensor)
                self.dst_sn = torch.index_select(self.dst_sn, dim=2, index=chosen_indices_tensor)

            self.detector.train()

            (self.src_node_recomputed, self.dst_node_recomputed), \
            (self.src_keypoints, self.dst_keypoints), \
            (self.src_sigmas, self.dst_sigmas), \
            (self.src_descriptors, self.dst_descriptors) = self.forward_siamese((self.src_pc, self.dst_pc),
                                                                                (self.src_sn, self.dst_sn),
                                                                                (self.src_node, self.dst_node),
                                                                                is_train=True, epoch=epoch)

            # transform src to self.src_keypoints_transformed
            self.src_keypoints_transformed = torch.matmul(self.src_R_dst, self.src_keypoints)  # Bx3x3 * Bx3xM -> Bx3xM
            self.src_keypoints_transformed = self.src_keypoints_transformed * self.src_scale_dst.unsqueeze(1).unsqueeze(2)  # Bx3xM * Bx1x1 -> Bx3xM
            self.src_keypoints_transformed = self.src_keypoints_transformed + self.src_shift_dst  # Bx3xM + Bx3x1 -> Bx3xM

            self.detector.zero_grad()

            # chamfer loss to align two keypoints
            self.loss_chamfer, self.chamfer_pure, self.chamfer_weighted = self.chamfer_criteria(self.src_keypoints_transformed, self.dst_keypoints,
                                                                                                self.src_sigmas, self.dst_sigmas)

            # keypoint on pc loss
            if self.opt.keypoint_on_pc_type == 'point_to_point':
                self.loss_keypoint_on_pc_src = torch.mean(self.keypoint_on_pc_criteria(self.src_keypoints, self.src_pc,
                                                                                       None)) * self.opt.keypoint_on_pc_alpha
                self.loss_keypoint_on_pc_dst = torch.mean(self.keypoint_on_pc_criteria(self.dst_keypoints, self.dst_pc,
                                                                                       None)) * self.opt.keypoint_on_pc_alpha
            elif self.opt.keypoint_on_pc_type == 'point_to_plane':
                self.loss_keypoint_on_pc_src = torch.mean(self.keypoint_on_pc_criteria(self.src_keypoints, self.src_pc,
                                                                                       self.src_sn)) * self.opt.keypoint_on_pc_alpha
                self.loss_keypoint_on_pc_dst = torch.mean(self.keypoint_on_pc_criteria(self.dst_keypoints, self.dst_pc,
                                                                                       self.dst_sn)) * self.opt.keypoint_on_pc_alpha

            self.loss = self.loss_chamfer + self.loss_keypoint_on_pc_src + self.loss_keypoint_on_pc_dst
            self.loss.backward()

            self.optimizer_detector.step()

    def test_model(self):
        self.detector.eval()

        (self.src_node_recomputed, self.dst_node_recomputed), \
        (self.src_keypoints, self.dst_keypoints), \
        (self.src_sigmas, self.dst_sigmas), \
        (self.src_descriptors, self.dst_descriptors) = self.forward_siamese((self.src_pc, self.dst_pc),
                                                                            (self.src_sn, self.dst_sn),
                                                                            (self.src_node, self.dst_node),
                                                                            is_train=False)

        # transform src to self.src_keypoints_transformed
        self.src_keypoints_transformed = torch.matmul(self.src_R_dst, self.src_keypoints)  # Bx3x3 * Bx3xM -> Bx3xM
        self.src_keypoints_transformed = self.src_keypoints_transformed * self.src_scale_dst.unsqueeze(1).unsqueeze(2)  # Bx3xM * Bx1x1 -> Bx3xM
        self.src_keypoints_transformed = self.src_keypoints_transformed + self.src_shift_dst  # Bx3xM + Bx3x1 -> Bx3xM

        # chamfer loss to align two keypoints
        self.loss_chamfer, self.chamfer_pure, self.chamfer_weighted = self.chamfer_criteria(self.src_keypoints_transformed, self.dst_keypoints,
                                                                                            self.src_sigmas, self.dst_sigmas)

        # keypoint on pc loss
        if self.opt.keypoint_on_pc_type == 'point_to_point':
            self.loss_keypoint_on_pc_src = torch.mean(self.keypoint_on_pc_criteria(self.src_keypoints, self.src_pc,
                                                                                   None)) * self.opt.keypoint_on_pc_alpha
            self.loss_keypoint_on_pc_dst = torch.mean(self.keypoint_on_pc_criteria(self.dst_keypoints, self.dst_pc,
                                                                                   None)) * self.opt.keypoint_on_pc_alpha
        elif self.opt.keypoint_on_pc_type == 'point_to_plane':
            self.loss_keypoint_on_pc_src = torch.mean(self.keypoint_on_pc_criteria(self.src_keypoints, self.src_pc,
                                                                                   self.src_sn)) * self.opt.keypoint_on_pc_alpha
            self.loss_keypoint_on_pc_dst = torch.mean(self.keypoint_on_pc_criteria(self.dst_keypoints, self.dst_pc,
                                                                                   self.dst_sn)) * self.opt.keypoint_on_pc_alpha

        self.loss = self.loss_chamfer + self.loss_keypoint_on_pc_src + self.loss_keypoint_on_pc_dst

    def freeze_model(self):
        for p in self.detector.parameters():
            p.requires_grad = False

    def run_model(self, pc, sn, node):
        self.detector.eval()
        with torch.no_grad():
            _, keypoints, sigmas, _ = self.forward(pc, sn, node, is_train=False, epoch=None)
        return keypoints, sigmas

    def run_model_siamese(self, pc_tuple, sn_tuple, node_tuple):
        self.detector.eval()
        with torch.no_grad():
            _, keypoints_tuple, sigmas_tuple, _ = self.forward_siamese(pc_tuple, sn_tuple, node_tuple, is_train=False, epoch=None)
        return keypoints_tuple, sigmas_tuple

    @staticmethod
    def build_pc_node_keypoint_visual(pc_np, node_np, keypoint_np=None, keypoint_other_np=None, sigmas_np=None,
                                      sigmas_other_np=None):
        pc_color_np = np.repeat(np.expand_dims(np.array([191, 191, 191], dtype=np.int64), axis=0),
                                pc_np.shape[0],
                                axis=0)  # 1x3 -> Nx3
        node_color_np = np.repeat(np.expand_dims(np.array([51, 204, 51], dtype=np.int64), axis=0),
                                  node_np.shape[0],
                                  axis=0)  # 1x3 -> Mx3
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

        pc_vis_np = np.concatenate((pc_np, node_np), axis=0)
        pc_vis_color_np = np.concatenate((pc_color_np, node_color_np), axis=0)
        if keypoint_np is not None:
            pc_vis_np = np.concatenate((pc_vis_np, keypoint_np), axis=0)
            pc_vis_color_np = np.concatenate((pc_vis_color_np, keypoint_color_np), axis=0)
        if keypoint_other_np is not None:
            pc_vis_np = np.concatenate((pc_vis_np, keypoint_other_np), axis=0)
            pc_vis_color_np = np.concatenate((pc_vis_color_np, keypoint_other_color_np), axis=0)

        return pc_vis_np, pc_vis_color_np

    # visualization with visdom
    def get_current_visuals(self):
        # build pc & node & keypoint with marker color
        src_pc_np = self.src_pc[0].cpu().numpy().transpose()  # Nx3
        src_node_np = self.src_node_recomputed[0].cpu().numpy().transpose()  # Mx3
        src_keypoint_np = self.src_keypoints[0].detach().cpu().numpy().transpose()  # Kx3
        src_sigmas_np = self.src_sigmas[0].detach().cpu().numpy()  # K

        # rotate to get better visualization if it is 2d rotation augmentation
        if self.opt.rot_3d == False and self.opt.rot_horizontal == True:
            [src_pc_np, src_node_np, src_keypoint_np] = augmentation.rotate_point_cloud_list_3d([src_pc_np, src_node_np, src_keypoint_np],
                                                                                                angles=[math.pi/4, 0, 0])
        src_data_vis_np, src_data_vis_color_np = self.build_pc_node_keypoint_visual(src_pc_np, src_node_np,
                                                                                    src_keypoint_np,
                                                                                    sigmas_np=src_sigmas_np)

        dst_pc_np = self.dst_pc[0].cpu().numpy().transpose()  # Nx3
        dst_node_np = self.dst_node_recomputed[0].cpu().numpy().transpose()  # Mx3
        dst_keypoint_np = self.dst_keypoints[0].detach().cpu().numpy().transpose()  # Kx3
        src_keypoint_transformed_np = self.src_keypoints_transformed[0].detach().cpu().numpy().transpose()  # Kx3
        dst_sigmas_np = self.dst_sigmas[0].detach().cpu().numpy()  # K

        # rotate to get better visualization if it is 2d rotation augmentation
        if self.opt.rot_3d == False and self.opt.rot_horizontal == True:
            [dst_pc_np, dst_node_np, dst_keypoint_np, src_keypoint_transformed_np] = augmentation.rotate_point_cloud_list_3d([dst_pc_np, dst_node_np, dst_keypoint_np, src_keypoint_transformed_np],
                                                                                                                             angles=[math.pi/4, 0, 0])
        dst_data_vis_np, dst_data_vis_color_np = self.build_pc_node_keypoint_visual(dst_pc_np,
                                                                                    dst_node_np,
                                                                                    dst_keypoint_np,
                                                                                    src_keypoint_transformed_np,
                                                                                    sigmas_np=dst_sigmas_np,
                                                                                    sigmas_other_np=src_sigmas_np
                                                                                    )

        return OrderedDict([('src_data_vis', (src_data_vis_np, src_data_vis_color_np)),
                            ('dst_data_vis', (dst_data_vis_np, dst_data_vis_color_np))])

    def get_current_errors(self):
        return OrderedDict([
            ('O_loss', self.loss.item()),
            ('O_chamfer', self.loss_chamfer.item()),
            ('O_key_on_pc', self.loss_keypoint_on_pc_src.item()+self.loss_keypoint_on_pc_dst.item()),
            ('E_loss', self.test_loss_average.item()),
            ('E_chamfer', self.test_chamfer_average.item()),
            ('E_key_on_pc', self.test_keypoint_on_pc_average.item()),
            ('E_cham_pure', self.test_chamfer_pure_average.item()),
            ('E_cham_weig', self.test_chamfer_weighted_average.item())
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

        # detector
        lr_detector = self.old_lr_detector * ratio
        if lr_detector < lr_clip:
            lr_detector = lr_clip
        for param_group in self.optimizer_detector.param_groups:
            param_group['lr'] = lr_detector
        print('update detector learning rate: %f -> %f' % (self.old_lr_detector, lr_detector))
        self.old_lr_detector = lr_detector
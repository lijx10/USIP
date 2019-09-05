import time
import copy
import numpy as np
import math

from scenenn import options_descriptor
print('====== descriptor ======')
opt_descriptor = options_descriptor.Options().parse()

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import os

from models.keypoint_descriptor import ModelDescriptorIndoor
from models.keypoint_detector import ModelDetector
from data.scenenn_descriptor_loader import SceneNNDescriptorLoader
from util.visualizer import Visualizer


def model_state_dict_parallel_convert(state_dict, mode):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    if mode == 'to_single':
        for k, v in state_dict.items():
            name = k[7:]  # remove 'module.' of DataParallel
            new_state_dict[name] = v
    elif mode == 'to_parallel':
        for k, v in state_dict.items():
            name = 'module.' + k  # add 'module.' of DataParallel
            new_state_dict[name] = v
    elif mode == 'same':
        new_state_dict = state_dict
    else:
        raise Exception('mode = to_single / to_parallel')

    return new_state_dict


def model_state_dict_convert_auto(state_dict, gpu_ids):
    for k, v in state_dict.items():
        if (k[0:7] == 'module.' and len(gpu_ids) >= 2) or (k[0:7] != 'module.' and len(gpu_ids) == 1):
            return state_dict
        elif k[0:7] == 'module.' and len(gpu_ids) == 1:
            return model_state_dict_parallel_convert(state_dict, mode='to_single')
        elif k[0:7] != 'module.' and len(gpu_ids) >= 2:
            return model_state_dict_parallel_convert(state_dict, mode='to_parallel')
        else:
            raise Exception('Error in model_state_dict_convert_auto')


if __name__=='__main__':

    is_use_random_keypoint = False

    trainset = SceneNNDescriptorLoader(opt_descriptor.dataroot, 'train', opt_descriptor) + \
               SceneNNDescriptorLoader(opt_descriptor.dataroot, 'val', opt_descriptor)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt_descriptor.batch_size, shuffle=True,
                                              num_workers=opt_descriptor.nThreads, drop_last=True, pin_memory=False)
    dataset_size = len(trainset)
    print('#training point clouds = %d' % len(trainset))



    testset = SceneNNDescriptorLoader(opt_descriptor.dataroot, 'test', opt_descriptor)
    testloader = torch.utils.data.DataLoader(testset, batch_size=opt_descriptor.batch_size, shuffle=False,
                                             num_workers=opt_descriptor.nThreads, pin_memory=False)
    print('#testing point clouds = %d' % len(testset))

    if False == is_use_random_keypoint:
        # create detector model, load state_dict
        model_detector = ModelDetector(opt_descriptor)
        folder_name = '%d-%d-k%dk%d-2dp-nojitter' % (opt_descriptor.input_pc_num, opt_descriptor.node_num, opt_descriptor.k, opt_descriptor.node_knn_k_1)
        model_detector.detector.load_state_dict(
            model_state_dict_convert_auto(
                torch.load(
                    '/ssd/keypoint-pc/scenenn/checkpoints/save/detector/' + folder_name + '/best.pth',
                    map_location='cpu'), opt_descriptor.gpu_ids))
        model_detector.freeze_model()
        print(folder_name)
    else:
        print('Using random keypoint!!!')

    # create descriptor model
    model_descriptor = ModelDescriptorIndoor(opt_descriptor)
    # model_descriptor.descriptor.load_state_dict(torch.load(''))

    visualizer = Visualizer(opt_descriptor)

    best_loss = 1e6
    lr_decay_step = 15
    for epoch in range(501):

        epoch_iter = 0
        for i, data in enumerate(trainloader):
            iter_start_time = time.time()
            epoch_iter += opt_descriptor.batch_size

            anc_pc, anc_sn, anc_node, \
            pos_pc, pos_sn, pos_node, \
            R, scale, shift = data

            # get the keypoints and sigmas
            anc_pc_cuda = anc_pc.to(opt_descriptor.device)
            anc_sn_cuda = anc_sn.to(opt_descriptor.device)
            anc_node_cuda = anc_node.to(opt_descriptor.device)
            pos_pc_cuda = pos_pc.to(opt_descriptor.device)
            pos_sn_cuda = pos_sn.to(opt_descriptor.device)
            pos_node_cuda = pos_node.to(opt_descriptor.device)

            if False == is_use_random_keypoint:
                (anc_keypoints, pos_keypoints), (anc_sigmas, pos_sigmas) = model_detector.run_model_siamese(
                    (anc_pc_cuda, pos_pc_cuda),
                    (anc_sn_cuda, pos_sn_cuda),
                    (anc_node_cuda, pos_node_cuda))

                # random sample to reduce computation cost
                # M_small = 384
                # sample_idx = torch.randperm(anc_keypoints.size()[2], dtype=torch.int64, device=anc_keypoints.device)[0:M_small]  # M_small
                # anc_keypoints = anc_keypoints[:, :, sample_idx]
                # anc_sigmas = anc_sigmas[:, sample_idx]
                # pos_keypoints = pos_keypoints[:, :, sample_idx]
                # pos_sigmas = pos_sigmas[:, sample_idx]
            else:
                random_idx = torch.randperm(n=anc_pc.size()[2], dtype=torch.int64, device=opt_descriptor.device)[0:anc_node.size(2)]  # M
                anc_keypoints = torch.gather(anc_pc_cuda, dim=2, index=random_idx.unsqueeze(0).unsqueeze(0).expand(anc_pc.size()[0], 3, anc_node.size(2)))  # Bx3xM
                anc_sigmas = torch.zeros((anc_pc.size()[0], anc_node.size(2)), dtype=torch.float32, device=anc_pc_cuda.device) + 0.01

                pos_keypoints = torch.gather(pos_pc_cuda, dim=2, index=random_idx.unsqueeze(0).unsqueeze(0).expand(pos_pc.size()[0], 3, pos_node.size(2)))  # Bx3xM
                pos_sigmas = torch.zeros((pos_pc.size()[0], pos_node.size(2)), dtype=torch.float32, device=pos_pc_cuda.device) + 0.01

            # train descriptor
            model_descriptor.set_input(anc_pc_cuda, anc_sn_cuda, anc_keypoints, anc_sigmas,
                                       pos_pc_cuda, pos_sn_cuda, pos_keypoints, pos_sigmas,
                                       R, scale, shift)

            model_descriptor.optimize(epoch=epoch)

            if i % int(32 / opt_descriptor.batch_size * 100) == 0 and i > 0:
            # if i % 10 and i > 0:
                # print/plot errors
                t = (time.time() - iter_start_time) / opt_descriptor.batch_size

                errors = model_descriptor.get_current_errors()

                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, opt_descriptor, errors)

            if i % int(32 / opt_descriptor.batch_size * 100) == 0 and i > 0:
                visuals = model_descriptor.get_current_visuals()
                visualizer.display_current_results(visuals, epoch, i)

        # test network
        if epoch >= 0 and epoch%1==0:
            batch_amount = 0
            model_descriptor.test_loss_average.zero_()
            model_descriptor.test_loss_triplet_average.zero_()
            model_descriptor.test_active_percentage_average.zero_()
            # model_descriptor.test_loss_same_average.zero_()
            model_descriptor.test_loss_chamfer_average.zero_()
            for i, data in enumerate(testloader):
                anc_pc, anc_sn, anc_node, \
                pos_pc, pos_sn, pos_node, \
                R, scale, shift = data

                # get the keypoints and sigmas
                anc_pc_cuda = anc_pc.to(opt_descriptor.device)
                anc_sn_cuda = anc_sn.to(opt_descriptor.device)
                anc_node_cuda = anc_node.to(opt_descriptor.device)
                pos_pc_cuda = pos_pc.to(opt_descriptor.device)
                pos_sn_cuda = pos_sn.to(opt_descriptor.device)
                pos_node_cuda = pos_node.to(opt_descriptor.device)

                if False == is_use_random_keypoint:
                    (anc_keypoints, pos_keypoints), (anc_sigmas, pos_sigmas) = model_detector.run_model_siamese(
                        (anc_pc_cuda, pos_pc_cuda),
                        (anc_sn_cuda, pos_sn_cuda),
                        (anc_node_cuda, pos_node_cuda))

                    # random sample to reduce computation cost
                    # M_small = 384
                    # sample_idx = torch.randperm(anc_keypoints.size()[2], dtype=torch.int64, device=anc_keypoints.device)[0:M_small]  # M_small
                    # anc_keypoints = anc_keypoints[:, :, sample_idx]
                    # anc_sigmas = anc_sigmas[:, sample_idx]
                    # pos_keypoints = pos_keypoints[:, :, sample_idx]
                    # pos_sigmas = pos_sigmas[:, sample_idx]
                else:
                    random_idx = torch.randperm(n=anc_pc.size()[2], dtype=torch.int64, device=opt_descriptor.device)[0:anc_node.size(2)]  # M
                    anc_keypoints = torch.gather(anc_pc_cuda, dim=2, index=random_idx.unsqueeze(0).unsqueeze(0).expand(anc_pc.size()[0], 3, anc_node.size(2)))  # Bx3xM
                    anc_sigmas = torch.zeros((anc_pc.size()[0], anc_node.size(2)), dtype=torch.float32, device=anc_pc_cuda.device) + 0.01

                    pos_keypoints = torch.gather(pos_pc_cuda, dim=2, index=random_idx.unsqueeze(0).unsqueeze(0).expand(pos_pc.size()[0], 3, pos_node.size(2)))  # Bx3xM
                    pos_sigmas = torch.zeros((pos_pc.size()[0], pos_node.size(2)), dtype=torch.float32, device=pos_pc_cuda.device) + 0.01


                    # train descriptor
                model_descriptor.set_input(anc_pc_cuda, anc_sn_cuda, anc_keypoints, anc_sigmas,
                                           pos_pc_cuda, pos_sn_cuda, pos_keypoints, pos_sigmas,
                                           R, scale, shift)

                model_descriptor.test_model()

                batch_amount += anc_pc.size()[0]

                # accumulate loss
                model_descriptor.test_loss_average += model_descriptor.loss.detach() * anc_pc.size()[0]
                model_descriptor.test_loss_triplet_average += model_descriptor.triplet_loss.detach() * anc_pc.size()[0]
                model_descriptor.test_active_percentage_average += model_descriptor.active_percentage.detach() * anc_pc.size()[0]
                # model_descriptor.test_loss_same_average += model_descriptor.same_scan_loss.detach() * anc_pc.size()[0]
                model_descriptor.test_loss_chamfer_average += model_descriptor.chamfer_loss.detach() * anc_pc.size()[0]


            # update best loss
            model_descriptor.test_loss_average /= batch_amount
            model_descriptor.test_loss_triplet_average /= batch_amount
            model_descriptor.test_active_percentage_average /= batch_amount
            # model_descriptor.test_loss_same_average /= batch_amount
            model_descriptor.test_loss_chamfer_average /= batch_amount

            if model_descriptor.test_loss_average.item() <= best_loss:
                best_loss = model_descriptor.test_loss_average.item()
            print('Tested network. So far best loss: %f' % best_loss)

            # save models
            if ((model_descriptor.test_loss_average.item() <= best_loss + 1e-5) and (epoch>0.5*lr_decay_step)) or (epoch > 1 and epoch % 10 == 0):
                print("Saving network...")
                model_descriptor.save_network(model_descriptor.descriptor, 'descriptor', 'gpu%d_%d_%f' % (opt_descriptor.gpu_ids[0], epoch, model_descriptor.test_loss_average.item()), opt_descriptor.gpu_ids[0])
                if opt_descriptor.reconstruction_alpha > 0.001:
                    model_descriptor.save_network(model_descriptor.reconstructor, 'reconstructor', 'gpu%d_%d_%f' % (opt_descriptor.gpu_ids[0], epoch, model_descriptor.test_loss_average.item()), opt_descriptor.gpu_ids[0])

        if epoch%lr_decay_step==0 and epoch > 0:
            model_descriptor.update_learning_rate(0.5)
        # batch normalization momentum decay:
        next_epoch = epoch + 1
        if (opt_descriptor.bn_momentum_decay_step is not None) and (next_epoch >= 1) and (
                next_epoch % opt_descriptor.bn_momentum_decay_step == 0):
            current_bn_momentum = opt_descriptor.bn_momentum * (
                    opt_descriptor.bn_momentum_decay ** (next_epoch // opt_descriptor.bn_momentum_decay_step))
            print('BN momentum updated to: %f' % current_bn_momentum)







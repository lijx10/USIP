import time
import copy
import numpy as np
import math

from oxford import options_detector
from oxford import options_descriptor
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

from models.keypoint_descriptor import ModelDescriptor
from models.keypoint_detector import ModelDetector
from data.oxford_descriptor_loader import OxfordDescriptorLoader
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

    trainset = OxfordDescriptorLoader(opt_descriptor.dataroot, 'train', opt_descriptor)
    dataset_size = len(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt_descriptor.batch_size, shuffle=True,
                                              num_workers=opt_descriptor.nThreads, drop_last=True, pin_memory=False)
    print('#training point clouds = %d' % len(trainset))

    testset = OxfordDescriptorLoader(opt_descriptor.dataroot, 'test', opt_descriptor)
    testloader = torch.utils.data.DataLoader(testset, batch_size=opt_descriptor.batch_size, shuffle=True,
                                             num_workers=opt_descriptor.nThreads, pin_memory=False)

    # create detector model, load state_dict
    model_detector = ModelDetector(opt_descriptor)
    folder_name = '%d-%d-k%dk%d-2d' % (opt_descriptor.input_pc_num, opt_descriptor.node_num, opt_descriptor.k, opt_descriptor.node_knn_k_1)
    model_detector.detector.load_state_dict(
        model_state_dict_convert_auto(
            torch.load(
                '/ssd/jiaxin/tsf/oxford/checkpoints/save/detector/' + folder_name + '/best.pth',
                map_location='cpu'), opt_descriptor.gpu_ids))
    model_detector.freeze_model()

    # create descriptor model
    model_descriptor = ModelDescriptor(opt_descriptor)
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
            index_batch = data

            # get the keypoints and sigmas
            anc_pc_cuda = anc_pc.to(opt_descriptor.device)
            anc_sn_cuda = anc_sn.to(opt_descriptor.device)
            anc_node_cuda = anc_node.to(opt_descriptor.device)
            pos_pc_cuda = pos_pc.to(opt_descriptor.device)
            pos_sn_cuda = pos_sn.to(opt_descriptor.device)
            pos_node_cuda = pos_node.to(opt_descriptor.device)
            # if set(opt_descriptor.gpu_ids) == set([2, 3]):
            if True:
                (anc_keypoints, pos_keypoints), (anc_sigmas, pos_sigmas) = model_detector.run_model_siamese(
                    (anc_pc_cuda, pos_pc_cuda),
                    (anc_sn_cuda, pos_sn_cuda),
                    (anc_node_cuda, pos_node_cuda))
            else:
                # smaller batch size to save memory
                anc_keypoints, anc_sigmas = model_detector.run_model(anc_pc_cuda, anc_sn_cuda, anc_node_cuda)
                pos_keypoints, pos_sigmas = model_detector.run_model(pos_pc_cuda, pos_sn_cuda, pos_node_cuda)

            # debug, fake keypoint and sigmas
            # random_idx = torch.randint(low=0, high=anc_pc.size()[2], size=(anc_pc.size()[0], opt_descriptor.node_num), dtype=torch.int64, device=opt_descriptor.device)  # BxM
            # anc_keypoints = torch.gather(anc_pc_cuda, dim=2, index=random_idx.unsqueeze(1).expand(anc_pc.size()[0], 3, opt_descriptor.node_num))  # Bx3xM
            # anc_sigmas = torch.rand((anc_pc.size()[0], opt_descriptor.node_num), dtype=torch.float32, device=opt_descriptor.device)
            # pos_keypoints = torch.gather(pos_pc_cuda, dim=2, index=random_idx.unsqueeze(1).expand(anc_pc.size()[0], 3, opt_descriptor.node_num))  # Bx3xM
            # pos_sigmas = torch.rand((anc_pc.size()[0], opt_descriptor.node_num), dtype=torch.float32, device=opt_descriptor.device)

            # height scaling, under CAM coordinate, y this down axis
            scale = np.random.uniform(low=0.25, high=1.2)
            if opt_descriptor.is_height_scaling:
                anc_pc_cuda[:, 1, :] *= scale
                anc_keypoints[:, 1, :] *= scale

                pos_pc_cuda[:, 1, :] *= scale
                pos_keypoints[:, 1, :] *= scale

            # get neg_idx
            neg_idx = trainset.mine_negative_sample(index_batch)

            # train descriptor
            model_descriptor.set_input(anc_pc_cuda, anc_sn_cuda, anc_keypoints, anc_sigmas,
                                       pos_pc_cuda, pos_sn_cuda, pos_keypoints, pos_sigmas,
                                       neg_idx)

            model_descriptor.optimize(epoch=epoch)

            if i % int(32 / opt_descriptor.batch_size * 100) == 0 and i > 0:
                # print/plot errors
                t = (time.time() - iter_start_time) / opt_descriptor.batch_size

                errors = model_descriptor.get_current_errors()

                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, opt_descriptor, errors)

                visuals = model_descriptor.get_current_visuals()
                visualizer.display_current_results(visuals, epoch, i)

        # test network
        if epoch >= 0 and epoch%1==0:
            batch_amount = 0
            model_descriptor.test_loss_average.zero_()
            model_descriptor.test_loss_triplet_average.zero_()
            model_descriptor.test_active_percentage_average.zero_()
            for i, data in enumerate(testloader):
                anc_pc, anc_sn, anc_node, \
                pos_pc, pos_sn, pos_node, \
                index_batch = data

                # get the keypoints and sigmas
                anc_pc_cuda = anc_pc.to(opt_descriptor.device)
                anc_sn_cuda = anc_sn.to(opt_descriptor.device)
                anc_node_cuda = anc_node.to(opt_descriptor.device)
                pos_pc_cuda = pos_pc.to(opt_descriptor.device)
                pos_sn_cuda = pos_sn.to(opt_descriptor.device)
                pos_node_cuda = pos_node.to(opt_descriptor.device)

                # if set(opt_descriptor.gpu_ids) == set([2, 3]):
                if True:
                    (anc_keypoints, pos_keypoints), (anc_sigmas, pos_sigmas) = model_detector.run_model_siamese(
                        (anc_pc_cuda, pos_pc_cuda),
                        (anc_sn_cuda, pos_sn_cuda),
                        (anc_node_cuda, pos_node_cuda))
                else:
                    # smaller batch size to save memory
                    anc_keypoints, anc_sigmas = model_detector.run_model(anc_pc_cuda, anc_sn_cuda, anc_node_cuda)
                    pos_keypoints, pos_sigmas = model_detector.run_model(pos_pc_cuda, pos_sn_cuda, pos_node_cuda)

                # debug, fake keypoint and sigmas
                # random_idx = torch.randint(low=0, high=anc_pc.size()[2], size=(anc_pc.size()[0], opt_descriptor.node_num), dtype=torch.int64, device=opt_descriptor.device)  # BxM
                # anc_keypoints = torch.gather(anc_pc_cuda, dim=2, index=random_idx.unsqueeze(1).expand(anc_pc.size()[0], 3, opt_descriptor.node_num))  # Bx3xM
                # anc_sigmas = torch.rand((anc_pc.size()[0], opt_descriptor.node_num), dtype=torch.float32, device=opt_descriptor.device)
                # pos_keypoints = torch.gather(pos_pc_cuda, dim=2, index=random_idx.unsqueeze(1).expand(anc_pc.size()[0], 3, opt_descriptor.node_num))  # Bx3xM
                # pos_sigmas = torch.rand((anc_pc.size()[0], opt_descriptor.node_num), dtype=torch.float32, device=opt_descriptor.device)

                # get neg_idx
                neg_idx = testset.mine_negative_sample(index_batch)

                # train descriptor
                model_descriptor.set_input(anc_pc_cuda, anc_sn_cuda, anc_keypoints, anc_sigmas,
                                           pos_pc_cuda, pos_sn_cuda, pos_keypoints, pos_sigmas,
                                           neg_idx)

                model_descriptor.test_model()

                batch_amount += anc_pc.size()[0]

                # accumulate loss
                model_descriptor.test_loss_average += model_descriptor.loss.detach() * anc_pc.size()[0]
                model_descriptor.test_loss_triplet_average += model_descriptor.triplet_loss.detach() * anc_pc.size()[0]
                model_descriptor.test_active_percentage_average += model_descriptor.active_percentage.detach() * anc_pc.size()[0]


            # update best loss
            model_descriptor.test_loss_average /= batch_amount
            model_descriptor.test_loss_triplet_average /= batch_amount
            model_descriptor.test_active_percentage_average /= batch_amount

            if model_descriptor.test_loss_average.item() <= best_loss:
                best_loss = model_descriptor.test_loss_average.item()
            print('Tested network. So far best loss: %f' % best_loss)

            # save models
            if ((model_descriptor.test_loss_average.item() <= best_loss + 1e-5) and (epoch>0.5*lr_decay_step)) or (epoch > 1 and epoch % 10 == 0):
                print("Saving network...")
                model_descriptor.save_network(model_descriptor.descriptor, 'descriptor', 'gpu%d_%d_%f' % (opt_descriptor.gpu_ids[0], epoch, model_descriptor.test_loss_average.item()), opt_descriptor.gpu_ids[0])

        if epoch%lr_decay_step==0 and epoch > 0:
            model_descriptor.update_learning_rate(0.5)
        # batch normalization momentum decay:
        next_epoch = epoch + 1
        if (opt_descriptor.bn_momentum_decay_step is not None) and (next_epoch >= 1) and (
                next_epoch % opt_descriptor.bn_momentum_decay_step == 0):
            current_bn_momentum = opt_descriptor.bn_momentum * (
                    opt_descriptor.bn_momentum_decay ** (next_epoch // opt_descriptor.bn_momentum_decay_step))
            print('BN momentum updated to: %f' % current_bn_momentum)







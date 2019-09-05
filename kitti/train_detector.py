import time
import copy
import numpy as np
import math

from kitti.options_detector import Options
opt = Options().parse()  # set CUDA_VISIBLE_DEVICES before import torch

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np

from models.keypoint_detector import ModelDetector
from data.kitti_detector_loader import KittiLoader
from util.visualizer import Visualizer
import models.operations


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
    trainset = KittiLoader(opt.dataroot, 'train', opt)
    dataset_size = len(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True,
                                              num_workers=opt.nThreads, drop_last=True, pin_memory=True)
    print('#training point clouds = %d' % len(trainset))

    testset = KittiLoader(opt.dataroot, 'test', opt)
    testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False,
                                             num_workers=opt.nThreads, pin_memory=True)
    print('#testing point clouds = %d' % len(testset))

    # create model, optionally load pre-trained model
    model = ModelDetector(opt)

    visualizer = Visualizer(opt)

    best_loss = 1e6
    lr_decay_step = 10
    for epoch in range(501):

        epoch_iter = 0
        for i, data in enumerate(trainloader):
            iter_start_time = time.time()
            epoch_iter += opt.batch_size

            src_pc, src_sn, src_node, \
            dst_pc, dst_sn, dst_node, \
            R, scale, shift = data
            model.set_input(src_pc, src_sn, src_node,
                            dst_pc, dst_sn, dst_node,
                            R, scale, shift)

            model.optimize(epoch=epoch)

            if i % int(32 / opt.batch_size * 100) == 0 and i > 0:
                # print/plot errors
                t = (time.time() - iter_start_time) / opt.batch_size

                errors = model.get_current_errors()

                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, opt, errors)

                # print(model.autoencoder.encoder.feature)
                visuals = model.get_current_visuals()
                visualizer.display_current_results(visuals, epoch, i)

        # test network
        # ========== extra info ==============
        # manually print some params
        sigma_mean = model.src_sigmas.mean()
        sigma_std = model.src_sigmas.std()
        sigma_max = torch.max(model.src_sigmas)
        sigma_min = torch.min(model.src_sigmas)
        print(' --- sigma mean: %f, std: %f, max: %f, min: %f' % (sigma_mean, sigma_std, sigma_max, sigma_min))
        if epoch >= 0 and epoch%1==0:
            batch_amount = 0
            model.test_loss_average.zero_()
            model.test_chamfer_average.zero_()
            model.test_keypoint_on_pc_average.zero_()
            model.test_chamfer_pure_average.zero_()
            model.test_chamfer_weighted_average.zero_()
            for i, data in enumerate(testloader):
                src_pc, src_sn, src_node, \
                dst_pc, dst_sn, dst_node, \
                R, scale, shift = data
                model.set_input(src_pc, src_sn, src_node,
                                dst_pc, dst_sn, dst_node,
                                R, scale, shift)
                model.test_model()

                batch_amount += src_pc.size()[0]

                # accumulate loss
                model.test_loss_average += model.loss.detach() * src_pc.size()[0]
                model.test_chamfer_average += model.loss_chamfer.detach() * src_pc.size()[0]
                model.test_keypoint_on_pc_average += (model.loss_keypoint_on_pc_src.detach() + model.loss_keypoint_on_pc_dst.detach()) * src_pc.size()[0]
                model.test_chamfer_pure_average += model.chamfer_pure.detach() * src_pc.size()[0]
                model.test_chamfer_weighted_average += model.chamfer_weighted.detach() * src_pc.size()[0]

            # update best loss
            model.test_loss_average /= batch_amount
            model.test_chamfer_average /= batch_amount
            model.test_keypoint_on_pc_average /= batch_amount
            model.test_chamfer_pure_average /= batch_amount
            model.test_chamfer_weighted_average /= batch_amount

            if model.test_loss_average.item() <= best_loss:
                best_loss = model.test_loss_average.item()
            print('Tested network. So far best loss: %f' % best_loss)

            # save models
            if (model.test_loss_average.item() <= best_loss + 1e-5) and (model.test_chamfer_pure_average.item() < 1.1) and (epoch>2*lr_decay_step):
                print("Saving network...")
                model.save_network(model.detector, 'detector', 'gpu%d_%d_%f' % (opt.gpu_ids[0], epoch, model.test_loss_average.item()), opt.gpu_ids[0])

        if epoch%lr_decay_step==0 and epoch > 0:
            model.update_learning_rate(0.5)
        # batch normalization momentum decay:
        next_epoch = epoch + 1
        if (opt.bn_momentum_decay_step is not None) and (next_epoch >= 1) and (
                next_epoch % opt.bn_momentum_decay_step == 0):
            current_bn_momentum = opt.bn_momentum * (
            opt.bn_momentum_decay ** (next_epoch // opt.bn_momentum_decay_step))
            print('BN momentum updated to: %f' % current_bn_momentum)

        # save network
        # if epoch%20==0 and epoch>0:
        #     print("Saving network...")
        #     model.save_network(model.classifier, 'cls', '%d' % epoch, opt.gpu_id)






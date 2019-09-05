import torch
import torch.nn as nn


class ChamferLoss_Brute_NoSigma(nn.Module):
    def __init__(self, opt):
        super(ChamferLoss_Brute_NoSigma, self).__init__()
        self.opt = opt
        self.dimension = 3

    def forward(self, pc_src_input, pc_dst_input, sigma_src=None, sigma_dst=None):
        '''
        :param pc_src_input: Bx3xM Tensor in GPU
        :param pc_dst_input: Bx3xN Tensor in GPU
        :param sigma_src: BxM Tensor in GPU
        :param sigma_dst: BxN Tensor in GPU
        :return:
        '''

        B, M = pc_src_input.size()[0], pc_src_input.size()[2]
        N = pc_dst_input.size()[2]

        pc_src_input_expanded = pc_src_input.unsqueeze(3).expand(B, 3, M, N)
        pc_dst_input_expanded = pc_dst_input.unsqueeze(2).expand(B, 3, M, N)

        # the gradient of norm is set to 0 at zero-input. There is no need to use custom norm anymore.
        diff = torch.norm(pc_src_input_expanded - pc_dst_input_expanded, dim=1, keepdim=False)  # BxMxN

        # pc_src vs selected pc_dst, M
        src_dst_min_dist, _ = torch.min(diff, dim=2, keepdim=False)  # BxM
        forward_loss = src_dst_min_dist.mean()

        # pc_dst vs selected pc_src, N
        dst_src_min_dist, _ = torch.min(diff, dim=1, keepdim=False)  # BxN
        backward_loss = dst_src_min_dist.mean()

        chamfer_pure = forward_loss + backward_loss
        chamfer_weighted = chamfer_pure

        return forward_loss + backward_loss, chamfer_pure, chamfer_weighted


# ============== detector loss ============== begin ======
class ChamferLoss_Brute(nn.Module):
    def __init__(self, opt):
        super(ChamferLoss_Brute, self).__init__()
        self.opt = opt
        self.dimension = 3

    def forward(self, pc_src_input, pc_dst_input, sigma_src=None, sigma_dst=None):
        '''
        :param pc_src_input: Bx3xM Tensor in GPU
        :param pc_dst_input: Bx3xN Tensor in GPU
        :param sigma_src: BxM Tensor in GPU
        :param sigma_dst: BxN Tensor in GPU
        :return:
        '''

        B, M = pc_src_input.size()[0], pc_src_input.size()[2]
        N = pc_dst_input.size()[2]

        pc_src_input_expanded = pc_src_input.unsqueeze(3).expand(B, 3, M, N)
        pc_dst_input_expanded = pc_dst_input.unsqueeze(2).expand(B, 3, M, N)

        # the gradient of norm is set to 0 at zero-input. There is no need to use custom norm anymore.
        diff = torch.norm(pc_src_input_expanded - pc_dst_input_expanded, dim=1, keepdim=False)  # BxMxN

        if sigma_src is None or sigma_dst is None:
            # pc_src vs selected pc_dst, M
            src_dst_min_dist, _ = torch.min(diff, dim=2, keepdim=False)  # BxM
            forward_loss = src_dst_min_dist

            # pc_dst vs selected pc_src, N
            dst_src_min_dist, _ = torch.min(diff, dim=1, keepdim=False)  # BxN
            backward_loss = dst_src_min_dist

            chamfer_pure = forward_loss + backward_loss
            chamfer_weighted = chamfer_pure
        else:
            # pc_src vs selected pc_dst, M
            src_dst_min_dist, src_dst_I = torch.min(diff, dim=2, keepdim=False)  # BxM, BxM
            selected_sigma_dst = torch.gather(sigma_dst, dim=1, index=src_dst_I)  # BxN -> BxM
            sigma_src_dst = (sigma_src + selected_sigma_dst) / 2
            forward_loss = (torch.log(sigma_src_dst) + src_dst_min_dist / sigma_src_dst).mean()

            # pc_dst vs selected pc_src, N
            dst_src_min_dist, dst_src_I = torch.min(diff, dim=1, keepdim=False)  # BxN, BxN
            selected_sigma_src = torch.gather(sigma_src, dim=1, index=dst_src_I)  # BxM -> BxN
            sigma_dst_src = (sigma_dst + selected_sigma_src) / 2
            backward_loss = (torch.log(sigma_dst_src) + dst_src_min_dist / sigma_dst_src).mean()

            # loss that do not involve in optimization
            chamfer_pure = (src_dst_min_dist.mean() + dst_src_min_dist.mean()).detach()
            weight_src_dst = (1.0/sigma_src_dst) / torch.mean(1.0/sigma_src_dst)
            weight_dst_src = (1.0/sigma_dst_src) / torch.mean(1.0/sigma_dst_src)
            chamfer_weighted = ((weight_src_dst * src_dst_min_dist).mean() +
                                (weight_dst_src * dst_src_min_dist).mean()).detach()

        return forward_loss + backward_loss, chamfer_pure, chamfer_weighted


class KeypointOnPCLoss(nn.Module):
    def __init__(self, opt):
        super(KeypointOnPCLoss, self).__init__()
        self.opt = opt

        self.single_side_chamfer = SingleSideChamferLoss_Brute(opt)
        self.keypoint_on_surface = PointOnSurfaceLoss(opt)

    def forward(self, keypoint, pc, sn=None):
        if sn is None:
            loss = self.single_side_chamfer(keypoint, pc)
        else:
            loss = self.keypoint_on_surface(keypoint, pc, sn)

        return loss


class SingleSideChamferLoss_Brute(nn.Module):
    def __init__(self, opt):
        super(SingleSideChamferLoss_Brute, self).__init__()
        self.opt = opt
        self.dimension = 3

    def forward(self, pc_src_input, pc_dst_input):
        '''
        :param pc_src_input: Bx3xM Variable in GPU
        :param pc_dst_input: Bx3xN Variable in GPU
        :return:
        '''

        B, M = pc_src_input.size()[0], pc_src_input.size()[2]
        N = pc_dst_input.size()[2]

        pc_src_input_expanded = pc_src_input.unsqueeze(3).expand(B, 3, M, N)
        pc_dst_input_expanded = pc_dst_input.unsqueeze(2).expand(B, 3, M, N)

        diff = torch.norm(pc_src_input_expanded - pc_dst_input_expanded, dim=1, keepdim=False)  # BxMxN

        # pc_src vs selected pc_dst, M
        src_dst_min_dist, _ = torch.min(diff, dim=2, keepdim=False)  # BxM

        return src_dst_min_dist


class PointOnSurfaceLoss(nn.Module):
    def __init__(self, opt):
        super(PointOnSurfaceLoss, self).__init__()
        self.opt = opt

    def forward(self, keypoint, pc, sn):
        '''

        :param keypoint: Bx3xM
        :param pc: Bx3xN
        :param sn: Bx3xN
        :return:
        '''

        B, M = keypoint.size()[0], keypoint.size()[2]
        N = pc.size()[2]

        keypoint_expanded = keypoint.unsqueeze(3).expand(B, 3, M, N)
        pc_expanded = pc.unsqueeze(2).expand(B, 3, M, N)

        diff = torch.norm(keypoint_expanded - pc_expanded, dim=1, keepdim=False)  # BxMxN

        # keypoint vs selected pc, M
        keypoint_pc_min_dist, keypoint_pc_min_I = torch.min(diff, dim=2, keepdim=False)  # BxM
        pc_selected = torch.gather(pc, dim=2, index=keypoint_pc_min_I.unsqueeze(1).expand(B, 3, M))  # Bx3xM
        sn_selected = torch.gather(sn, dim=2, index=keypoint_pc_min_I.unsqueeze(1).expand(B, 3, M))  # Bx3xM

        # keypoint on surface loss
        keypoint_minus_pc = keypoint - pc_selected  # Bx3xM
        keypoint_minus_pc_norm = torch.norm(keypoint_minus_pc, dim=1, keepdim=True)  # Bx1xM
        keypoint_minus_pc_normalized = keypoint_minus_pc / (keypoint_minus_pc_norm + 1e-7)  # Bx3xM

        sn_selected = sn_selected.permute(0, 2, 1)  # BxMx3
        keypoint_minus_pc_normalized = keypoint_minus_pc_normalized.permute(0, 2, 1)  # BxMx3

        loss = torch.matmul(sn_selected.unsqueeze(2), keypoint_minus_pc_normalized.unsqueeze(3)) ** 2  # BxMx1x3 * BxMx3x1 -> BxMx1x1 -> 1

        return loss

# ============== detector loss ============== end ======


# ============== descriptor loss ============== begin ======

class DescPairScanLoss(nn.Module):
    '''
    Positive: For each keypoint A in src scan, find keypoint B in positive scan, with the most similar descriptor, minimize the distance
    Negative: For each keypoint B in src scan, find keypoint B in negative scan, with the most similar descriptor, maximize the distance
    the 'minimize / maximize' is via triplet loss
    '''
    def __init__(self, opt):
        super(DescPairScanLoss, self).__init__()
        self.opt = opt

    def forward(self, anc_descriptors, pos_descriptors, neg_descriptors, anc_sigmas):
        '''

        :param anc_descriptors: BxCxM
        :param pos_descriptors: BxCxM
        :param neg_descriptors: BxCxM
        :param anc_sigmas: BxM
        :return:
        '''

        anc_descriptors_Mx1 = anc_descriptors.unsqueeze(3)  # BxCxMx1
        pos_descriptors_1xM = pos_descriptors.unsqueeze(2)  # BxCx1xM
        neg_descriptors_1xM = neg_descriptors.unsqueeze(2)  # BxCx1xM

        # positive loss
        anc_pos_diff = torch.norm(anc_descriptors_Mx1 - pos_descriptors_1xM, dim=1, keepdim=False)  # BxCxMxM -> BxMxM
        min_anc_pos_diff, _ = torch.min(anc_pos_diff, dim=2, keepdim=False)  # BxM

        # negative loss
        anc_neg_diff = torch.norm(anc_descriptors_Mx1 - neg_descriptors_1xM, dim=1, keepdim=False)  # BxCxMxM -> BxMxM
        min_anc_neg_diff, _ = torch.min(anc_neg_diff, dim=2, keepdim=False)  # BxM

        # triplet loss
        before_clamp_loss = min_anc_pos_diff - min_anc_neg_diff + self.opt.triple_loss_gamma  # BxM
        active_percentage = torch.mean((before_clamp_loss > 0).float(), dim=1, keepdim=False)  # B

        # 1. without sigmas
        # loss = torch.clamp(before_clamp_loss, min=0)

        # 2. with sigmas, use only the anc_sigmas
        # sigma is the uncertainly, smaller->more important. Turn it into weight by alpha - sigma
        anc_weights = torch.clamp(self.opt.sigma_max - anc_sigmas, min=0)  # BxM
        # normalize to be mean of 1
        anc_weights_mean = torch.mean(anc_weights, dim=1, keepdim=True)  # Bx1
        anc_weights = (anc_weights / anc_weights_mean).detach()  # BxM
        loss = anc_weights * torch.clamp(before_clamp_loss, min=0)  # BxM

        return loss, active_percentage


class DescCGFLoss(nn.Module):
    def __init__(self, opt):
        super(DescCGFLoss, self).__init__()
        self.opt = opt

    def forward(self, anc_keypoints, anc_descriptors, pos_keypoints, pos_descriptors, anc_sigmas,
                anc_pc=None, pos_pc=None):
        '''
        follow CGF triplet loss
        positive pair: for each keypoint in anc, find keypoints in pos with d<threshold, random sample one
        negative pair: for each keypoint in anc, find keypoitns in pos with the closest but d>threshold.
                       This is different with original CGF. If my detector works perfectly, it may be difficult to find
                       keypoints with threshold<d<2*threshold
        :param anc_keypoints: Bx3xM, already transformed to the coordinate of pos
        :param anc_descriptors: BxCxM
        :param pos_keypoints: Bx3xM
        :param pos_descriptors: BxCxM
        :param anc_sigmas: BxM
        :return:
        '''

        B, M = anc_keypoints.size()[0], anc_keypoints.size()[2]

        anc_descriptors_Mx1 = anc_descriptors.unsqueeze(3)  # BxCxMx1
        pos_descriptors_1xM = pos_descriptors.unsqueeze(2)  # BxCx1xM
        descriptor_diff = torch.norm(anc_descriptors_Mx1 - pos_descriptors_1xM, dim=1, keepdim=False)  # BxCxMxM -> BxMxM

        anc_keypoints_Mx1 = anc_keypoints.unsqueeze(3)  # Bx3xMx1
        pos_keypoints_1xM = pos_keypoints.unsqueeze(2)  # Bx3x1xM
        keypoint_diff = torch.norm(anc_keypoints_Mx1 - pos_keypoints_1xM, dim=1, keepdim=False)  # Bx3xMxM -> BxMxM

        # 1. positive pair
        positive_mask_BMM = keypoint_diff <= self.opt.CGF_radius  # BxMxM
        positive_mask_BM, _ = torch.max(positive_mask_BMM, dim=2, keepdim=False)  # BxM, to indicate whether there is a match

        # 1.1 random sample a match
        random_mat = torch.rand((B, M, M), dtype=torch.float32, device=anc_keypoints.device, requires_grad=False)  # [0, 1)
        random_mat_nearby_mask = positive_mask_BMM.float() * random_mat  # BxMxM
        _, nearby_idx = torch.max(random_mat_nearby_mask, dim=2, keepdim=True)  # BxMx1
        positive_dist = torch.gather(descriptor_diff, dim=2, index=nearby_idx).squeeze(2)  # BxMx1 -> BxM

        # 2. negative pair
        # 2.1 for each keypoint in anc, find keypoitns in pos with the closest but d>threshold.
        augmented_keypoint_diff = keypoint_diff + positive_mask_BMM.float() * 1000  # BxMxM
        _, far_close_idx = torch.min(augmented_keypoint_diff, dim=2, keepdim=True)  # BxMx1
        far_close_dist = torch.gather(descriptor_diff, dim=2, index=far_close_idx).squeeze(2)  # BxMx1 -> BxM

        # 2.2 randomly choose a keypoint with d>threshold
        outside_radius_mask = keypoint_diff > self.opt.CGF_radius  # BxMxM
        random_mat_outside = torch.rand((B, M, M), dtype=torch.float32, device=anc_keypoints.device, requires_grad=False)  # [0, 1)
        random_mat_outside_masked = random_mat_outside * outside_radius_mask.float()  # BxMxM
        _, outside_idx = torch.max(random_mat_outside_masked, dim=2, keepdim=True)  # BxMx1
        outside_random_dist = torch.gather(descriptor_diff, dim=2, index=outside_idx).squeeze(2)  # BxM

        # 2.3 assemble a negative_dist by combining far_close_dist & outside_random_dist
        random_mat_selection = torch.rand((B, M), dtype=torch.float32, device=anc_keypoints.device, requires_grad=False)  # BxM
        selection_mat = (random_mat_selection < 0.5).float()  # BxM
        negative_dist = selection_mat * far_close_dist + (1 - selection_mat) * outside_random_dist  # BxM

        # triplet loss
        # consider only the matched keypoints, so a re-scale is necessary
        scaling = (M / (torch.sum(positive_mask_BM.float(), dim=1, keepdim=False) + 1)).detach()  # B
        # print(scaling)
        # print(anc_keypoints.size())
        # print(anc_descriptors.size())
        before_clamp_loss = (positive_dist - negative_dist + self.opt.triple_loss_gamma) * positive_mask_BM.float()  # BxM
        active_percentage = torch.sum((before_clamp_loss > 1e-5).float(), dim=1, keepdim=False) / (torch.sum(positive_mask_BM.float(), dim=1, keepdim=False) + 1)  # B

        # with sigmas, use only the anc_sigmas
        # sigma is the uncertainly, smaller->more important. Turn it into weight by alpha - sigma
        anc_weights = torch.clamp(self.opt.sigma_max - anc_sigmas, min=0)  # BxM
        # normalize to be mean of 1
        anc_weights_mean = torch.mean(anc_weights, dim=1, keepdim=True)  # Bx1
        anc_weights = (anc_weights / anc_weights_mean).detach()  # BxM
        loss = anc_weights * torch.clamp(before_clamp_loss, min=0) * scaling.unsqueeze(1)  # BxM

        # debug
        # anc_pc_batch_np = torch.transpose(anc_pc, 1, 2).detach().cpu().numpy()
        # pos_pc_batch_np = torch.transpose(pos_pc, 1, 2).detach().cpu().numpy()
        # anc_keypoints_batch_np = torch.transpose(anc_keypoints, 1, 2).detach().cpu().numpy()
        # pos_keypoints_batch_np = torch.transpose(pos_keypoints, 1, 2).detach().cpu().numpy()
        # for b in range(B):
        #     print('---\nscaling %f' % scaling[b].item())
        #     anc_pc_np = anc_pc_batch_np[b]  # Nx3
        #     pos_pc_np = pos_pc_batch_np[b]  # Nx3
        #     anc_keypoints_np = anc_keypoints_batch_np[b]  # Mx3
        #     pos_keypoints_np = pos_keypoints_batch_np[b]  # Mx3
        #
        #     fig = plt.figure()
        #     ax = Axes3D(fig)
        #     ax.scatter(pos_pc_np[:, 0].tolist(),
        #                pos_pc_np[:, 1].tolist(),
        #                pos_pc_np[:, 2].tolist(),
        #                s=5, c=[1, 0.8, 0.8])
        #     ax.scatter(anc_pc_np[:, 0].tolist(),
        #                anc_pc_np[:, 1].tolist(),
        #                anc_pc_np[:, 2].tolist(),
        #                s=5, c=[0.8, 0.8, 1])
        #
        #     # plot the first matched keypoint
        #     for m in range(M):
        #         if positive_mask_BM[b, m] == 1:
        #             ax.scatter(anc_keypoints_np[m, 0],
        #                        anc_keypoints_np[m, 1],
        #                        anc_keypoints_np[m, 2],
        #                        s=30, c=[1, 0, 0])
        #             ax.scatter(pos_keypoints_np[nearby_idx[b, m].item(), 0],
        #                        pos_keypoints_np[nearby_idx[b, m].item(), 1],
        #                        pos_keypoints_np[nearby_idx[b, m].item(), 2],
        #                        s=30, c=[0, 0, 1])
        #             ax.scatter(pos_keypoints_np[far_close_idx[b, m].item(), 0],
        #                        pos_keypoints_np[far_close_idx[b, m].item(), 1],
        #                        pos_keypoints_np[far_close_idx[b, m].item(), 2],
        #                        s=30, c=[0, 1, 0])
        #             ax.scatter(pos_keypoints_np[outside_idx[b, m].item(), 0],
        #                        pos_keypoints_np[outside_idx[b, m].item(), 1],
        #                        pos_keypoints_np[outside_idx[b, m].item(), 2],
        #                        s=30, c=[0.4, 0, 0.8])
        #             print('matched dist: %f' % np.linalg.norm(anc_keypoints_np[m] - pos_keypoints_np[nearby_idx[b, m].item()]))
        #             print('far-close dist: %f' % np.linalg.norm(anc_keypoints_np[m] - pos_keypoints_np[far_close_idx[b, m].item()]))
        #             print('outside random dist: %f' % np.linalg.norm(anc_keypoints_np[m] - pos_keypoints_np[outside_idx[b, m].item()]))
        #             break
        #
        #     vis_tools.axisEqual3D(ax)
        #     ax.set_xlabel('x')
        #     ax.set_ylabel('y')
        #     ax.set_zlabel('z')
        #     plt.show()

        return loss, active_percentage


# ============== descriptor loss ============== end ======

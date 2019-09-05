import time
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.multiprocessing as mp
import threading
import ctypes

# import numba
# import numba.cuda

# generalized batch size
CUDA_SHARED_MEM_DIM_X = 24
# size of SOM
CUDA_SHARED_MEM_DIM_Y = 512


# Numba compiler is not threadsafe
compiler_lock = threading.Lock()


def zero_edge(x, padding):
    '''

    :param x: BxCxHxW Variable/Tensor
    :param padding: int
    :return:
    '''
    if (padding is None) or (padding <= 0):
        return x

    H = x.size()[2]
    W = x.size()[3]

    H_padding_idx = list(range(0, padding))
    H_padding_idx_tail = list(range(H-padding, H))
    H_padding_idx.extend(H_padding_idx_tail)

    W_padding_idx = list(range(0, padding))
    W_padding_idx_tail = list(range(W-padding, W))
    W_padding_idx.extend(W_padding_idx_tail)

    x[:, :, H_padding_idx, :] = 0
    x[:, :, :, W_padding_idx] = 0

    return x


# ================================== mask max ======================================
# class MaskedMaxThread:
#     def __init__(self, thread_num):
#         self.batch_each_worker = 2
#         self.thread_num = thread_num
#
#     def worker(self, i):
#         batch_size = self.data.size()[0]
#         node_num = self.mask.size()[3]
#         for i in range(i * self.batch_each_worker, (i + 1) * self.batch_each_worker):
#             if i>=batch_size:
#                 break
#             # iterate over the clusters
#             for j in range(node_num):
#                 indexes = torch.nonzero(self.mask[i, 0, :, j])
#                 if len(indexes.size()) > 0:
#                     selected_rows = self.data[i].index_select(dim=1, index=indexes[:, 0])  # Cxk
#                     _, idx = selected_rows.max(dim=1)
#                     self.gather_idx[i, :, j] = indexes[:, 0][idx]
#
#     def compute(self, data, mask):
#         '''
#         :param data: BxCxN tensor in CPU
#         :param mask: Bx1xNxnode_num tensor in CPU
#         :return gather_idx: BxCxnode_num tensor in CPU
#         '''
#         batch_size = data.size()[0]
#         self.batch_each_worker = math.ceil(batch_size / self.thread_num)
#
#         self.data = data.cpu()
#         self.mask = mask.cpu()
#         self.gather_idx = torch.LongTensor(batch_size, data.size()[1], mask.size()[3]).zero_()
#
#         threads = []
#         for i in range(self.thread_num):
#             t = threading.Thread(target=self.worker, args=(i, ))
#             t.start()
#             threads.append(t)
#         for t in threads:
#             t.join()
#
#         return self.gather_idx


# def get_devicendarray_float32(t):
#     assert t.type() == 'torch.cuda.FloatTensor'
#     ctx = numba.cuda.cudadrv.driver.driver.get_context()
#     mp = numba.cuda.cudadrv.driver.MemoryPointer(ctx, ctypes.c_ulong(t.data_ptr()), t.numel()*4)
#     return numba.cuda.cudadrv.devicearray.DeviceNDArray(t.size(), [i*4 for i in t.stride()], np.dtype('float32'),
#                                                   gpu_data=mp, stream=torch.cuda.current_stream().cuda_stream)
#
#
# def get_devicendarray_int32(t):
#     assert t.type() == 'torch.cuda.IntTensor'
#     ctx = numba.cuda.cudadrv.driver.driver.get_context()
#     mp = numba.cuda.cudadrv.driver.MemoryPointer(ctx, ctypes.c_ulong(t.data_ptr()), t.numel()*4)
#     return numba.cuda.cudadrv.devicearray.DeviceNDArray(t.size(), [i*4 for i in t.stride()], np.dtype('int32'),
#                                                   gpu_data=mp, stream=torch.cuda.current_stream().cuda_stream)


# @numba.cuda.jit('float32[:,:,:], int32[:,:], int32[:,:,:]')
# def indexed_max_shared_mem(data, index, max_idx):
#     '''
#     :param data: BxCxN
#     :param index: BxN
#     :param max_idx: BxCxK
#     :param max_val: BxCxK
#     :return:
#     '''
#
#     max_value_shared = numba.cuda.shared.array(shape=(CUDA_SHARED_MEM_DIM_X, CUDA_SHARED_MEM_DIM_Y), dtype=numba.float32)
#     # max_value_private = numba.cuda.shared.array(shape=(64), dtype=numba.float32)
#
#     c = numba.cuda.blockIdx.x
#     b = numba.cuda.threadIdx.x
#
#     # initialize shared memory
#     for k in range(CUDA_SHARED_MEM_DIM_Y):
#         max_value_shared[b, k] = -10000
#     # numba.cuda.syncthreads()
#
#     for n in range(index.shape[1]):
#         k = index[b, n]
#         data_point = data[b, c, n]
#         if data_point > max_value_shared[b, k]:
#             max_value_shared[b, k] = data_point
#             max_idx[b, c, k] = n
#         # numba.cuda.syncthreads()
#
#
# @numba.cuda.jit('float32[:,:,:], int32[:,:], int32[:,:,:], float32[:,:,:]')
# def indexed_max(data, index, max_idx, max_val):
#     '''
#     :param data: BxCxN
#     :param index: BxN
#     :param max_idx: BxCxK
#     :param max_val: BxCxK
#     :return:
#     '''
#
#     c = numba.cuda.blockIdx.x
#     b = numba.cuda.threadIdx.x
#     for n in range(index.shape[1]):
#         k = index[b, n]
#         data_point = data[b, c, n]
#         if data_point > max_val[b, c, k]:
#             max_val[b, c, k] = data_point
#             max_idx[b, c, k] = n
#         # cuda.syncthreads()
#
#
#
# class MaskedMax:
#     def __init__(self, som_node_number):
#         self.cpu_masked_max = MaskedMaxThread(thread_num=8)
#         self.M = som_node_number
#
#     def compute(self, data, min_idx, mask):
#         '''
#         som node number M.
#         :param data: BxCxkN, FloatTensor
#         :param min_idx: BxkN, LongTensor containing indices of [0,M-1]
#         :param mask: Bx1xNxM, ByteTensor
#         :return: gather_index, BxCxM LongTensor
#         '''
#
#         # ensure that data, min_idx and self.device, self.max_idx, self.max_val locate in the same device
#         # numba.cuda.select_device(data.device.index)
#         # print(numba.cuda.current_context())
#         with numba.cuda.gpus[data.device.index]:
#
#             B = data.size()[0]
#             C = data.size()[1]
#
#             device = data.device
#             max_idx = torch.zeros((B, C, self.M), dtype=torch.int32, device=device)
#
#             # ============= cuda ===================
#             data_cuda = get_devicendarray_float32(data.data)
#             index = min_idx.int()
#             index_cuda = get_devicendarray_int32(index.data)
#             max_idx_cuda = get_devicendarray_int32(max_idx.data)
#
#             if B <= CUDA_SHARED_MEM_DIM_X and self.M <= CUDA_SHARED_MEM_DIM_Y:
#                 # utilizing shared memory
#                 # assert the shared memory size is less that what we allocated
#                 indexed_max_shared_mem[C, B](data_cuda, index_cuda, max_idx_cuda)
#                 numba.cuda.synchronize()
#             else:
#                 # utilizing global memory
#                 max_val = torch.zeros((B, C, self.M), dtype=torch.float32, device=device).fill_(-10000)
#                 max_val_cuda = get_devicendarray_float32(max_val.data)
#                 indexed_max[C, B](data_cuda, index_cuda, max_idx_cuda, max_val_cuda)
#                 numba.cuda.synchronize()
#
#             gather_index = max_idx.long()
#             # ============= cuda ===================
#
#             # ============= cpu ===================
#             # gather_index_cpu = self.cpu_masked_max.compute(data.cpu(), mask.cpu()).to(self.device)
#             # ============= cpu ===================
#
#             # debug
#             # print(self.max_idx.device)
#             # print(gather_index.device)
#             # print(torch.min(gather_index - gather_index_cpu))
#             # print(torch.max(gather_index - gather_index_cpu))
#
#         return gather_index
# ================================== mask max ======================================


# ================================== get the k nearest neighbors of the SOM nodes / features ======================================
# @numba.cuda.jit('float32[:, :, :], int32[:, :, :], float32[:, :, :, :]')
# def knn_gather(som_node, som_node_knn_I, som_node_neighbors):
#     '''
#
#     :param som_node: Bx3xN
#     :param som_node_knn_I: BxNxK
#     :param som_node_neighbors: Bx3xNxK
#     :return:
#     '''
#
#     n = numba.cuda.blockIdx.x  # n \in [0, N-1]
#     k = numba.cuda.blockIdx.y  # k
#     b = numba.cuda.threadIdx.x
#     c = numba.cuda.threadIdx.y
#
#     som_node_neighbors[b, c, n, k] = som_node[b, c, som_node_knn_I[b, n, k]]


def knn_gather_wrapper(som_node, som_node_knn_I):
    '''

    :param som_node: Bx3xN
    :param som_node_knn_I: BxNxK
    :param som_node_neighbors: Bx3xNxK
    :return:
    '''
    B = som_node.size()[0]
    C = som_node.size()[1]
    N = som_node.size()[2]
    K = som_node_knn_I.size()[2]
    assert C==3

    # with numba.cuda.gpus[som_node.device.index]:
    #     som_node_neighbors = torch.zeros((B, C, N, K), dtype=torch.float32, device=som_node.device)
    #
    #     som_node_cuda = get_devicendarray_float32(som_node.data)
    #     som_node_knn_I_cuda = get_devicendarray_int32(som_node_knn_I.int().data)
    #     som_node_neighbors_cuda = get_devicendarray_float32(som_node_neighbors.data)
    #
    #     knn_gather[(N, K), (B, C)](som_node_cuda, som_node_knn_I_cuda, som_node_neighbors_cuda)

    som_node_neighbors = knn_gather_by_indexing(som_node, som_node_knn_I)

    return som_node_neighbors


def knn_gather_by_indexing(som_node, som_node_knn_I):
    '''

    :param som_node: BxCxN
    :param som_node_knn_I: BxNxK
    :param som_node_neighbors: BxCxNxK
    :return:
    '''
    B = som_node.size()[0]
    C = som_node.size()[1]
    N = som_node.size()[2]
    K = som_node_knn_I.size()[2]

    som_node_knn_I = som_node_knn_I.unsqueeze(1).expand(B, C, N, K).contiguous().view(B, C, N*K)
    som_node_neighbors = torch.gather(som_node, dim=2, index=som_node_knn_I).view(B, C, N, K)

    return som_node_neighbors

# ================================== get the k nearest neighbors of the SOM nodes / features ======================================



# ================================== ball query for point clouds ======================================
# @numba.cuda.jit('float32[:, :, :], int32[:, :, :], float32, int32')
# def ball_query(node_to_point_dist, output_points_idx, radius, nsamples):
#     '''
#
#     :param node_to_point_dist: BxMxN
#     :param output_points_idx: BxMxN_SAMPLES
#     :param radius: float32
#     :param nsamples: int32
#     :return:
#     '''
#
#     # shared memory per-block, i.e., per node
#     # output_points_shared = numba.cuda.shared.array(shape=(BATCH_SIZE, 3, N_SAMPLES), dtype=numba.float32)
#     output_points_unique_number_shared = numba.cuda.shared.array(shape=32, dtype=numba.int32)
#
#     m = numba.cuda.blockIdx.x
#     b = numba.cuda.threadIdx.x
#
#     # initialize shared memory
#     output_points_unique_number_shared[b] = 0
#
#     for n in range(node_to_point_dist.shape[2]):
#         unique_idx = output_points_unique_number_shared[b]
#         if unique_idx < nsamples:
#             if node_to_point_dist[b, m, n] <= radius:
#                 output_points_idx[b, m, unique_idx] = n
#                 output_points_unique_number_shared[b] += 1
#         else:
#             break
#
#     # fill the un-defined points
#     unique_idx = output_points_unique_number_shared[b]
#     if unique_idx < nsamples:
#         for i in range(nsamples - unique_idx):
#             idx_repeat = output_points_idx[b, m, i % unique_idx]
#             output_points_idx[b, m, unique_idx + i] = idx_repeat
#
#
#
# def ball_query_wrapper(pc, node, radius, nsamples):
#     '''
#
#     :param pc: Bx3xN
#     :param node: Bx3xM
#     :param radius:
#     :param nsamples:
#     :return: node_neighbor_ball, Bx3xMxnsamples
#     '''
#     with numba.cuda.gpus[pc.device.index]:
#
#         B, N = pc.size()[0], pc.size()[2]
#         M = node.size()[2]
#
#         # 0, random permutation along dim-N, done outside, because the surface normal should be permuted as well
#
#         # 1, get the distance matrix: node_to_point_dist, BxMxN
#         node_expanded = node.unsqueeze(3)  # Bx3xMx1
#         pc_expanded = pc.unsqueeze(2)  # Bx3x1xN
#         node_to_point_dist = torch.norm(node_expanded-pc_expanded, p=2, dim=1, keepdim=False).detach()  # BxMxN
#
#         # 2, build output tensor
#         node_neighbor_ball_idx = torch.zeros((B, M, nsamples), dtype=torch.int32, device=pc.device, requires_grad=False)
#
#         # 3, convert tensor to cuda array
#         # pc_cuda = get_devicendarray_float32(pc.data)
#         node_to_point_dist_cuda = get_devicendarray_float32(node_to_point_dist.data)
#         node_neighbor_ball_idx_cuda = get_devicendarray_int32(node_neighbor_ball_idx.data)
#
#         # 4, cuda code
#         ball_query[M, B](node_to_point_dist_cuda, node_neighbor_ball_idx_cuda, radius, nsamples)
#         numba.cuda.synchronize()
#
#     return node_neighbor_ball_idx.long()
# ================================== ball query for point clouds ======================================


# ============ test ===========
def get_angles(a, b):
    '''
    calculate the angle between vector a and b
    :param a: Bx3xMxK tensor
    :param b: Bx3xMxK tensor
    :return: Bx1xMxK tensor
    '''
    axb = torch.cross(a, b, dim=1)  # Bx3xMxK
    a_1x3 = a.permute(0, 2, 3, 1).contiguous().unsqueeze(3)  # BxMxKx3 -> BxMxKx1x3
    b_3x1 = b.permute(0, 2, 3, 1).contiguous().unsqueeze(4)  # BxMxKx3 -> BxMxKx3x1
    ab = torch.matmul(a_1x3, b_3x1).squeeze(3).squeeze(3)  # BxMxKx1x1

    angle = torch.atan2(torch.norm(axb, dim=1, keepdim=False), ab).unsqueeze(1)
    return angle


if __name__=='__main__':
    # from kitti.options_detector import Options
    # opt = Options().parse()  # set CUDA_VISIBLE_DEVICES before import torch
    print('Done.')


import time
import numpy as np
import math
import torch
from models import operations

import index_max

if __name__=='__main__':
    B = 8
    C = 128
    N = 163840
    M = 512

    data = torch.rand((B, C, N), dtype=torch.float32)
    index = torch.randint(0, M, (B, N), dtype=torch.int32)
    max_idx = torch.zeros((B, C, M), dtype=torch.int32)

    begin_t = time.time()
    max_idx_single_cpu = index_max.forward_cpu(data.detach(), index.detach(), M).long()
    end_t = time.time()
    print('cpu single thread time: %f' % (end_t - begin_t))

    begin_t = time.time()
    max_idx_multi_cpu = index_max.forward_multi_thread_cpu(data, index, M, 8).long()
    end_t = time.time()
    print('cpu multi thread time: %f' % (end_t - begin_t))



    data_cuda = data.cuda()
    index_cuda = index.cuda()


    begin_t = time.time()
    for i in range(100):
        max_idx_cuda = index_max.forward_cuda(data_cuda, index_cuda, M).long()
    end_t = time.time()
    print('cuda cpp time, 100 times: %f' % (end_t - begin_t))

    begin_t = time.time()
    for i in range(100):
        max_idx_cuda_shared_mem = index_max.forward_cuda_shared_mem(data_cuda, index_cuda, M).long()
    end_t = time.time()
    print('cuda cpp shared mem time, 100 times: %f' % (end_t - begin_t))

    mask_max = operations.MaskedMax(M)
    begin_t = time.time()
    for i in range(100):
        max_idx_gt = mask_max.compute(data_cuda, index_cuda, None)
    end_t = time.time()
    print('cuda operations.py time, 100 times: %f' % (end_t - begin_t))

    print(torch.max(max_idx_gt.cpu() - max_idx_single_cpu))
    print(torch.min(max_idx_gt.cpu() - max_idx_single_cpu))

    print(torch.max(max_idx_gt.cpu() - max_idx_multi_cpu))
    print(torch.min(max_idx_gt.cpu() - max_idx_multi_cpu))

    print(torch.max(max_idx_gt.cpu() - max_idx_cuda.cpu()))
    print(torch.min(max_idx_gt.cpu() - max_idx_cuda.cpu()))

    print(torch.max(max_idx_gt.cpu() - max_idx_cuda_shared_mem.cpu()))
    print(torch.min(max_idx_gt.cpu() - max_idx_cuda_shared_mem.cpu()))



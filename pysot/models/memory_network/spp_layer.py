
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

import time

class SPPLayer(nn.Module):

    def __init__(self, num_levels, pool_type='max_pool'):
        super(SPPLayer, self).__init__()

        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):
        bs, c, h, w = x.size()
        pooling_layers = []
        for i in range(self.num_levels):
            kernel_size = h // (2 ** i)
            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(x, kernel_size=kernel_size,
                                      stride=kernel_size).view(bs, -1)
            else:
                tensor = F.avg_pool2d(x, kernel_size=kernel_size,
                                      stride=kernel_size).view(bs, -1)
            pooling_layers.append(tensor)
        x = torch.cat(pooling_layers, dim=-1)
        return x

class CustomSPPLayer(nn.Module):

    def __init__(self, kernel_size, num_levels, stride=1, pool_type='max_pool'):
        super(CustomSPPLayer, self).__init__()
        self.stride = stride

        self.num_levels = num_levels
        self.pool_type = pool_type
        self.spp_layer = SPPLayer(self.num_levels, self.pool_type)

        self.kernel_size = kernel_size
        self.w_height = kernel_size[0]
        self.w_width = kernel_size[1]

        self.in_height = None
        self.in_width = None

        self.out_height = None
        self.out_width = None
        
    
    def forward(self, x):
        bs, c, h, w = x.size()

        self.in_height = h
        self.in_width = w

        self.out_height = int((h - self.w_height) / self.stride) + 1
        self.out_width = int((w - self.w_width) / self.stride) + 1

        
        for i in range(self.out_height):
            for j in range(self.out_width):
                start_i = i * self.stride
                start_j = j * self.stride
                end_i = start_i + self.w_height
                end_j = start_j + self.w_width
                out_ = self.spp_layer(x[:, :, start_i: end_i, start_j: end_j])
                if i==0 and j==0:
                    out = torch.zeros(out_.size()[0], out_.size()[1], self.out_height, self.out_width)
                out[:,:,i,j] = out_
        return out


if __name__ == "__main__":
    t1 = time.time()
    a = torch.rand(4,64,32,32)
    l = CustomSPPLayer([17, 17], 2)
    m = l(a)
    print(m.size())
    print(time.time()-t1)

    t2 = time.time()
    a = torch.rand(4,256,17,17)
    l = SPPLayer(2)
    m = l(a)
    print(m.size())
    print((time.time()-t2)*17)

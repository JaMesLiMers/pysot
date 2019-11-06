
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

import time

class MemoryBase(nn.Module):

    def __init__(self, memory_size, key_dim, value_dim, value_h, value_w):
        super(MemoryBase, self).__init__()
        # init memory arguments
        self.memory_size = memory_size
        self.key_dim = key_dim
        self.value_h = value_h
        self.value_w = value_w 
        self.value_dim = value_dim
        # initialize the memory block
        self.key_memory = torch.randn((self.memory_size, self.key_dim))
        self.value_memory = torch.randn((self.memory_size, self.value_dim, self.value_h, self.value_w))

        # init the u as 0s
        self.memory_u = torch.zeros(self.memory_size)
        # define the similarty and activate functions
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.softmax = nn.Softmax(dim=0)
        self.sigmoid = nn.Sigmoid()

    def memory_read(self, x_key):
        """ Memory read, input x_key to give output z """
        cos_res = self.cos_sim(x_key, self.key_memory)
        weight = self.softmax(cos_res)
        weight = torch.unsqueeze(weight, 1)
        weight = torch.unsqueeze(weight, 2)
        weight = torch.unsqueeze(weight, 3) 
        result = weight * self.value_memory
        result = torch.sum(result, dim=0, keepdim=True)
        return result
        
    def memory_write(self, z_key, z_value, gamma=0.4, c_init=1):
        """ Memory write, input z_key and z_value, then update memory according to u and update u. 
            (if similarity is less than 0.9)"""
        cos_res = self.cos_sim(z_key, self.key_memory)
        weight = self.sigmoid(cos_res)
        # print(weight)
        if torch.max(weight) < 0.9:
            min_idx_u = torch.argmin(self.memory_u)
            self.key_memory[min_idx_u] = z_key.squeeze()
            self.value_memory[min_idx_u] = z_value.squeeze()
            one_hot = torch.zeros(self.memory_size).scatter_(0, min_idx_u, 1)
            self.memory_u = (1 - one_hot) * (gamma * self.memory_u + weight) + one_hot * c_init
            print(self.memory_u)

    def fusion(self, z_value, f_z_value, alpha=0.7):
        """fuse the z_value we get and first z"""
        return alpha * z_value + (1 - alpha) * f_z_value

    def forward(self, x_key, f_z_value):
        """memory read and fusion"""
        new_z = self.memory_read(x_key)
        result = self.fusion(new_z, f_z_value)
        return result


if __name__ == '__main__':
    # test of memory
    t1 = time.time()
    # init test data
    # for z and x
    memory = MemoryBase(10, 60, 5, 10, 10)
    for i in range(10):
        z_key = torch.randn(1, 60)
        z_value = torch.randn(1, 5, 10, 10)
        x_key = torch.randn(1, 60)
        memory.memory_write(z_key, z_value)
        read_z = memory.memory_read(x_key)
        read_z_ = memory(x_key, z_value)
        print(read_z.size())
        print(read_z_.size())


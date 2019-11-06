
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from pysot.models.memory_network import spp_layer

import time


def conv2d_dw_group(x, kernel):
    """depthwise conv2d"""
    batch, channel = kernel.shape[:2]
    x = x.view(1, batch*channel, x.size(2), x.size(3))  # 1 * (b*c) * k * k
    kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))  # (b*c) * 1 * H * W
    out = F.conv2d(x, kernel, groups=batch*channel)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out


class DepthCorr(nn.Module):
    """depth correlation"""
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3):
        super(DepthCorr, self).__init__()
        # adjust layer for asymmetrical features
        # 对z的转换变为hidden的channel的kernel目标
        self.conv_kernel = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, padding=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),   # 对上一层直接进行修改
                )
        # 对x的转换变为hidden的channel的搜索目标
        self.conv_search = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, padding=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                )
        # 进行深度可分离卷积后的channelwise卷积，产出目标个channel
        self.head = nn.Sequential(
                nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, out_channels, kernel_size=1)
                )

    def forward_corr(self, kernel, input):
        """ 进行深度可分离卷积 """
        # 包装输入到hidden的channel
        kernel = self.conv_kernel(kernel)
        # 包装输出到hidden的channel
        input = self.conv_search(input)
        # 进行深度可分离卷积
        feature = conv2d_dw_group(input, kernel)
        return feature

    def forward(self, kernel, search):
        """先变为hidden channel的大小作深度可分离卷积，后channel wise卷积产生目标channel的数量"""
        feature = self.forward_corr(kernel, search)
        # 进行channel wise的卷积
        out = self.head(feature)
        return out


class KeyGenerator(nn.Module):
    """to generate input x and z's key"""
    def __init__(self, in_channels, hidden, key_channels, z_size):
        super(KeyGenerator, self).__init__()
        # define the input channel and output channel
        self.in_channels = in_channels
        self.hidden = hidden
        self.key_channels = key_channels

        # the template size
        self.z_size = z_size

        # define a layer to generate key map
        self.key_map_generator = nn.Sequential(
            nn.Conv2d(self.in_channels, self.key_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(inplace=True))

        # a layers to generate attention map
        self.depth_corr = DepthCorr(self.in_channels, self.hidden, 1)
        self.softmax = nn.Softmax(2)

        # spatial pooling for input x and z(in different ways)
        self.spp_z = spp_layer.SPPLayer(num_levels=3, pool_type='max_pool')
        self.spp_x = spp_layer.CustomSPPLayer(
            kernel_size=[self.z_size, self.z_size], num_levels=3, pool_type='max_pool')

    def key_map(self, x):
        """generate key map"""
        out = self.key_map_generator(x)
        return out
    
    def attention_mask(self, z, x):
        """to generate attention mask"""
        out = self.depth_corr(z, x)
        out = self.softmax(out.view(*out.size()[:2], -1)).view_as(out)
        return out

    def key_vector_x(self, x):
        """for input x's, key tensor (1,c,h,w)"""
        out = self.spp_x(x)
        return out
    
    def key_vector_z(self, z):
        """for input z's, key vector (1,c)"""
        out = self.spp_z(z)
        return out

    def forward(self, last_z, x):
        """for input last frame z and x, output a x_key_vector (1, c)"""
        x_key = self.key_map(x)
        x_att = self.attention_mask(last_z, x)
        x_key_tensor = self.key_vector_x(x_key)
        x_key_masked = x_key_tensor.mul(x_att)
        x_key_vec = torch.sum(x_key_masked, [2,3])
        return x_key, x_att, x_key_tensor, x_key_masked, x_key_vec
        
if __name__ == "__main__":
    t1 = time.time()
    # init test data
    # for z and x
    z1 = torch.ones(2, 5, 10, 10)
    x1 = torch.ones(2, 5, 15, 15)
    #test the key generator
    key_generator = KeyGenerator(in_channels=5, hidden=5, key_channels=2, z_size=10)
    x_key, x_att, x_key_tensor, x_key_masked, x_key_vec = key_generator(z1, x1)
    print(x_key.size())
    print(x_att.size())
    print(x_key_tensor.size())
    print(x_key_masked.size())
    # print(x_key_masked)
    print(x_key_vec.size())
    print(time.time() - t1)
from tkinter.tix import Tree
from turtle import forward
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from .modelio import LoadableModel, store_config_args
import tensorflow as tf
from . import layers
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
# def iteration(k:int, x:torch.tensor, )
from tqdm import tqdm
import copy

class G_block(nn.Module):
    def __init__(self, inshape=None):
        super(G_block, self).__init__()

        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # self.network = nn.ModuleList()
        # for i in range(iter_num):
        self.conv = nn.Sequential(
            nn.Conv3d(3,16,3,1,1), # padding = 1
            nn.Conv3d(16,32,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32,32,3,1,1)
        )
    
    def forward(self,x):
        return self.conv(x)

class G_back_block(nn.Module):
    def __init__(self, inshape=None):
        super(G_back_block, self).__init__()

        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # self.network = nn.ModuleList()
        # for i in range(iter_num):
        self.conv = nn.Sequential(
            nn.Conv3d(32,32,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32,16,3,1,1),
            nn.Conv3d(16,3,3,1,1), # padding = 1
        )
    
    def forward(self,x):
        return self.conv(x)

class softhreshold(nn.Module): # soft函数
    def __init__(self,inshape):
        super(softhreshold,self).__init__()
        self.theta = torch.randn(inshape, requires_grad=True).to(device)
    
    def forward(self, x:torch.tensor):
        soft = torch.sign(x) * torch.max(torch.abs(x) - self.theta, torch.zeros(x.shape).to(device))
        return soft

class NLBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, mode='embedded', 
                 dimension=3, bn_layer=True):
        """Implementation of Non-Local Block with 4 different pairwise functions but doesn't include subsampling trick
        args:
            in_channels: original channel size (1024 in the paper)
            inter_channels: channel size inside the block if not specifed reduced to half (512 in the paper)
            mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation
            dimension: can be 1 (temporal), 2 (spatial), 3 (spatiotemporal)
            bn_layer: whether to add batch norm
        """
        super(NLBlockND, self).__init__()

        assert dimension in [1, 2, 3]
        
        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')
            
        self.mode = mode
        self.dimension = dimension

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # the channel size is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        
        # assign appropriate convolutional, max pool, and batch norm layers for different dimensions
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        # function g in the paper which goes through conv. with kernel size 1
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        # add BatchNorm layer after the last conv layer
        if bn_layer:
            self.W_z = nn.Sequential(
                    conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                    bn(self.in_channels)
                )
            # from section 4.1 of the paper, initializing params of BN ensures that the initial state of non-local block is identity mapping
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)

            # from section 3.3 of the paper by initializing Wz to 0, this block can be inserted to any existing architecture
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        # define theta and phi for all operations except gaussian
        if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        
        if self.mode == "concatenate":
            self.W_f = nn.Sequential(
                    nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1),
                    nn.ReLU()
                )
            
    def forward(self, x):
        """
        args
            x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
        """

        batch_size = x.size(0)
        
        # (N, C, THW)
        # this reshaping and permutation is from the spacetime_nonlocal function in the original Caffe2 implementation
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        if self.mode == "gaussian":
            theta_x = x.view(batch_size, self.in_channels, -1)
            phi_x = x.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "embedded" or self.mode == "dot":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "concatenate":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)
            
            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)
            
            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))
        
        if self.mode == "gaussian" or self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)
        elif self.mode == "dot" or self.mode == "concatenate":
            N = f.size(-1) # number of position in x
            f_div_C = f / N
        
        y = torch.matmul(f_div_C, g_x)
        
        # contiguous here just allocates contiguous chunk of memory
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        
        W_y = self.W_z(y)
        # residual connection
        z = W_y + x

        return z

# class NLBlockND(nn.Module):
#     def __init__(self):
#         super().__init__()

class EPN_Net(LoadableModel):
    @store_config_args
    def __init__(self, inshape, f_function=None, iteration_S=int):
        super().__init__()
        # self.alpha = torch.randn(inshape).to(device)# 或许需要初始化函数的
        self.alpha = []
        self.inshape = inshape
        self.gamma = []
        self.theta = []
        self.history = []
        # self.iter_num = iter_num * 2
        self.f = f_function
        self.transformer = layers.SpatialTransformer(inshape)
        self.block_sequence = nn.ModuleList()

        self.forward_block = nn.Sequential(
            G_block(inshape=inshape),
            softhreshold(inshape=inshape),
            # NLBlockND(in_channels=32),
            G_back_block(inshape=inshape)
        )

        self.back_block = nn.Sequential(
            G_block(inshape=inshape),
            softhreshold(inshape=inshape),
            # NLBlockND(in_channels=32),
            G_back_block(inshape=inshape)
        )

        for i in range(iteration_S):
            self.block_sequence.append(nn.Sequential(self.forward_block, self.back_block))

            self.alpha.append(torch.randn(inshape, requires_grad=True).to(device))
            self.gamma.append(torch.randn(inshape, requires_grad=True).to(device))
            self.theta.append(torch.randn(inshape, requires_grad=True).to(device))

        self.iteration = iteration_S
        self.beta = self.alpha
        # for i in range(self.)
    
    def forward(self,x, f, m): # k: x为图片列表中第k项
        '''
        x: phi
        m: moving img
        f: fixed img
        '''
        # x_history: [x_+0.5]
        # 注：需要修改generator格式，使其每次执行时告诉模型，对应的图像是哪张
        x_new = torch.randn(self.inshape, requires_grad=True).to(device)

        for k in range(self.iteration):
            if len(self.history) != 0:
                new_temp = x_new.data + self.gamma[k] * (x_new.data - self.history[0].data) # 上一个，就是上一phase中的第二个
            else:
                new_temp = x.data
                self.history = [new_temp]

            x_tieta_k = new_temp.detach()
            x_tieta_k.requires_grad_(True)

            loss = self.f(f,self.transformer(m, x_tieta_k))# 标量梯度？
            x_tieta_k.retain_grad()
            loss.backward()

            b_half = x_tieta_k - self.alpha[k] * x_tieta_k.grad #requires_grad = true

            x_forward = self.block_sequence[k][0](b_half)
            x_half = x_forward + b_half

            self.history = [x_half]

            # temp = x_half + self.gamma[k] * (x_half - x)
            x_final = x_half + self.gamma[k] * (x_half - x)
            x_2 = x_final.detach()
            # x_final.retain_grad()

            x_2.requires_grad_(True)
            x_2.retain_grad()

            y = self.f(f, self.transformer(m, x_2))
            y.backward()
            b = x_2 - self.beta[k] * x_2.grad
            x_2.retain_grad()

            new_temp = self.block_sequence[k][1](b) + b
            x_new = new_temp.detach() # x_k+1
            x_new.requires_grad_(True)

            x_2.grad.zero_()

        return x



            

p = torch.tensor([1,2,3]).float().requires_grad_(True)
q = torch.tensor([7,8,9]).float().requires_grad_(True)
y = p * q
y.sum().backward()
print(p.grad)


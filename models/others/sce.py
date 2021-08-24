""" code reference: https://github.com/ShuaiyiHuang/DCCNet """

'''
Shuaiyi Huang
Implement Spatial Context Encoder.
'''

import torch
import torch.nn as nn
import torch.nn.functional as Func
from torch.nn.modules.utils import _quadruple


def generate_spatial_descriptor(data, kernel_size):
    '''
    Applies self local similarity with fixed sliding window.
    Args:
        data: featuren map, variable of shape (b,c,h,w)
        kernel_size: width/heigh of local window, int
    Returns:
        output: global spatial map, variable of shape (b,c,h,w)
    '''

    padding = int(kernel_size // 2)  # 5.7//2 = 2.0, 5.2//2 = 2.0
    b, c, h, w = data.shape
    p2d = _quadruple(padding)  # (pad_l,pad_r,pad_t,pad_b)
    data_padded = Func.pad(data, p2d, 'constant', 0)  # output variable
    assert data_padded.shape == (
    b, c, (h + 2 * padding), (w + 2 * padding)), 'Error: data_padded shape{} wrong!'.format(data_padded.shape)

    output = torch.zeros(size=[b, kernel_size * kernel_size, h, w], requires_grad=data.requires_grad)
    if data.is_cuda:
        output = output.cuda(data.get_device())

    for hi in range(h):
        for wj in range(w):
            q = data[:, :, hi, wj].contiguous()  # (b,c)
            i = hi + padding  # h index in datapadded
            j = wj + padding  # w index in datapadded

            hs = i - padding
            he = i + padding + 1
            ws = j - padding
            we = j + padding + 1
            patch = data_padded[:, :, hs:he, ws:we].contiguous()  # (b,c,k,k)
            assert (patch.shape == (b, c, kernel_size, kernel_size))
            hk, wk = kernel_size, kernel_size

            # reshape features for matrix multiplication
            feature_a = q.view(b, c, 1 * 1).transpose(1, 2)  # (b,1,c) input is not contigous
            feature_b = patch.view(b, c, hk * wk)  # (b,c,L)

            # perform matrix mult.
            feature_mul = torch.bmm(feature_a, feature_b)  # (b,1,L)
            assert (feature_mul.shape == (b, 1, hk * wk))
            # indexed [batch,row_A,col_A,row_B,col_B]
            correlation_tensor = feature_mul.unsqueeze(1)  # (b,L)
            output[:, :, hi, wj] = correlation_tensor.squeeze(1).squeeze(1)

    return output


def featureL2Norm(feature):
    epsilon = 1e-6
    norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
    return torch.div(feature, norm)


class SpatialContextEncoder(torch.nn.Module):
    '''
    Spatial Context Encoder.
    Author: Shuaiyi Huang
    Input:
        x: feature of shape (b,c,h,w)
    Output:
        feature_embd: context-aware semantic feature of shape (b,c+k**2,h,w), where k is the kernel size of spatial descriptor
    '''

    def __init__(self, planes=None, kernel_size=None):
        super(SpatialContextEncoder, self).__init__()
        self.kernel_size = kernel_size
        self.conv1x1_in = nn.Sequential(nn.Conv2d(planes[0], planes[1], kernel_size=1, bias=False, padding=0),
                                        nn.BatchNorm2d(planes[1]),
                                        nn.ReLU(inplace=True))
        self.embeddingFea = nn.Sequential(nn.Conv2d(planes[1] + kernel_size * kernel_size, planes[2],
                                                    kernel_size=1, bias=False, padding=0),
                                          nn.BatchNorm2d(planes[2]),
                                          nn.ReLU(inplace=True))
        self.conv1x1_out = nn.Sequential(
            nn.Conv2d(planes[2], planes[3], kernel_size=1, bias=False, padding=0),
            nn.BatchNorm2d(planes[3]))

        print('SpatialContextEncoder initialization: input_dim {},hidden_dim {}'.format(planes[1], planes[2]))
        return

    def forward(self, x):
        # channel reduction
        x = self.conv1x1_in(x)
        feature_gs = generate_spatial_descriptor(x, kernel_size=self.kernel_size)

        # Add L2norm
        feature_gs = featureL2Norm(feature_gs)

        # concatenate
        feature_cat = torch.cat([x, feature_gs], 1)

        # embed
        feature_embd = self.embeddingFea(feature_cat)

        # channel expansion
        feature_embd = self.conv1x1_out(feature_embd)
        return feature_embd

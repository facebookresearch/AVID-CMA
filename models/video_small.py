import torch
import torch.nn as nn
from models.network_blocks import Basic3DBlock, BasicR2P1DBlock

__all__ = [
    'R2Plus1D_Small',
    'VGGish3D_Small'
]

class R2Plus1D_Small(nn.Module):
    def __init__(self, depth=18):
        super(R2Plus1D_Small, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), padding=(1, 3, 3), stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        if depth == 10:
            self.conv2x = BasicR2P1DBlock(64, 64)
            self.conv3x = BasicR2P1DBlock(64, 128, stride=(2, 2, 2))
            self.conv4x = BasicR2P1DBlock(128, 256, stride=(2, 2, 2))
            self.conv5x = BasicR2P1DBlock(256, 512, stride=(2, 2, 2))
        elif depth == 18:
            self.conv2x = nn.Sequential(BasicR2P1DBlock(64, 64), BasicR2P1DBlock(64, 64))
            self.conv3x = nn.Sequential(BasicR2P1DBlock(64, 128, stride=(2, 2, 2)), BasicR2P1DBlock(128, 128))
            self.conv4x = nn.Sequential(BasicR2P1DBlock(128, 256, stride=(2, 2, 2)), BasicR2P1DBlock(256, 256))
            self.conv5x = nn.Sequential(BasicR2P1DBlock(256, 512, stride=(2, 2, 2)), BasicR2P1DBlock(512, 512))

        self.pool = nn.AdaptiveMaxPool3d((1, 1, 1))
        self.out_dim = 512

    def forward(self, x, return_embs=False):
        x_c1 = self.conv1(x)
        x_b1 = self.conv2x(x_c1)
        x_b2 = self.conv3x(x_b1)
        x_b3 = self.conv4x(x_b2)
        x_b4 = self.conv5x(x_b3)
        x_pool = self.pool(x_b4)
        if return_embs:
            return {'conv1': x_c1, 'conv2x': x_b1, 'conv3x': x_b2, 'conv4x': x_b3, 'conv5x': x_b4, 'pool': x_pool}
        else:
            return x_pool

    def FLOPs(self, inpt_size):
        import numpy as np
        size = self.conv1(torch.randn(inpt_size, device=self.conv1[0].weight.device)).shape
        flops = np.prod(self.conv1[0].weight.shape) * np.prod(size[2:])

        for convx in [self.conv2x, self.conv3x, self.conv4x, self.conv5x]:
            for mdl in convx:
                flops_tmp, size = mdl.FLOPs(size)
                flops += flops_tmp

        return flops, size


class VGGish3D_Small(nn.Module):
    def __init__(self, depth):
        super(VGGish3D_Small, self).__init__()
        assert depth == 10

        self.conv1 = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=7, padding=3, stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.block1 = Basic3DBlock(64, 64)
        self.block2 = Basic3DBlock(64, 128, stride=(2, 2, 2))
        self.block3 = Basic3DBlock(128, 256, stride=(2, 2, 2))
        self.block4 = Basic3DBlock(256, 512)

        self.pool = nn.AdaptiveMaxPool3d((1, 1, 1))
        self.out_dim = 512

    def forward(self, x, return_embs=False):
        x_c1 = self.conv1(x)
        x_b1 = self.block1(x_c1)
        x_b2 = self.block2(x_b1)
        x_b3 = self.block3(x_b2)
        x_b4 = self.block4(x_b3)
        x_pool = self.pool(x_b4)

        if return_embs:
            return {'conv1': x_c1, 'block1': x_b1, 'block2': x_b2, 'block3': x_b3, 'block4': x_b4, 'pool': x_pool}
        else:
            return x_pool

    def FLOPs(self, inpt_size):
        import numpy as np
        size = self.conv1(torch.randn(inpt_size)).shape
        flops = np.prod(self.conv1[0].weight.shape) * np.prod(size[2:])

        flops_tmp, size = self.block1.FLOPs(size)
        flops += flops_tmp

        flops_tmp, size = self.block2.FLOPs(size)
        flops += flops_tmp

        flops_tmp, size = self.block3.FLOPs(size)
        flops += flops_tmp

        flops_tmp, size = self.block4.FLOPs(size)
        flops += flops_tmp
        return flops, size



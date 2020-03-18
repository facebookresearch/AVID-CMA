"""
Adapted from https://github.com/facebookresearch/VMZ/blob/4c14ee6f8eae8e2ac97fc4c05713b8a112eb1f28/lib/models/video_model.py
"""
import torch
import torch.nn as nn
from models.network_blocks import Basic3DBlock, Basic3DResBlock, BasicR2P1DBlock, Basic3DBlockSpatial


class R2Plus1D(nn.Module):
    def __init__(self, depth=18):
        super(R2Plus1D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), padding=(1, 3, 3), stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
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
        elif depth == 34:
            self.conv2x = nn.Sequential(BasicR2P1DBlock(64, 64), BasicR2P1DBlock(64, 64), BasicR2P1DBlock(64, 64))
            self.conv3x = nn.Sequential(BasicR2P1DBlock(64, 128, stride=(2, 2, 2)), BasicR2P1DBlock(128, 128), BasicR2P1DBlock(128, 128), BasicR2P1DBlock(128, 128))
            self.conv4x = nn.Sequential(BasicR2P1DBlock(128, 256, stride=(2, 2, 2)), BasicR2P1DBlock(256, 256), BasicR2P1DBlock(256, 256), BasicR2P1DBlock(256, 256), BasicR2P1DBlock(256, 256), BasicR2P1DBlock(256, 256))
            self.conv5x = nn.Sequential(BasicR2P1DBlock(256, 512, stride=(2, 2, 2)), BasicR2P1DBlock(512, 512), BasicR2P1DBlock(512, 512))
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
            try:
                for mdl in convx:
                    flops_tmp, size = mdl.FLOPs(size)
                    flops += flops_tmp
            except Exception:
                flops_tmp, size = convx.FLOPs(size)
                flops += flops_tmp


        return flops, size

    def Params(self):
        import numpy as np
        params = np.sum([np.prod(p.shape) for p in self.parameters()])
        return params


# 3D-Rotation style
class Resnet3D(nn.Module):
    def __init__(self, depth=18):
        super(Resnet3D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(7, 7, 7), padding=(3, 3, 3), stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )

        if depth == 10:
            self.conv2x = Basic3DResBlock(64, 64)
            self.conv3x = Basic3DResBlock(64, 128, stride=(2, 2, 2))
            self.conv4x = Basic3DResBlock(128, 256, stride=(2, 2, 2))
            self.conv5x = Basic3DResBlock(256, 512, stride=(2, 2, 2))
        elif depth == 18:
            self.conv2x = nn.Sequential(Basic3DResBlock(64, 64), Basic3DResBlock(64, 64))
            self.conv3x = nn.Sequential(Basic3DResBlock(64, 128, stride=(2, 2, 2)), Basic3DResBlock(128, 128))
            self.conv4x = nn.Sequential(Basic3DResBlock(128, 256, stride=(2, 2, 2)), Basic3DResBlock(256, 256))
            self.conv5x = nn.Sequential(Basic3DResBlock(256, 512, stride=(2, 2, 2)), Basic3DResBlock(512, 512))
        elif depth == 34:
            self.conv2x = nn.Sequential(Basic3DResBlock(64, 64), Basic3DResBlock(64, 64), Basic3DResBlock(64, 64))
            self.conv3x = nn.Sequential(Basic3DResBlock(64, 128, stride=(2, 2, 2)), Basic3DResBlock(128, 128), Basic3DResBlock(128, 128), Basic3DResBlock(128, 128))
            self.conv4x = nn.Sequential(Basic3DResBlock(128, 256, stride=(2, 2, 2)), Basic3DResBlock(256, 256), Basic3DResBlock(256, 256), Basic3DResBlock(256, 256), Basic3DResBlock(256, 256), Basic3DResBlock(256, 256))
            self.conv5x = nn.Sequential(Basic3DResBlock(256, 512, stride=(2, 2, 2)), Basic3DResBlock(512, 512), Basic3DResBlock(512, 512))
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
            return {'conv2x': x_b1, 'conv3x': x_b2, 'conv4x': x_b3, 'conv5x': x_b4, 'pool': x_pool}
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

    def Params(self):
        import numpy as np
        params = np.sum([np.prod(p.shape) for p in self.parameters()])
        return params


# AVTS style
class VGGishMC3(nn.Module):
    def __init__(self):
        super(VGGishMC3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), padding=(1, 3, 3), stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.conv2x = nn.Sequential(Basic3DBlock(64, 64))
        self.conv3x = nn.Sequential(Basic3DBlockSpatial(64, 128, stride=(2, 2, 2)))
        self.conv4x = nn.Sequential(Basic3DBlockSpatial(128, 256, stride=(2, 2, 2)))
        self.conv5x = nn.Sequential(Basic3DBlockSpatial(256, 512, stride=(1, 2, 2)))

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

    def Params(self):
        import numpy as np
        params = np.sum([np.prod(p.shape) for p in self.parameters()])
        return params


def main():
    pass

if __name__ == '__main__':
    main()
#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

mkdir -p checkpoints/AVID/Kinetics/Cross-N1024
wget -O checkpoints/AVID/Kinetics/Cross-N1024/checkpoint.pth.tar https://dl.fbaipublicfiles.com/avid-cma/checkpoints/AVID_Kinetics_Cross-N1024_checkpoint.pth.tar

mkdir -p checkpoints/AVID-CMA/Kinetics/InstX-N1024-PosW-N64-Top32
wget -O checkpoints/AVID-CMA/Kinetics/InstX-N1024-PosW-N64-Top32/checkpoint.pth.tar https://dl.fbaipublicfiles.com/avid-cma/checkpoints/AVID-CMA_Kinetics_InstX-N1024-PosW-N64-Top32_checkpoint.pth.tar

mkdir -p checkpoints/AVID/Audioset/Cross-N1024
wget -O checkpoints/AVID/Audioset/Cross-N1024/checkpoint.pth.tar https://dl.fbaipublicfiles.com/avid-cma/checkpoints/AVID_Audioset_Cross-N1024_checkpoint.pth.tar

mkdir -p checkpoints/AVID-CMA/Audioset/InstX-N1024-PosW-N64-Top32
wget -O checkpoints/AVID-CMA/Audioset/InstX-N1024-PosW-N64-Top32/checkpoint.pth.tar https://dl.fbaipublicfiles.com/avid-cma/checkpoints/AVID-CMA_Audioset_InstX-N1024-PosW-N64-Top32_checkpoint.pth.tar


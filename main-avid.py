# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os
import random
import time
import warnings
import yaml

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp

import utils.logger
from utils import main_utils

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('cfg', help='model directory')
parser.add_argument('--quiet', action='store_true')

parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:15475', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')


def main():
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.cfg))

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, cfg))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args, cfg)


def main_worker(gpu, ngpus_per_node, args, cfg):
    args.gpu = gpu

    # Setup environment
    args = main_utils.initialize_distributed_backend(args, ngpus_per_node)
    logger, tb_writter, model_dir = main_utils.prep_environment(args, cfg)

    # Define model
    model = main_utils.build_model(cfg['model'], logger)
    model, args, cfg['dataset']['batch_size'], cfg['num_workers'] = main_utils.distribute_model_to_cuda(model, args, cfg['dataset']['batch_size'], cfg['num_workers'], ngpus_per_node)

    # Define dataloaders
    train_loader = main_utils.build_dataloaders(cfg['dataset'], cfg['num_workers'], args.distributed, logger)

    # Define criterion
    device = args.gpu if args.gpu is not None else 0
    cfg['loss']['args']['embedding_dim'] = model.module.out_dim
    cfg['loss']['args']['device'] = device
    train_criterion = main_utils.build_criterion(cfg['loss'], logger=logger)

    # Define optimizer
    optimizer, scheduler = main_utils.build_optimizer(
        params=list(model.parameters())+list(train_criterion.parameters()),
        cfg=cfg['optimizer'],
        logger=logger)
    ckp_manager = main_utils.CheckpointManager(model_dir, rank=args.rank)

    # Optionally resume from a checkpoint
    start_epoch, end_epoch = 0, cfg['optimizer']['num_epochs']
    if cfg['resume']:
        if ckp_manager.checkpoint_exists(last=True):
            start_epoch = ckp_manager.restore(restore_last=True, model=model, optimizer=optimizer, train_criterion=train_criterion)
            scheduler.step(start_epoch)
            logger.add_line("Checkpoint loaded: '{}' (epoch {})".format(ckp_manager.last_checkpoint_fn(), start_epoch))
        else:
            logger.add_line("No checkpoint found at '{}'".format(ckp_manager.last_checkpoint_fn()))

    cudnn.benchmark = True

    ############################ TRAIN #########################################
    test_freq = cfg['test_freq'] if 'test_freq' in cfg else 1
    for epoch in range(start_epoch, end_epoch):
        if epoch in cfg['optimizer']['lr']['milestones']:
            ckp_manager.save(epoch, model=model, train_criterion=train_criterion, optimizer=optimizer, filename='checkpoint-ep{}.pth.tar'.format(epoch))
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        scheduler.step(epoch)
        train_criterion.set_epoch(epoch)

        # Train for one epoch
        logger.add_line('='*30 + ' Epoch {} '.format(epoch) + '='*30)
        logger.add_line('LR: {}'.format(scheduler.get_lr()))
        run_phase('train', train_loader, model, optimizer, train_criterion, epoch, args, cfg, logger, tb_writter)
        if epoch % test_freq == 0 or epoch == end_epoch - 1:
            ckp_manager.save(epoch+1, model=model, optimizer=optimizer, train_criterion=train_criterion)


def run_phase(phase, loader, model, optimizer, criterion, epoch, args, cfg, logger, tb_writter):
    from utils import metrics_utils
    logger.add_line('\n{}: Epoch {}'.format(phase, epoch))
    batch_time = metrics_utils.AverageMeter('Time', ':6.3f', window_size=100)
    data_time = metrics_utils.AverageMeter('Data', ':6.3f', window_size=100)
    loss_meter = metrics_utils.AverageMeter('Loss', ':.3e')
    progress = utils.logger.ProgressMeter(len(loader), [batch_time, data_time, loss_meter],
                                          phase=phase, epoch=epoch, logger=logger, tb_writter=tb_writter)

    # switch to train mode
    model.train(phase == 'train')

    end = time.time()
    device = args.gpu if args.gpu is not None else 0
    for i, sample in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # Prepare batch
        video, audio, index = sample['frames'], sample['audio'], sample['index']
        video = video.cuda(device, non_blocking=True)
        audio = audio.cuda(device, non_blocking=True)
        index = index.cuda(device, non_blocking=True)

        # compute audio and video embeddings
        if phase == 'train':
            video_emb, audio_emb = model(video, audio)
        else:
            with torch.no_grad():
                video_emb, audio_emb = model(video, audio)

        # compute loss
        loss, loss_debug = criterion(video_emb, audio_emb, index)
        loss_meter.update(loss.item(), video.size(0))

        # compute gradient and do SGD step during training
        if phase == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print to terminal and tensorboard
        step = epoch * len(loader) + i
        if (i+1) % cfg['print_freq'] == 0 or i == 0 or i+1 == len(loader):
            progress.display(i+1)
            if tb_writter is not None:
                for key in loss_debug:
                    tb_writter.add_scalar('{}-batch/{}'.format(phase, key), loss_debug[key].item(), step)

    # Sync metrics across all GPUs and print final averages
    if args.distributed:
        progress.synchronize_meters(args.gpu)
        progress.display(len(loader)*args.world_size)

    if tb_writter is not None:
        for meter in progress.meters:
            tb_writter.add_scalar('{}-epoch/{}'.format(phase, meter.name), meter.avg, epoch)


if __name__ == '__main__':
    main()

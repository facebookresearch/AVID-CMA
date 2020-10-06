# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import time
import yaml
import torch

from utils import main_utils, eval_utils
import utils.logger
import torch.multiprocessing as mp


parser = argparse.ArgumentParser(description='Evaluation on ESC Sound Classification')
parser.add_argument('cfg', metavar='CFG', help='config file')
parser.add_argument('model_cfg', metavar='CFG', help='config file')
parser.add_argument('--quiet', action='store_true')
parser.add_argument('--test-only', action='store_true')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--distributed', action='store_true')
parser.add_argument('--port', default='1234')


def main():
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.cfg))
    if args.test_only:
        cfg['test_only'] = True
    if args.resume:
        cfg['resume'] = True
    if args.debug:
        cfg['num_workers'] = 1
        cfg['dataset']['batch_size'] = 4

    ngpus = torch.cuda.device_count()
    for fold in range(1, cfg['dataset']['num_folds']+1):
        if args.distributed:
            mp.spawn(main_worker, nprocs=ngpus, args=(ngpus, fold, args, cfg))
        else:
            main_worker(None, ngpus, fold, args, cfg)


def main_worker(gpu, ngpus, fold, args, cfg):
    args.gpu = gpu
    args.world_size = ngpus

    # Prepare folder and logger
    eval_dir, model_cfg, logger = eval_utils.prepare_environment(args, cfg, fold)

    # Model
    model, ckp_manager = eval_utils.build_model(model_cfg, cfg, eval_dir, args, logger)

    # Optimizer
    optimizer, scheduler = main_utils.build_optimizer(model.parameters(), cfg['optimizer'], logger)

    # Datasets
    train_loader, test_loader, dense_loader = eval_utils.build_dataloaders(
        cfg['dataset'], fold, cfg['num_workers'], args.distributed, logger)

    ################################ Train ################################
    start_epoch, end_epoch = 0, cfg['optimizer']['num_epochs']
    if (cfg['resume'] or args.test_only) and ckp_manager.checkpoint_exists(last=True):
        start_epoch = ckp_manager.restore(model, optimizer, scheduler, restore_last=True)
        logger.add_line("Loaded checkpoint '{}' (epoch {})".format(ckp_manager.last_checkpoint_fn(), start_epoch))

    if not cfg['test_only']:
        logger.add_line("=" * 30 + "   Training   " + "=" * 30)
        for epoch in range(start_epoch, end_epoch):
            scheduler.step(epoch=epoch)
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
                test_loader.sampler.set_epoch(epoch)

            logger.add_line('='*30 + ' Epoch {} '.format(epoch) + '='*30)
            logger.add_line('LR: {}'.format(scheduler.get_lr()))
            run_phase('train', train_loader, model, optimizer, epoch, args, cfg, logger)
            run_phase('test', test_loader, model, None, epoch, args, cfg, logger)
            ckp_manager.save(model, optimizer, scheduler, epoch)

    ################################ Eval ################################
    logger.add_line('\n' + '=' * 30 + ' Final evaluation ' + '=' * 30)
    cfg['dataset']['test']['clips_per_video'] = 25
    train_loader, test_loader, dense_loader = eval_utils.build_dataloaders(cfg['dataset'], fold, cfg['num_workers'], args.distributed, logger)
    top1_dense, top5_dense = run_phase('test_dense', dense_loader, model, None, end_epoch, args, cfg, logger)
    top1, top5 = run_phase('test', test_loader, model, None, end_epoch, args, cfg, logger)

    logger.add_line('\n' + '=' * 30 + ' Evaluation done ' + '=' * 30)
    for ft in top1:
        logger.add_line('')
        logger.add_line('[{}] Clip@1: {:6.2f}'.format(ft, top1[ft]))
        logger.add_line('[{}] Clip@5: {:6.2f}'.format(ft, top5[ft]))
        logger.add_line('[{}] Video@1: {:6.2f}'.format(ft, top1_dense[ft]))
        logger.add_line('[{}] Video@5: {:6.2f}'.format(ft, top5_dense[ft]))


def run_phase(phase, loader, model, optimizer, epoch, args, cfg, logger):
    from utils import metrics_utils
    logger.add_line('\n{}: Epoch {}'.format(phase, epoch))
    feature_names = cfg['model']['args']['feat_names']
    batch_time = metrics_utils.AverageMeter('Time', ':6.3f', 100)
    data_time = metrics_utils.AverageMeter('Data', ':6.3f', 100)
    loss_meters = {ft: metrics_utils.AverageMeter('Loss', ':.4e', 0) for ft in feature_names}
    top1_meters = {ft: metrics_utils.AverageMeter('Acc@1', ':6.2f', 0) for ft in feature_names}
    top5_meters = {ft: metrics_utils.AverageMeter('Acc@5', ':6.2f', 0) for ft in feature_names}
    progress = {'timers': utils.logger.ProgressMeter(len(loader), meters=[batch_time, data_time], phase=phase, epoch=epoch, logger=logger)}
    progress.update({ft: utils.logger.ProgressMeter(len(loader), meters=[loss_meters[ft], top1_meters[ft], top5_meters[ft]], phase=phase, epoch=epoch, logger=logger) for ft in feature_names})

    # switch to train/test mode
    model.train(phase == 'train')

    if phase in {'test_dense', 'test'}:
        model = BatchWrapper(model, cfg['dataset']['batch_size'])

    end = time.time()
    criterion = torch.nn.CrossEntropyLoss()
    softmax = torch.nn.Softmax(dim=1)
    for it, sample in enumerate(loader):
        data_time.update(time.time() - end)

        video = sample['frames']
        target = sample['label'].cuda()
        if args.gpu is not None:
            video = video.cuda(args.gpu, non_blocking=True)

        if phase == 'test_dense':
            batch_size, clips_per_sample = video.shape[0], video.shape[1]
            video = video.flatten(0, 1).contiguous()

        # compute outputs
        if phase == 'train':
            logits = model(video)
        else:
            with torch.no_grad():
                logits = model(video)

        # compute loss and measure accuracy
        total_loss = 0.
        for ft in feature_names:
            if phase == 'test_dense':
                confidence = softmax(logits[ft]).view(batch_size, clips_per_sample, -1).mean(1)
                target_tiled = target.unsqueeze(1).repeat(1, clips_per_sample).view(-1)
                loss = criterion(logits[ft], target_tiled)
            else:
                confidence = softmax(logits[ft])
                loss = criterion(logits[ft], target)
            total_loss += loss

            with torch.no_grad():
                acc1, acc5 = metrics_utils.accuracy(confidence, target, topk=(1, 5))
                loss_meters[ft].update(loss.item(), target.size(0))
                top1_meters[ft].update(acc1[0].item(), target.size(0))
                top5_meters[ft].update(acc5[0].item(), target.size(0))

        # compute gradient and do SGD step
        if phase == 'train':
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (it + 1) % 100 == 0 or it == 0 or it + 1 == len(loader):
            for ft in progress:
                progress[ft].display(it+1)

    if args.distributed:
        for ft in progress:
            progress[ft].synchronize_meters(args.gpu)
            progress[ft].display(len(loader) * args.world_size)

    return {ft: top1_meters[ft].avg for ft in feature_names}, {ft: top5_meters[ft].avg for ft in feature_names}


class BatchWrapper:
    def __init__(self, model, batch_size):
        self.model = model
        self.batch_size = batch_size

    def __call__(self, x):
        from collections import defaultdict
        outs = defaultdict(list)
        for i in range(0, x.shape[0], self.batch_size):
            odict = self.model(x[i:i + self.batch_size])
            for k in odict:
                outs[k] += [odict[k]]
        for k in outs:
            outs[k] = torch.cat(outs[k], 0)
        return outs


if __name__ == '__main__':
    main()
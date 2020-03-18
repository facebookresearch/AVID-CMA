import argparse
import time
import yaml
import torch

from utils import main_utils, eval_utils
import torch.multiprocessing as mp


parser = argparse.ArgumentParser(description='Evaluation on ESC Sound Classification')
parser.add_argument('cfg', metavar='CFG', help='config file')
parser.add_argument('model_cfg', metavar='CFG', help='config file')
parser.add_argument('--quiet', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--test-only', action='store_true')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--distributed', action='store_true')
parser.add_argument('--port', default='1234')


def main():
    ngpus = torch.cuda.device_count()
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.cfg))
    if args.test_only:
        cfg['test_only'] = True
    if args.resume:
        cfg['resume'] = True
    if args.debug:
        cfg['num_workers'] = 1
        cfg['dataset']['batch_size'] = 32

    if args.distributed:
        mp.spawn(main_worker, nprocs=ngpus, args=(ngpus, cfg['dataset']['fold'], args, cfg))
    else:
        main_worker(None, ngpus, cfg['dataset']['fold'], args, cfg)


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
    if cfg['resume'] and ckp_manager.checkpoint_exists(last=True):
        start_epoch = ckp_manager.restore(model, optimizer, scheduler, restore_last=True)
        logger.add_line("Loaded checkpoint '{}' (epoch {})".format(ckp_manager.last_checkpoint_fn(), start_epoch))

    if not cfg['test_only']:
        logger.add_line("=" * 30 + "   Training   " + "=" * 30)
        # Warmup. Train classifier only for a few epochs.
        if start_epoch == 0 and 'warmup_classifier' in cfg['optimizer'] and cfg['optimizer']['warmup_classifier']:
            n_wu_epochs = cfg['optimizer']['warmup_epochs'] if 'warmup_epochs' in cfg['optimizer'] else 5
            cls_opt, _ = main_utils.build_optimizer(
                params=[p for n, p in model.named_parameters() if 'feature_extractor' not in n],
                cfg={'lr': {'base_lr': cfg['optimizer']['lr']['base_lr'], 'milestones': [n_wu_epochs,], 'gamma': 1.},
                     'weight_decay': cfg['optimizer']['weight_decay'],
                     'name': cfg['optimizer']['name']}
            )
            for epoch in range(n_wu_epochs):
                run_phase('train', train_loader, model, cls_opt, epoch, args, cfg, logger)
                top1, _ = run_phase('test', test_loader, model, None, epoch, args, cfg, logger)

        # Main training loop
        for epoch in range(start_epoch, end_epoch):
            scheduler.step(epoch=epoch)
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
                test_loader.sampler.set_epoch(epoch)

            logger.add_line('='*30 + ' Epoch {} '.format(epoch) + '='*30)
            logger.add_line('LR: {}'.format(scheduler.get_lr()))
            run_phase('train', train_loader, model, optimizer, epoch, args, cfg, logger)
            top1, _ = run_phase('test', test_loader, model, None, epoch, args, cfg, logger)
            ckp_manager.save(model, optimizer, scheduler, epoch, eval_metric=top1)

    ################################ Eval ################################
    logger.add_line('\n' + '=' * 30 + ' Final evaluation ' + '=' * 30)
    cfg['dataset']['test']['clips_per_video'] = 25  # Evaluate clip-level predictions with 25 clips per video for metric stability
    train_loader, test_loader, dense_loader = eval_utils.build_dataloaders(cfg['dataset'], fold, cfg['num_workers'], args.distributed, logger)
    top1_dense, top5_dense = run_phase('test_dense', dense_loader, model, None, end_epoch, args, cfg, logger)
    top1, top5 = run_phase('test', test_loader, model, None, end_epoch, args, cfg, logger)

    logger.add_line('\n' + '=' * 30 + ' Evaluation done ' + '=' * 30)
    logger.add_line('Clip@1: {:6.2f}'.format(top1))
    logger.add_line('Clip@5: {:6.2f}'.format(top5))
    logger.add_line('Video@1: {:6.2f}'.format(top1_dense))
    logger.add_line('Video@5: {:6.2f}'.format(top5_dense))


def run_phase(phase, loader, model, optimizer, epoch, args, cfg, logger):
    logger.add_line('\n{}: Epoch {}'.format(phase, epoch))

    batch_time = main_utils.AverageMeter('Time', ':6.3f', window_size=100)
    data_time = main_utils.AverageMeter('Data', ':6.3f', window_size=100)
    loss_meter = main_utils.AverageMeter('Loss', ':.4e')
    top1_meter = main_utils.AverageMeter('Acc@1', ':6.2f')
    top5_meter = main_utils.AverageMeter('Acc@5', ':6.2f')
    progress = main_utils.ProgressMeter(len(loader), meters=[batch_time, data_time, loss_meter, top1_meter, top5_meter],
                                        phase=phase, epoch=epoch, logger=logger)

    # Distribute batch sequentially
    class BatchWrapper:
        def __init__(self, model, batch_size):
            self.model = model
            self.batch_size = batch_size

        def __call__(self, x):
            outs = []
            for i in range(0, x.shape[0], self.batch_size):
                outs += [self.model(x[i:i + self.batch_size])]
            return torch.cat(outs, 0)

    # switch to train/test mode
    if phase == 'train':
        model.train(True)
    else:
        model.train(False)

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
            video = torch.flatten(video, 0, 1).contiguous()

        # compute outputs
        if phase == 'train':
            logits = model(video)
        else:
            with torch.no_grad():
                logits = model(video)

        # compute loss and measure accuracy
        if phase == 'test_dense':
            confidence = softmax(logits).view(batch_size, clips_per_sample, -1).mean(1)
            labels_tiled = target.unsqueeze(1).repeat(1, clips_per_sample).view(-1)
            loss = criterion(logits, labels_tiled)
        else:
            confidence = softmax(logits)
            loss = criterion(logits, target)

        with torch.no_grad():
            acc1, acc5 = main_utils.accuracy(confidence, target, topk=(1, 5))
            loss_meter.update(loss.item(), target.size(0))
            top1_meter.update(acc1[0], target.size(0))
            top5_meter.update(acc5[0], target.size(0))

        # compute gradient and do SGD step
        if phase == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (it + 1) % 100 == 0 or it == 0 or it + 1 == len(loader):
            progress.display(it+1)

    if args.distributed:
        main_utils.synchronize_meters(progress, args.gpu)
        progress.display(len(loader) * args.world_size)

    return top1_meter.avg, top5_meter.avg


if __name__ == '__main__':
    main()
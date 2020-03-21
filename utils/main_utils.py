import os, sys
from collections import deque
import shutil
import torch
import numpy as np
import torch.distributed as dist
import torch.utils.data as data
import datetime


def save_checkpoint(state, is_best, model_dir='.', filename=None):
    if filename is None:
        filename = '{}/checkpoint.pth.tar'.format(model_dir)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '{}/model_best.pth.tar'.format(model_dir))


class Logger(object):
    def __init__(self, quiet=False, log_fn=None, rank=0, prefix=""):
        self.rank = rank if rank is not None else 0
        self.quiet = quiet
        self.log_fn = log_fn

        self.prefix = ""
        if prefix:
            self.prefix = prefix + ' | '

        self.file_pointers = []
        if self.rank == 0:
            if self.quiet:
                open(log_fn, 'w').close()

    def add_line(self, content):
        if self.rank == 0:
            msg = self.prefix+content
            if self.quiet:
                fp = open(self.log_fn, 'a')
                fp.write(msg+'\n')
                fp.flush()
                fp.close()
            else:
                print(msg)
                sys.stdout.flush()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', window_size=0):
        self.name = name
        self.fmt = fmt
        self.window_size = window_size
        self.reset()

    def reset(self):
        if self.window_size > 0:
            self.q = deque(maxlen=self.window_size)
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        if self.window_size > 0:
            self.q.append((val, n))
            self.count = sum([n for v, n in self.q])
            self.sum = sum([v * n for v, n in self.q])
        else:
            self.sum += val * n
            self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, phase, epoch=None, logger=None, tb_writter=None):
        self.batches_per_epoch = num_batches
        self.batch_fmtstr = self._get_batch_fmtstr(epoch, num_batches)
        self.meters = meters
        self.phase = phase
        self.epoch = epoch
        self.logger = logger
        self.tb_writter = tb_writter

    def display(self, batch):
        step = self.epoch * self.batches_per_epoch + batch
        date = str(datetime.datetime.now())
        entries = ['{} | {} {}'.format(date, self.phase, self.batch_fmtstr.format(batch))]
        entries += [str(meter) for meter in self.meters]
        if self.logger is None:
            print('\t'.join(entries))
        else:
            self.logger.add_line('\t'.join(entries))

        if self.tb_writter is not None:
            for meter in self.meters:
                self.tb_writter.add_scalar('{}/Batch-{}'.format(self.phase, meter.name), meter.val, step)

    def _get_batch_fmtstr(self, epoch, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        epoch_str = '[{}]'.format(epoch) if epoch is not None else ''
        return epoch_str+'[' + fmt + '/' + fmt.format(num_batches) + ']'

def prep_environment(args, cfg):
    from torch.utils.tensorboard import SummaryWriter

    # Prepare loggers (must be configured after initialize_distributed_backend())
    model_dir = '{}/{}'.format(cfg['model']['model_dir'], cfg['model']['name'])
    if args.rank == 0:
        prep_output_folder(model_dir, False)
    log_fn = '{}/train.log'.format(model_dir)
    logger = Logger(quiet=args.quiet, log_fn=log_fn, rank=args.rank)

    logger.add_line(str(datetime.datetime.now()))
    if any(['SLURM' in env for env in list(os.environ.keys())]):
        logger.add_line("=" * 30 + "   SLURM   " + "=" * 30)
        for env in os.environ.keys():
            if 'SLURM' in env:
                logger.add_line('{:30}: {}'.format(env, os.environ[env]))

    logger.add_line("=" * 30 + "   Config   " + "=" * 30)
    def print_dict(d, ident=''):
        for k in d:
            if isinstance(d[k], dict):
                logger.add_line("{}{}".format(ident, k))
                print_dict(d[k], ident='  '+ident)
            else:
                logger.add_line("{}{}: {}".format(ident, k, str(d[k])))
    print_dict(cfg)

    logger.add_line("=" * 30 + "   Args   " + "=" * 30)
    for k in args.__dict__:
        logger.add_line('{:30} {}'.format(k, args.__dict__[k]))

    tb_writter = None
    if cfg['log2tb'] and args.rank == 0 and not cfg['test_only']:
        tb_dir = '{}/tensorboard'.format(model_dir)
        os.system('mkdir -p {}'.format(tb_dir))
        tb_writter = SummaryWriter(tb_dir)

    return logger, tb_writter, model_dir


def build_optimizer(params, cfg, logger=None):
    if cfg['name'] == 'sgd':
        optimizer = torch.optim.SGD(
            params=params,
            lr=cfg['lr']['base_lr'],
            momentum=cfg['momentum'],
            weight_decay=cfg['weight_decay'],
            nesterov=cfg['nesterov']
        )

    elif cfg['name'] == 'adam':
        optimizer = torch.optim.Adam(
            params=params,
            lr=cfg['lr']['base_lr'],
            weight_decay=cfg['weight_decay'],
            betas=cfg['betas'] if 'betas' in cfg else [0.9, 0.999]
        )

    else:
        raise ValueError('Unknown optimizer.')

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['lr']['milestones'], gamma=cfg['lr']['gamma'])
    return optimizer, scheduler


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def prep_output_folder(model_dir, evaluate):
    if evaluate:
        assert os.path.isdir(model_dir)
    else:
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)

def parameter_description(model):
    desc = ''
    for n, p in model.named_parameters():
        desc += "{:70} | {:10} | {:30} | {}\n".format(
            n, 'Trainable' if p.requires_grad else 'Frozen',
            ' x '.join([str(s) for s in p.size()]), str(np.prod(p.size())))
    return desc


def initialize_distributed_backend(args, ngpus_per_node):
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + args.gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    if args.rank == -1:
        args.rank = 0
    return args


def distribute_model_to_cuda(models, args, batch_size, num_workers, ngpus_per_node):
    squeeze = False
    if not isinstance(models, list):
        models = [models]
        squeeze = True

    for i in range(len(models)):
        if args.distributed:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                models[i].cuda(args.gpu)
                models[i] = torch.nn.parallel.DistributedDataParallel(models[i], device_ids=[args.gpu])
            else:
                models[i].cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                models[i] = torch.nn.parallel.DistributedDataParallel(models[i])
        elif args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            models[i] = models[i].cuda(args.gpu)
        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            models[i] = torch.nn.DataParallel(models[i]).cuda()

    if squeeze:
        models = models[0]

    if args.distributed and args.gpu is not None:
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        batch_size = int(batch_size / ngpus_per_node)
        num_workers = int((num_workers + ngpus_per_node - 1) / ngpus_per_node)

    return models, args, batch_size, num_workers


def synchronize_meters(progress, cur_gpu):
    metrics = torch.tensor([m.avg for m in progress.meters]).cuda(cur_gpu)
    metrics_gather = [torch.ones_like(metrics) for _ in range(dist.get_world_size())]
    dist.all_gather(metrics_gather, metrics)

    metrics = torch.stack(metrics_gather).mean(0).cpu().numpy()
    for meter, m in zip(progress.meters, metrics):
        meter.avg = m


def build_criterion(cfg, logger=None):
    import criterions
    criterion = criterions.__dict__[cfg['name']](**cfg['args'])
    if logger is not None:
        logger.add_line(str(criterion))

    return criterion


def build_dataloader(db_cfg, split_cfg, num_workers, distributed):
    import torch.utils.data as data
    import torch.utils.data.distributed
    from datasets import preprocessing
    import datasets

    # Video transforms
    num_frames = int(db_cfg['video_clip_duration'] * db_cfg['video_fps'])
    if db_cfg['transforms'] == 'crop+color':
        video_transform = preprocessing.VideoPrep_Crop_CJ(
            resize=db_cfg['frame_size'],
            crop=(db_cfg['crop_size'], db_cfg['crop_size']),
            augment=split_cfg['use_augmentation'],
            num_frames=num_frames,
            pad_missing=True,
        )

    elif db_cfg['transforms'] == 'msc+color':
        video_transform = preprocessing.VideoPrep_MSC_CJ(
            crop=(db_cfg['crop_size'], db_cfg['crop_size']),
            augment=split_cfg['use_augmentation'],
            num_frames=num_frames,
            pad_missing=True,
        )

    else:
        raise ValueError('Unknown transform')

    # Audio transforms
    audio_transforms = [
        preprocessing.AudioPrepLibrosa(
            trim_pad=True,
            duration=db_cfg['audio_clip_duration'],
            augment=split_cfg['use_augmentation'],
            missing_as_zero=True),
        preprocessing.LogSpectrogram(
            db_cfg['audio_fps'],
            n_fft=db_cfg['n_fft'],
            hop_size=1. / db_cfg['spectrogram_fps'],
            normalize=True,
            spect_db=True)
    ]
    audio_fps_out = db_cfg['spectrogram_fps']
    audio_shape = (1, int(audio_fps_out * db_cfg['audio_clip_duration']), db_cfg['n_fft'] // 2 + 1)

    if db_cfg['name'] == 'audioset':
        dataset = datasets.AudioSet
    elif db_cfg['name'] == 'kinetics':
        dataset = datasets.Kinetics
    else:
        raise ValueError('Unknown dataset')

    clips_per_video = split_cfg['clips_per_video'] if 'clips_per_video' in split_cfg else 1
    db = dataset(
        subset=split_cfg['split'],
        full_res=db_cfg['full_res'],
        return_video=True,
        video_clip_duration=db_cfg['video_clip_duration'],
        video_fps=db_cfg['video_fps'],
        video_fps_out=db_cfg['video_fps'],
        video_shape=(3, num_frames, db_cfg['crop_size'], db_cfg['crop_size']),
        video_transform=video_transform,
        return_audio=True,
        audio_clip_duration=db_cfg['audio_clip_duration'],
        audio_fps=db_cfg['audio_fps'],
        audio_fps_out=audio_fps_out,
        audio_shape=audio_shape,
        audio_transform=audio_transforms,
        max_offsync_augm=0.5 if split_cfg['use_augmentation'] else 0,
        return_labels=False,
        return_index=True,
        missing_audio_as_zero=False,
        mode='clip',
        clips_per_video=clips_per_video,
    )

    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(db)
    else:
        sampler = None

    loader = torch.utils.data.DataLoader(
        db,
        batch_size=db_cfg['batch_size'],
        shuffle=(sampler is None),
        drop_last=split_cfg['drop_last'],
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler)

    return loader


def build_dataloaders(cfg, num_workers, distributed, logger):
    train_loader = build_dataloader(cfg, cfg['train'], num_workers, distributed)
    logger.add_line("\n"+"="*30+"   Train data   "+"="*30)
    logger.add_line(str(train_loader.dataset))
    return train_loader


class CheckpointManager(object):
    def __init__(self, checkpoint_dir, rank=0):
        self.checkpoint_dir = checkpoint_dir
        self.rank = rank
        self.best_metric = 0.

    def save(self, epoch, filename=None, eval_metric=0., **kwargs):
        if self.rank != 0:
            return

        is_best = False
        if eval_metric > self.best_metric:
            self.best_metric = eval_metric
            is_best = True

        state = {'epoch': epoch}
        for k in kwargs:
            state[k] = kwargs[k].state_dict()

        if filename is None:
            save_checkpoint(state=state, is_best=is_best, model_dir=self.checkpoint_dir)
        else:
            save_checkpoint(state=state, is_best=False, filename='{}/{}'.format(self.checkpoint_dir, filename))

    def last_checkpoint_fn(self):
        return '{}/checkpoint.pth.tar'.format(self.checkpoint_dir)

    def best_checkpoint_fn(self):
        return '{}/model_best.pth.tar'.format(self.checkpoint_dir)

    def checkpoint_fn(self, last=False, best=False):
        assert best or last
        assert not (last and best)
        if last:
            return self.last_checkpoint_fn()
        if best:
            return self.best_checkpoint_fn()

    def checkpoint_exists(self, last=False, best=False):
        return os.path.isfile(self.checkpoint_fn(last, best))

    def restore(self, fn=None, restore_last=False, restore_best=False, **kwargs):
        checkpoint_fn = fn if fn is not None else self.checkpoint_fn(restore_last, restore_best)
        ckp = torch.load(checkpoint_fn, map_location={'cuda:0': 'cpu'})
        start_epoch = ckp['epoch']
        for k in kwargs:
            if k == 'train_criterion':
                kwargs[k].load_state_dict(ckp[k], strict=False)
            else:
                kwargs[k].load_state_dict(ckp[k])
        return start_epoch


def build_model(cfg, logger=None):
    import models
    assert cfg['arch'] in models.__dict__, 'Unknown model architecture'
    model = models.__dict__[cfg['arch']](**cfg['args'])

    if logger is not None:
        if isinstance(model, (list, tuple)):
            logger.add_line("=" * 30 + "   Model   " + "=" * 30)
            for m in model:
                logger.add_line(str(m))
            logger.add_line("=" * 30 + "   Parameters   " + "=" * 30)
            for m in model:
                logger.add_line(parameter_description(m))
        else:
            logger.add_line("=" * 30 + "   Model   " + "=" * 30)
            logger.add_line(str(model))
            logger.add_line("=" * 30 + "   Parameters   " + "=" * 30)
            logger.add_line(parameter_description(model))

    return model


def _gather_from_all(tensor):
    gathered_tensor = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_tensor, tensor)
    gathered_tensor = torch.cat(gathered_tensor, 0)
    return gathered_tensor


def _to_one_hot(y, n_dims, dtype=torch.cuda.FloatTensor):
    scatter_dim = len(y.size())
    y_tensor = y.type(torch.cuda.LongTensor).view(*y.size(), -1)
    zeros = torch.zeros(*y.size(), n_dims).type(dtype)
    return zeros.scatter(scatter_dim, y_tensor, 1)


class AliasMethod(object):
    """
    From: https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    """
    def __init__(self, probs):

        if probs.sum() > 1:
            probs.div_(probs.sum())
        K = len(probs)
        self.prob = torch.zeros(K)
        self.alias = torch.LongTensor([0]*K)

        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            self.prob[kk] = K*prob
            if self.prob[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self.alias[small] = large
            self.prob[large] = (self.prob[large] - 1.0) + self.prob[small]

            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last_one in smaller+larger:
            self.prob[last_one] = 1

    def to(self, device):
        self.prob = self.prob.to(device)
        self.alias = self.alias.to(device)

    def draw(self, N):
        """
        Draw N samples from multinomial
        :param N: number of samples
        :return: samples
        """
        K = self.alias.size(0)

        kk = torch.zeros(N, dtype=torch.long, device=self.prob.device).random_(0, K)
        prob = self.prob.index_select(0, kk)
        alias = self.alias.index_select(0, kk)
        # b is whether a random number is greater than q
        b = torch.bernoulli(prob)
        oq = kk.mul(b.long())
        oj = alias.mul((1-b).long())

        return oq + oj

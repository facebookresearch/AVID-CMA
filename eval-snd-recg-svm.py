import yaml
import argparse
import torch
from torch import nn
from torch.utils import data
import models
from datasets import preprocessing
from datasets.esc50_dataset import ESC50
from utils import main_utils, eval_utils


parser = argparse.ArgumentParser(description='Evaluation on ESC Sound Classification')
parser.add_argument('cfg', metavar='CFG', help='config file')
parser.add_argument('model_cfg', metavar='CFG', help='config file')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--quiet', action='store_true')


def main():
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.cfg))
    if args.debug:
        cfg['num_workers'] = 1
        cfg['dataset']['train']['clips_per_video'] = 1
        cfg['dataset']['test_dense']['clips_per_video'] = 1

    for fold in range(1, cfg['dataset']['n_folds']+1):
        main_worker(fold, args, cfg)


def extract_features(model, loader, feat_names, pooling_ops=None, dense=False, logger=None):
    import numpy as np
    from collections import defaultdict

    model = model.train(False)
    with torch.no_grad():
        feats, labels = defaultdict(list), []
        pooling_ops = {feat_names[i]: eval('nn.'+pooling_ops[i]) if pooling_ops[i] is not None else None for i in range(len(feat_names))}
        for it, (audio, target) in enumerate(loader):
            if dense:
                batch_size, clips_per_sample = audio.shape[0], audio.shape[1]
                audio = audio.flatten(0, 1)
            embs = model(audio.cuda(), return_embs=True)
            for k in embs:
                if k not in feat_names:
                    continue
                if pooling_ops[k] is not None:
                    embs[k] = pooling_ops[k](embs[k])
                if dense:
                    embs[k] = embs[k].view(batch_size, clips_per_sample, -1)
                else:
                    embs[k] = embs[k].view(audio.shape[0], -1)
                feats[k] += [embs[k].cpu().numpy()]
            if len(feats) != len(feat_names):
                print(feat_names)
                print(embs.keys())
                raise ValueError
            labels += [target.numpy()]
            if it % 20 == 0:
                logger.add_line('Extracting features: {}/{}'.format(it, len(loader)))
    return {k: np.concatenate(feats[k], 0) for k in feat_names}, np.concatenate(labels, 0)


def main_worker(fold, args, cfg):
    # Prepare folder and logger
    args.distributed = False
    args.gpu = 0
    eval_dir, model_cfg, logger = eval_utils.prepare_environment(args, cfg, fold)

    # Model
    model = build_model(model_cfg, logger).cuda()

    # Extract features
    train_loader, test_loader, dense_loader = build_dataloaders(cfg['dataset'], fold, cfg['num_workers'])
    Xtr, Ytr = extract_features(model, train_loader, cfg['model']['feat_names'], pooling_ops=cfg['model']['pooling_ops'], logger=logger)
    Xte, Yte = extract_features(model, dense_loader, cfg['model']['feat_names'], pooling_ops=cfg['model']['pooling_ops'], dense=True, logger=logger)

    from sklearn.svm import LinearSVC, SVC
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import StandardScaler

    acc = {}
    for k in cfg['model']['feat_names']:
        clf = OneVsRestClassifier(LinearSVC(C=cfg['model']['C'], loss='squared_hinge', intercept_scaling=1.0, random_state=0, tol=0.0001, max_iter=200, verbose=1), n_jobs=-1)
        # clf = OneVsRestClassifier(SVC(kernel='linear', C=cfg['model']['C'], random_state=0, tol=0.0001, max_iter=2000), n_jobs=-1)
        scaler = StandardScaler()

        logger.add_line('\n'+'='*60+'\nFitting SVM to {}'.format(k))
        scaler.fit(Xtr[k])

        clf.fit(scaler.transform(Xtr[k]), Ytr)

        logger.add_line('Evaluating'.format(k))
        bs, nt, nf = Xte[k].shape
        Ste = clf.decision_function(scaler.transform(Xte[k].reshape(bs*nt, nf))).reshape(bs, nt, cfg['model']['n_classes'])
        Pte = Ste.mean(1).argmax(1)
        acc[k] = (Yte == Pte).mean()*100
        logger.add_line('{}: {}'.format(k, acc[k]))

    for k in cfg['model']['feat_names']:
        logger.add_line('Final {}: {}'.format(k, acc[k]))


def build_model(feat_cfg, logger):
    model = models.__dict__[feat_cfg['arch']](**feat_cfg['args'])

    ckp_fn = '{}/{}/checkpoint.pth.tar'.format(feat_cfg['model_dir'], feat_cfg['name'])
    ckp = torch.load(ckp_fn, map_location={'cuda:0': 'cpu'})
    try:
        model.load_state_dict({k.replace('module.', ''): ckp['state_dict'][k] for k in ckp['state_dict']})
    except KeyError:
        model.load_state_dict({k.replace('module.', ''): ckp['model'][k] for k in ckp['model']})
    del ckp

    model = model.audio_model

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    logger.add_line("=" * 30 + "   Model   " + "=" * 30)
    logger.add_line(str(model))
    logger.add_line("=" * 30 + "   Parameters   " + "=" * 30)
    logger.add_line(main_utils.parameter_description(model))

    return model


def build_dataloader(db_cfg, split_cfg, fold, num_workers):
    audio_transforms = [
        preprocessing.AudioPrepLibrosa(
            trim_pad=True,
            duration=db_cfg['clip_duration'],
            augment=split_cfg['use_augmentation'],
            missing_as_zero=True),
        preprocessing.SpectrogramLibrosa2(
            db_cfg['audio_fps'],
            n_fft=db_cfg['n_fft'],
            hop_size=1. / db_cfg['spectrogram_fps'],
            normalize=True,
            spect_db=True)
    ]

    if db_cfg['name'] == 'esc50':
        db = ESC50(
            split_cfg['split'].format(fold=fold),
            rate=db_cfg['audio_fps'],
            clip_duration=db_cfg['clip_duration'],
            transform=audio_transforms,
            target_transform=None,
            mode=split_cfg['mode'],
            clips_per_sample=split_cfg['clips_per_video'],
            augment=split_cfg['use_augmentation'],
        )

    elif db_cfg['name'] == 'dcase':
        from datasets.dcase_dataset import DCASE
        db = DCASE(
            split_cfg['split'],
            rate=db_cfg['audio_fps'],
            clip_duration=db_cfg['clip_duration'],
            transform=audio_transforms,
            target_transform=None,
            mode=split_cfg['mode'],
            clips_per_sample=split_cfg['clips_per_video'],
            augment=split_cfg['use_augmentation'],
        )

    else:
        raise ValueError('Unknown dataset')

    loader = data.DataLoader(
        db,
        batch_size=db_cfg['batch_size'] if split_cfg['mode'] == 'clip' else max(1, db_cfg['batch_size']//split_cfg['clips_per_video']),
        num_workers=num_workers,
        pin_memory=True,
        shuffle=split_cfg['use_shuffle'],
        drop_last=split_cfg['drop_last']
    )
    return loader


def build_dataloaders(cfg, fold, num_workers):
    train_loader = build_dataloader(cfg, cfg['train'], fold, num_workers)
    test_loader = build_dataloader(cfg, cfg['test'], fold, num_workers)
    dense_loader = build_dataloader(cfg, cfg['test_dense'], fold, num_workers)
    return train_loader, test_loader, dense_loader


if __name__ == '__main__':
    main()

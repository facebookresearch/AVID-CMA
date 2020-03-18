import os
import json
import numpy as np
import torch.utils.data as data

from datasets.video_dataset import VideoDataset
from utils.ioutils import av_wrappers

ROOT = '/datasets01_101/ucf101/112018/'
ANNO_DIR = '/datasets01_101/ucf101/112018/ucfTrainTestlist/'
CACHE_DIR = '/checkpoint/pmorgado/data_cache/ucf101'


def get_metadata():
    classes_fn = ROOT+'/ucfTrainTestlist/classInd.txt'
    classes = [l.strip().split()[1] for l in open(classes_fn)]

    cache_fn = CACHE_DIR+'/meta.json'
    if os.path.isfile(cache_fn):
        ucf_meta = json.load(open(cache_fn))
        return ucf_meta, classes

    all_files, all_labels = [], []
    for list_fn in ['trainlist01.txt', 'testlist01.txt']:
        for ln in open('{}/{}'.format(ANNO_DIR, list_fn)):
            fn = ln.strip().split()[0]
            lbl = classes.index(fn.split('/')[0])
            path = '{}/data/{}'.format(ROOT, fn.split('/')[-1])
            all_files.append(path)
            all_labels.append(lbl)

    ucf_meta = []
    for path, lbl in zip(all_files, all_labels):
        video_meta, audio_meta = av_wrappers.av_meta(path, video=0, audio=0)
        if audio_meta is None:
            audio_meta = {'start_time': -1, 'duration': -1}

        ucf_meta.append({
            'fn': path,
            'label': lbl,
            'label_str': classes[lbl],
            'video_start': float(video_meta['start_time']),
            'video_duration': float(video_meta['duration']),
            'audio_start': float(audio_meta['start_time']),
            'audio_duration': float(audio_meta['duration']),
        })
    json.dump(ucf_meta, open(cache_fn, 'w'))
    return ucf_meta, classes


def filter_samples(meta, subset=None, ignore_video_only=False, min_clip_duration=None):
    # Filter by subset
    if subset is not None:
        subset = set([ln.strip().split()[0] for ln in open('{}/{}.txt'.format(ANNO_DIR, subset))])
        meta = [m for m in meta if '{}/{}'.format(m['label_str'], m['fn'].split('/')[-1]) in subset]

    # Filter videos with no audio
    if ignore_video_only:
        meta = [m for m in meta if m['audio_start'] != -1 or m['audio_duration'] != -1]

    # Filter videos with less than clip_length duration
    if min_clip_duration is not None:
        for m in meta:
            video_st = m['video_start']
            video_ft = m['video_start'] + m['video_duration']
            if m['audio_start'] == -1 and m['audio_duration'] == -1:
                m['audio_start'], m['audio_duration'] = m['video_start'], m['video_duration']
            audio_st = m['audio_start']
            audio_ft = m['audio_start'] + m['audio_duration']
            m['sample_duration'] = min(audio_ft, video_ft) - max(audio_st, video_st)
        meta = [m for m in meta if m['sample_duration'] > min_clip_duration]
    return meta


class UCF(VideoDataset):
    def __init__(self, subset,
                 full_res=True,
                 video_clip_duration=0.5,
                 return_video=True,
                 video_fps=16.,
                 video_shape=(3, 8, 224, 224),
                 video_fps_out=25,
                 video_transform=None,
                 return_audio=False,
                 audio_clip_duration=2.,
                 audio_fps=22050,
                 audio_shape=(13, 64, 257),
                 audio_fps_out=64,
                 audio_transform=None,
                 return_labels=False,
                 missing_audio_as_zero=False,
                 max_offsync_augm=0,
                 time_scale_max_ratio=1,
                 mode='clip',
                 clips_per_video=20,
                 ):

        self.name = 'UCF-101'
        self.root = ROOT
        self.subset = subset

        meta, classes = get_metadata()
        meta = filter_samples(meta, subset=subset, ignore_video_only=False)
        video_fns = [m['fn'].split('/')[-1] for m in meta]
        audio_fns = [m['fn'].split('/')[-1] for m in meta]
        labels = [m['label'] for m in meta]
        sample_time_lims = [[m['video_start'], m['video_start'] + m['video_duration'] - 1/25.,
                             m['audio_start'], m['audio_start'] + m['audio_duration'] - 1/25.] for m in meta]
        self.classes = classes
        self.num_classes = len(self.classes)
        self.num_videos = len(meta)

        super(UCF, self).__init__(
            video_clip_duration=video_clip_duration,
            return_video=return_video,
            video_root='{}/data'.format(ROOT),
            video_fns=video_fns,
            video_fps=video_fps,
            video_fps_out=video_fps_out,
            video_shape=video_shape,
            video_transform=video_transform,
            return_audio=return_audio,
            audio_clip_duration=audio_clip_duration,
            audio_fns=audio_fns,
            audio_fps=audio_fps,
            audio_fps_out=audio_fps_out,
            audio_shape=audio_shape,
            audio_transform=audio_transform,
            return_labels=return_labels,
            labels=labels,
            max_offsync_augm=max_offsync_augm,
            time_scale_max_ratio=time_scale_max_ratio,
            time_lims=sample_time_lims,
            mode=mode,
            clips_per_video=clips_per_video,
            missing_audio_as_zero=missing_audio_as_zero,
        )


def benchmark(batch_size, num_workers):
    import time
    from datasets import preprocessing

    CLIP_DURATION = 1.
    VIDEO_FPS = 16.
    AUDIO_FPS = 48000
    SPECTROGRAM_FPS = 64.
    FRAME_SIZE = 256
    CROP_SIZE = 224

    video_transform = preprocessing.VideoPreprocessing(resize=FRAME_SIZE, crop=(CROP_SIZE, CROP_SIZE), augment=True)
    audio_transform = preprocessing.AudioPrepLibrosa(duration=CLIP_DURATION, missing_as_zero=True, augment=True)
    dataset = UCF('trainlist01',
                clip_duration=CLIP_DURATION,
                return_video=True,
                video_fps=VIDEO_FPS,
                video_transform=video_transform,
                return_audio=False,
                audio_fps=AUDIO_FPS,
                audio_transform=audio_transform,
                return_labels=True,
                missing_audio_as_zero=True)

    print(dataset)

    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True)

    tt = time.time()
    read_times = []
    frames_per_clip = int(VIDEO_FPS * CLIP_DURATION)
    for idx, batch in enumerate(loader):
        read_times.append(time.time() - tt)
        tt = time.time()

        secs_per_clip = np.mean(read_times[1:]) / batch_size
        print('Iter {:03d} | Secs per batch {:.3f} | Clips per sec {:.3f} | Frames per sec  {:.3f}'.format(
            idx, secs_per_clip * batch_size, 1. / secs_per_clip, frames_per_clip / secs_per_clip
        ))
        if idx > 100:
            break

    secs_per_clip = np.mean(read_times[1:]) / batch_size

    print('')
    print('Num workers     | {}'.format(num_workers))
    print('Batch size      | {}'.format(batch_size))
    print('Frames per clip | {}'.format(frames_per_clip))
    print('Secs per batch  | {:.3f}'.format(secs_per_clip * batch_size))
    print('Clips per sec   | {:.3f}'.format(1. / secs_per_clip))
    print('Frames per sec  | {:.3f}'.format(frames_per_clip / secs_per_clip))


if __name__ == '__main__':
    for w in [1]:
        for bs in [62]:
            print('=' * 60)
            benchmark(batch_size=bs, num_workers=w)
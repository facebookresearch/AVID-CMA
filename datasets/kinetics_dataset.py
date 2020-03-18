import sys
sys.path.insert(0, '.')
import os
import json
import csv
import torch
import numpy as np
import torch.utils.data as data
import multiprocessing as mp
from utils.ioutils import av_wrappers

from datasets.video_dataset import VideoDataset


ROOT = '/datasets01_101/kinetics/070618/400/'
ANNO_DIR = '/datasets01_101/kinetics/070618/400/list'
CACHE_DIR = '/checkpoint/pmorgado/data_cache/kinetics_400'

AUDIO_VERSION = ''
AUDIO_EXT = 'mp4'
AUDIO_FORMAT = 'mp4'

# VIDEO_VERSION = '_avi-288p'
# VIDEO_EXT = 'avi'
# VIDEO_FORMAT = 'avi'
VIDEO_VERSION = ''
VIDEO_EXT = 'mp4'
VIDEO_FORMAT = 'mp4'


class KineticsClasses:
    def __init__(self):
        import csv
        self.classes = sorted(list(set([d['label'] for d in csv.DictReader(open(ANNO_DIR + '/kinetics-400_val.csv'))])))
        self.classes = [cls.replace(' ', '_') for cls in self.classes]
        self.class_label = {cls: idx for idx, cls in enumerate(self.classes)}

    def __getitem__(self, index):
        return self.classes[index]

    def __len__(self):
        return len(self.classes)

    def class2index(self, class_string):
        return self.class_label[class_string]


def get_video_meta(m, classes):
    cls = m['label'].replace(' ', '_')
    label = classes.class2index(cls)
    vid = m['youtube_id']
    segment = (int(m['time_start']), int(m['time_end']))
    video_fn = '{}/{}{}/{}/{}_{:06d}_{:06d}.{}'.format(ROOT, m['split'], VIDEO_VERSION, cls, vid, segment[0], segment[1], VIDEO_EXT)
    audio_fn = '{}/{}{}/{}/{}_{:06d}_{:06d}.{}'.format(ROOT, m['split'], AUDIO_VERSION, cls, vid, segment[0], segment[1], AUDIO_EXT)
    if not os.path.isfile(video_fn) or not os.path.isfile(audio_fn):
        return None

    video_meta, _ = av_wrappers.av_meta(video_fn, video=0, format=VIDEO_FORMAT)
    _, audio_meta = av_wrappers.av_meta(audio_fn, audio=0, format=AUDIO_FORMAT)
    if video_meta is None or audio_meta is None:
        return None

    m['video_fn'] = video_fn
    m['audio_fn'] = audio_fn
    m['label'] = label
    m['video_meta'] = {
        'fps': float(video_meta['fps']),
        'start_time': float(video_meta['start_time']),
        'duration': float(video_meta['duration']),
        'width': video_meta['size'][0],
        'height': video_meta['size'][1],
    }
    m['audio_meta'] = {
        'fps': float(audio_meta['fps']),
        'start_time': float(audio_meta['start_time']),
        'duration': float(audio_meta['duration']),
        'channels': audio_meta['channels'],
    }
    return m


def get_metadata(subset):
    classes = KineticsClasses()
    cache_fn = '{}/{}{}-meta.json'.format(CACHE_DIR, subset, VIDEO_VERSION)
    if os.path.isfile(cache_fn):
        db_meta = json.load(open(cache_fn))
        return db_meta, classes

    db_meta = list(csv.DictReader(open('{}/kinetics-400_{}.csv'.format(ANNO_DIR, subset))))

    NUM_WORKERS = 70
    q_in = mp.Queue()
    q_out = mp.Queue()

    def dispatch(q, db_meta):
        for ii, m in enumerate(db_meta):
            q.put((ii, m), block=True)
        for _ in range(NUM_WORKERS):
            q.put(None, block=True)

    def process(q_in, q_out, classes):
        while True:
            packet = q_in.get(block=True)
            if packet is None:
                break

            ii, m = packet
            if ii % 1000 == 0:
                print('{}: {}'.format(ii, m['youtube_id']))
            m = get_video_meta(m, classes)

            if m is not None:
                q_out.put(m, block=True)
        q_out.put(None, block=True)

    disp_proc = mp.Process(target=dispatch, args=(q_in, db_meta))
    disp_proc.start()

    workers = []
    for _ in range(NUM_WORKERS):
        w = mp.Process(target=process, args=(q_in, q_out, classes))
        w.start()
        workers.append(w)

    done = 0
    db_meta = []
    while True:
        m = q_out.get(block=True)
        if m is None:
            done += 1
        else:
            db_meta.append(m)
        if done == NUM_WORKERS:
            break

    for w in workers:
        w.join()
    disp_proc.join()

    json.dump(db_meta, open(cache_fn, 'w'))
    return db_meta, classes


def filter_samples(meta, ignore_stills=False, min_clip_duration=None):
    # Filter videos in ignore list
    if ignore_stills:
        ignore_fn = '{}/ignore.lst'.format(CACHE_DIR)
        if os.path.isfile(ignore_fn):
            ignore = set([l.strip() for l in open(ignore_fn)])
            meta = [m for m in meta if m['YTID'] not in ignore]

    # Filter videos with less than clip_length duration
    if min_clip_duration is not None:
        for m in meta:
            audio_st = m['audio_meta']['start_time']
            audio_ft = audio_st + m['audio_meta']['duration']
            video_st = m['video_meta']['start_time']
            video_ft = video_st + m['video_meta']['duration']
            m['video_duration'] = min(audio_ft, video_ft) - max(audio_st, video_st)
        meta = [m for m in meta if m['video_duration'] > min_clip_duration]
    return meta


class Kinetics(VideoDataset):
    def __init__(self, subset,
                 return_video=True,
                 video_clip_duration=1.,
                 video_fps=25.,
                 video_shape=(3, 8, 224, 224),
                 video_fps_out=25,
                 video_transform=None,
                 return_audio=False,
                 audio_clip_duration=1.,
                 audio_fps=None,
                 audio_shape=(13, 64, 257),
                 audio_fps_out=64,
                 audio_transform=None,
                 return_labels=False,
                 return_index=False,
                 missing_audio_as_zero=False,
                 max_offsync_augm=0,
                 time_scale_max_ratio=1,
                 mode='clip',
                 clips_per_video=1,
                 full_res=False
                 ):

        global AUDIO_VERSION, VIDEO_EXT, VIDEO_VERSION, VIDEO_FORMAT, ROOT
        if full_res:
            ROOT = '/checkpoint/pmorgado/kinetics/070618/400/'
            VIDEO_VERSION = '-gop32'
            AUDIO_VERSION = '-gop32'
            VIDEO_EXT = 'mp4'
            VIDEO_FORMAT = 'mp4'
        else:
            ROOT = '/datasets01_101/kinetics/070618/400/'
            VIDEO_VERSION = '_avi-288p'
            VIDEO_EXT = 'avi'
            VIDEO_FORMAT = 'avi'

        meta, classes = get_metadata(subset)
        meta = filter_samples(meta, ignore_stills=False, min_clip_duration=max(video_clip_duration, audio_clip_duration)*1.5)

        time_lims = np.array([
            [m['video_meta']['start_time'], m['video_meta']['start_time'] + m['video_meta']['duration'],
             m['audio_meta']['start_time'], m['audio_meta']['start_time'] + m['audio_meta']['duration']]
            for m in meta])     # (Video SS, Video FF, Audio SS, Audio FF)

        video_fns = ['/'.join(m['video_fn'].split('/')[-3:]) for m in meta]
        audio_fns = ['/'.join(m['audio_fn'].split('/')[-3:]) for m in meta]
        labels = [m['label'] for m in meta]

        super(Kinetics, self).__init__(
            return_video=return_video,
            video_root=ROOT,
            video_fns=video_fns,
            video_clip_duration=video_clip_duration,
            video_fps=video_fps,
            video_shape=video_shape,
            video_fps_out=video_fps_out,
            video_transform=video_transform,
            return_audio=return_audio,
            audio_root=ROOT,
            audio_fns=audio_fns,
            audio_clip_duration=audio_clip_duration,
            audio_fps=audio_fps,
            audio_shape=audio_shape,
            audio_fps_out=audio_fps_out,
            audio_transform=audio_transform,
            return_labels=return_labels,
            labels=labels,
            return_index=return_index,
            mode=mode,
            clips_per_video=clips_per_video,
            max_offsync_augm=max_offsync_augm,
            time_scale_max_ratio=time_scale_max_ratio,
            time_lims=time_lims,
            missing_audio_as_zero=missing_audio_as_zero,
        )

        self.name = 'AudioSet dataset'
        self.root = ROOT
        self.subset = subset

        self.classes = classes
        self.num_videos = len(meta)
        self.num_classes = len(classes)

        self.sample_id = np.array([m['youtube_id'].encode('utf-8') for m in meta])


def benchmark(batch_size=16, num_workers=4):
    import time
    from datasets import preprocessing
    import yaml

    cfg = yaml.safe_load(open('configs/main/avts/avts-easy-kinetics-lr0.01.yaml'))
    db_cfg = cfg['dataset']
    split_cfg = db_cfg['test']

    video_transform = preprocessing.VideoPrep_MSC_CJ(
        crop=(db_cfg['crop_size'], db_cfg['crop_size']),
        augment=split_cfg['use_augmentation'],
        pad_missing=True,
        num_frames=int(db_cfg['video_fps'] * db_cfg['clip_duration']),
    )

    # Audio transforms
    audio_transforms = [
        preprocessing.AudioPrepLibrosa(
            trim_pad=True,
            duration=db_cfg['clip_duration'],
            augment=split_cfg['use_augmentation'],
            missing_as_zero=True)]

    audio_transforms += [
        preprocessing.MelSpectrogramLibrosa(
            n_mels=db_cfg['n_mels'],
            n_fft=db_cfg['n_fft'],
            hop_size=1. / db_cfg['spectrogram_fps'],
            normalize=db_cfg['normalize_fft'] if 'normalize_fft' in db_cfg else True,
            spect_db=True,
        )
    ]
    audio_fps_out = db_cfg['spectrogram_fps']
    audio_shape = (1, int(db_cfg['spectrogram_fps'] * db_cfg['clip_duration']), db_cfg['n_mels'])

    clips_per_video = split_cfg['clips_per_video'] if 'clips_per_video' in split_cfg else 1
    dataset = Kinetics(
        subset=split_cfg['split'],
        full_res=True,
        video_clip_duration=db_cfg['clip_duration'],
        return_video=True,
        video_fps=db_cfg['video_fps'],
        video_fps_out=db_cfg['video_fps'],
        video_shape=(3, int(db_cfg['video_fps'] * db_cfg['clip_duration']), db_cfg['crop_size'], db_cfg['crop_size']),
        video_transform=video_transform,
        return_audio=True,
        audio_fps=db_cfg['audio_fps'],
        audio_fps_out=audio_fps_out,
        audio_shape=audio_shape,
        audio_transform=audio_transforms,
        max_offsync_augm=0,
        return_labels=False,
        missing_audio_as_zero=False,
        time_scale_max_ratio=db_cfg['time_scale_max_ratio'] if 'time_scale_max_ratio' in db_cfg else 1,
        mode='clip',
        clips_per_video=clips_per_video,
    )

    print(dataset)

    loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers, 
            pin_memory=True)

    tt = time.time()
    read_times = []
    for idx, batch in enumerate(loader):
        if idx > 10:
            read_times.append(time.time()-tt)

            frames_per_clip = int(db_cfg['video_fps']*db_cfg['clip_duration'])
            secs_per_clip = np.mean(read_times) / batch_size
            print('Iter {:03d} | Secs per batch {:.3f} | Clips per sec {:.3f} | Frames per sec  {:.3f}'.format(
                idx, secs_per_clip * batch_size, 1. / secs_per_clip, frames_per_clip / secs_per_clip
            ))

        if idx > 100:
            break
        tt = time.time()

    frames_per_clip = int(db_cfg['video_fps']*db_cfg['clip_duration'])
    secs_per_clip = np.mean(read_times) / batch_size

    print('')
    print('Num workers     | {}'.format(num_workers))
    print('Batch size      | {}'.format(batch_size))
    print('Frames per clip | {}'.format(frames_per_clip))
    print('Secs per batch  | {:.3f}'.format(secs_per_clip * batch_size))
    print('Clips per sec   | {:.3f}'.format(1. / secs_per_clip))
    print('Frames per sec  | {:.3f}'.format(frames_per_clip / secs_per_clip))


if __name__ == '__main__':
    for w in [1]:
        for bs in [8]:
            print('='*60)
            benchmark(batch_size=bs, num_workers=w)

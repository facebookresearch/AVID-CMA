import sys
sys.path.insert(0, '.')
import os
import json
import csv
import numpy as np
import multiprocessing as mp
from utils.ioutils import av_wrappers
from datasets.video_dataset import VideoDataset

ANN_DIR = '/datasets01_101/audioset/042319/'
CACHE_DIR = '/checkpoint/pmorgado/data_cache/audioset'

AUDIO_ROOT = '/datasets01_101/audioset/042319/data'
AUDIO_VERSION = ''
AUDIO_EXT = 'flac'

# VIDEO_ROOT = '/checkpoint/pmorgado/audioset/042319/data'
VIDEO_ROOT = '/datasets01_101/audioset/042319/data'
VIDEO_VERSION = '_avi-288p'
VIDEO_EXT = 'avi'
# VIDEO_VERSION = ''
# VIDEO_EXT = 'mp4'

def get_video_meta(m, classes):
    positive_labels = [classes.class2index(cls) for cls in m['positive_labels']]
    vid = m['YTID']
    segment = (int(m['start_seconds'] * 1000), int(m['end_seconds'] * 1000))
    video_fn = '{}/{}_segments/video{}/{}_{:d}_{:d}.{}'.format(
        VIDEO_ROOT, m['subset'], VIDEO_VERSION, vid, segment[0], segment[1], VIDEO_EXT)
    audio_fn = '{}/{}_segments/audio{}/{}_{:d}_{:d}.{}'.format(
        AUDIO_ROOT, m['subset'], AUDIO_VERSION, vid, segment[0], segment[1], AUDIO_EXT)
    if not os.path.isfile(video_fn) or not os.path.isfile(audio_fn):
        return None

    try:
        video_meta, _ = av_wrappers.av_meta(video_fn, video=0)
        _, audio_meta = av_wrappers.av_meta(audio_fn, audio=0)
    except Exception:
        return None

    if video_meta is None or audio_meta is None:
        return None

    m.update({
        'label': positive_labels,
        'video_start': float(video_meta['start_time']),
        'video_duration': float(video_meta['duration']),
        'audio_start': float(audio_meta['start_time']),
        'audio_duration': float(audio_meta['duration']),
    })
    return m


def load_metadata_from_cache(subset):
    cache_fn = '{}/{}-meta.json'.format(CACHE_DIR, subset)
    if os.path.isfile(cache_fn):
        db_meta = json.load(open(cache_fn))
        return db_meta
    else:
        return None


def save_metadata_to_cache(db_meta, subset):
    cache_fn = '{}/{}-meta.json'.format(CACHE_DIR, subset)
    json.dump(db_meta, open(cache_fn, 'w'))


def get_metadata(subset):
    classes = AudiosetClasses()
    db_meta = load_metadata_from_cache(subset)
    if db_meta is not None:
        return db_meta, classes

    f = open('{}/{}_segments.csv'.format(ANN_DIR, subset))
    db_meta = []
    for l in f:
        if l.startswith('#'):
            continue
        ann = l.strip().split(', ')
        db_meta.append(
            {'YTID': ann[0],
             'start_seconds': float(ann[1]),
             'end_seconds': float(ann[2]),
             'positive_labels': ann[3][1:-1].split(','),
             'subset': subset})

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
                print('{}: {}'.format(ii, m['YTID']))
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

    save_metadata_to_cache(db_meta, subset)
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
            audio_st = m['audio_start']
            audio_ft = m['audio_start'] + m['audio_duration']
            video_st = m['video_start']
            video_ft = m['video_start'] + m['video_duration']
            m['sample_duration'] = min(audio_ft, video_ft) - max(audio_st, video_st)
        meta = [m for m in meta if m['sample_duration'] > min_clip_duration]
    return meta


class AudiosetClasses:
    def __init__(self):
        ann_list = list(csv.DictReader(open(CACHE_DIR + '/class_labels_indices.csv')))
        self.classes = [ann['mid'] for ann in ann_list]
        self.class_label = {ann['mid']: int(ann['index']) for ann in ann_list}
        self.display_name = {ann['mid']: ann['display_name'] for ann in ann_list}

    def __getitem__(self, index):
        return self.display_name[self.classes[index]]

    def __len__(self):
        return len(self.classes)

    def class2index(self, class_string):
        return self.class_label[class_string]


class AudioSet(VideoDataset):
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
                 audio_shape=(1, 64, 257),
                 audio_fps_out=64,
                 audio_transform=None,
                 return_labels=False,
                 return_index=False,
                 missing_audio_as_zero=False,
                 max_offsync_augm=0,
                 mode='clip',
                 clips_per_video=1,
                 return_nneig=False,
                 nneigs_cutoff=10,
                 nneigs_fn=None,
                 return_signatures=False,
                 signatures_fns=None,
                 full_res=False
                 ):

        global VIDEO_VERSION, VIDEO_EXT, VIDEO_ROOT
        if full_res:
            VIDEO_ROOT = '/checkpoint/pmorgado/audioset/042319/data'
            VIDEO_VERSION = '-g32'
            VIDEO_EXT = 'mp4'
        else:
            VIDEO_VERSION = '_avi-288p'
            VIDEO_EXT = 'avi'

        meta, classes = get_metadata(subset)
        if subset == 'unbalanced_train':
            meta = filter_samples(meta, ignore_stills=False, min_clip_duration=max(video_clip_duration, audio_clip_duration) + 0.1)
        else:
            meta = filter_samples(meta, ignore_stills=True, min_clip_duration=max(video_clip_duration, audio_clip_duration) + 0.1)

        time_lims = np.array([
            [m['video_start'], m['video_start'] + m['video_duration'],
             m['audio_start'], m['audio_start'] + m['audio_duration']] for m in meta])
        video_root = '{}/{}_segments/video{}'.format(VIDEO_ROOT, subset.split('-')[0], VIDEO_VERSION)
        video_fns = ['{}_{:d}_{:d}.{}'.format(
            m['YTID'], int(m['start_seconds'] * 1000), int(m['end_seconds'] * 1000), VIDEO_EXT) for m in meta]
        audio_root = '{}/{}_segments/audio'.format(AUDIO_ROOT, subset.split('-')[0])
        audio_fns = ['{}_{:d}_{:d}.{}'.format(
            m['YTID'], int(m['start_seconds'] * 1000), int(m['end_seconds'] * 1000), AUDIO_EXT) for m in meta]
        I = np.eye(len(classes))
        labels = [(I[m['label']].sum(0) > 0).astype(int) for m in meta]

        super(AudioSet, self).__init__(
            return_video=return_video,
            video_clip_duration=video_clip_duration,
            video_root=video_root,
            video_fns=video_fns,
            video_fps=video_fps,
            video_fps_out=video_fps_out,
            video_shape=video_shape,
            video_transform=video_transform,
            return_audio=return_audio,
            audio_clip_duration=audio_clip_duration,
            audio_root=audio_root,
            audio_fns=audio_fns,
            audio_fps=audio_fps,
            audio_fps_out=audio_fps_out,
            audio_shape=audio_shape,
            audio_transform=audio_transform,
            return_labels=return_labels,
            labels=labels,
            return_index=return_index,
            max_offsync_augm=max_offsync_augm,
            time_lims=time_lims,
            mode=mode,
            clips_per_video=clips_per_video,
            missing_audio_as_zero=missing_audio_as_zero,
            return_nneig=return_nneig,
            nneigs_cutoff=nneigs_cutoff,
            nneigs_fn=nneigs_fn,
            return_signatures=return_signatures,
            signatures_fns=signatures_fns,
        )

        self.name = 'AudioSet dataset'
        self.root = VIDEO_ROOT
        self.subset = subset

        # self.classes = classes
        self.num_videos = len(meta)
        self.num_classes = len(classes)

        self.sample_id = np.array([m['YTID'].encode('utf-8') for m in meta])


def benchmark(batch_size=16, num_workers=4):
    import time
    import torch.utils.data as data
    from datasets import preprocessing
    import yaml

    cfg = yaml.safe_load(open('configs/main/mc-avc/av_vggish_small-l3d5-mc32b-100k.yaml'))
    db_cfg = cfg['dataset']
    split_cfg = db_cfg['train']

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
        preprocessing.SpectrogramLibrosa(
            n_fft=db_cfg['n_fft'],
            hop_size=1. / db_cfg['spectrogram_fps'],
            normalize=db_cfg['normalize_fft'] if 'normalize_fft' in db_cfg else True,
            spect_db=True,
        )
    ]
    audio_fps_out = db_cfg['spectrogram_fps']
    audio_shape = (1, int(db_cfg['spectrogram_fps'] * db_cfg['clip_duration']), 129)

    dataset = AudioSet(
        subset='unbalanced_train-100k',
        full_res=False,
        clip_duration=db_cfg['clip_duration'],
        return_video=True,
        video_fps=db_cfg['video_fps'],
        video_fps_out=db_cfg['video_fps'],
        video_shape=(3, int(db_cfg['video_fps']*db_cfg['clip_duration']), db_cfg['crop_size'], db_cfg['crop_size']),
        video_transform=video_transform,
        return_audio=False,
        audio_fps=db_cfg['audio_fps'],
        audio_fps_out=audio_fps_out,
        audio_shape=audio_shape,
        audio_transform=audio_transforms,
        max_offsync_augm=0,
        return_labels=False,
        missing_audio_as_zero=False,
        mode='clip',
        )
    print(dataset)

    loader = data.DataLoader(
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
            print('=' * 60)
            benchmark(batch_size=bs, num_workers=w)

import os
import json

from datasets.video_dataset import VideoDataset
from utils.ioutils import av_wrappers

ROOT = '/datasets01_101/hmdb51/112018/'
ANNO_DIR = '/datasets01_101/hmdb51/112018/splits/'
CACHE_DIR = 'datasets/cache/hmdb51'

def get_metadata():
    classes = [l.strip() for l in open('{}/classes.txt'.format(CACHE_DIR), 'r')]

    cache_fn = CACHE_DIR+'/meta.json'
    if os.path.isfile(cache_fn):
        hmdb_meta = json.load(open(cache_fn))
        return hmdb_meta, classes

    all_files, all_labels = [], []
    for cls in classes:
        for ln in open('{}/{}_test_split1.txt'.format(ANNO_DIR, cls)):
            fn = ln.strip().split()[0]
            lbl = classes.index(cls)
            path = '{}/{}'.format(cls, fn)
            all_files.append(path)
            all_labels.append(lbl)

    hmdb_meta = []
    for path, lbl in zip(all_files, all_labels):
        try:
            fn = '{}/data/{}'.format(ROOT, path)
            video_meta, audio_meta = av_wrappers.av_meta(fn, video=0, audio=0)
        except UnicodeDecodeError:
            continue
        if audio_meta is None:
            audio_meta = {'start_time': -1, 'duration': -1}

        hmdb_meta.append({
            'fn': path,
            'label': lbl,
            'label_str': classes[lbl],
            'video_start': float(video_meta['start_time']),
            'video_duration': float(video_meta['duration']),
            'audio_start': float(audio_meta['start_time']),
            'audio_duration': float(audio_meta['duration']),
        })
    json.dump(hmdb_meta, open(cache_fn, 'w'))
    return hmdb_meta, classes


def filter_samples(meta, classes, subset=None, min_clip_duration=None):
    # Filter by subset
    if subset is not None:
        subset, split = subset.split('-')
        selected_files = set()
        for cls in classes:
            for ln in open('{}/{}_test_{}.txt'.format(ANNO_DIR, cls, split)):
                fn, ss = ln.strip().split()
                fn = '{}/{}'.format(cls, fn)
                if subset == 'train' and ss == '1' or subset == 'test' and ss == '2':
                    selected_files.add(fn)
        meta = [m for m in meta if m['fn'] in selected_files]

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


class HMDB(VideoDataset):
    def __init__(self, subset,
                 full_res=True,
                 return_video=True,
                 video_clip_duration=1.,
                 video_fps=25.,
                 video_shape=(3, 8, 224, 224),
                 video_fps_out=25,
                 video_transform=None,
                 return_audio=False,
                 audio_clip_duration=1.,
                 audio_fps=22050,
                 audio_shape=(13, 64, 257),
                 audio_fps_out=64,
                 audio_transform=None,
                 return_labels=False,
                 missing_audio_as_zero=False,
                 max_offsync_augm=0,
                 mode='clip',
                 clips_per_video=20,
                 ):

        self.name = 'HMDB-101'
        self.root = ROOT
        self.subset = subset

        meta, classes = get_metadata()
        meta = filter_samples(meta, classes, subset=subset)
        video_fns = [m['fn'] for m in meta]
        audio_fns = [m['fn'] for m in meta]
        labels = [m['label'] for m in meta]
        sample_time_lims = [[m['video_start'], m['video_start'] + m['video_duration'] - 1/30.,
                             m['audio_start'], m['audio_start'] + m['audio_duration'] - 1/30.] for m in meta]
        self.classes = classes
        self.num_classes = len(self.classes)
        self.num_videos = len(meta)

        super(HMDB, self).__init__(
            return_video=return_video,
            video_clip_duration=video_clip_duration,
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
            time_lims=sample_time_lims,
            mode=mode,
            clips_per_video=clips_per_video,
            missing_audio_as_zero=missing_audio_as_zero,
        )

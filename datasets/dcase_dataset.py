import torch.utils.data as data
import csv
import numpy as np
import torch
from utils.ioutils import av_wrappers
import random

ROOT = '/checkpoint/pmorgado/dcase'


def wav_loader(filename, rate=None, start_time=0, duration=None):
    _, (data, sr) =  av_wrappers.av_loader(filename, audio_fps=rate, start_time=start_time, duration=duration, return_video=False, return_audio=True)
    return data, sr


class DCASE(data.Dataset):
    def __init__(self, subset, rate=48000, clip_duration=1., transform=None, target_transform=None, augment=False, mode='clip', clips_per_sample=1):
        super(DCASE, self).__init__()
        self.root = ROOT
        self.subset = subset
        self.rate = rate
        self.clip_duration = clip_duration
        self.mode = mode
        self.clips_per_sample = clips_per_sample
        self.augment = augment

        self.transform = transform if transform is not None else lambda x: x
        self.target_transform = target_transform

        if self.subset == 'train':
            fns = [l.split()[0] for l in open(ROOT+'/train.txt')]
            labels = [int(l.split()[1]) for l in open(ROOT + '/train.txt')]
        elif self.subset == 'test':
            fns = [l.split()[0] for l in open(ROOT+'/test.txt')]
            labels = [int(l.split()[1]) for l in open(ROOT + '/test.txt')]
        else:
            raise ValueError('unknown subset')

        from utils.ioutils.av_wrappers import av_meta
        self.time_lims = np.zeros((len(fns), 2))
        for i, fn in enumerate(fns):
            fn = '{}/{}'.format(ROOT, fn)
            _, m = av_meta(fn, audio=0)
            self.time_lims[i] = (float(m['start_time']), float(m['start_time'])+float(m['duration']), )

        self.classes = [str(lbl) for lbl in range(10)]
        self.filenames = np.array([fn.encode('utf-8') for fn in fns])
        self.labels = np.array(labels).astype(int)

        if mode == 'clip':
            self.resample()

    def resample(self):
        import random
        samples = []
        for i in range(self.labels.size):
            min_t = self.time_lims[i, 0]
            max_t = self.time_lims[i, 1]-self.clip_duration
            samples.extend([(float(i), random.uniform(min_t, max_t)) for _ in range(self.clips_per_sample)])
        self.samples = np.array(samples)

    def get_random_window(self, index):
        clip_idx = index % self.labels.size
        min_t, max_t = self.time_lims[clip_idx]
        ss = random.uniform(min_t, max_t-self.clip_duration*2 - 0.1)
        return clip_idx, ss

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        if self.mode == 'clip':
            clip_idx, ss = self.get_random_window(index)
            return self.get_sample(int(clip_idx), ss)
        elif self.mode == 'video':
            chunks = []
            for ss in np.linspace(0, self.time_lims[index, 1], self.clips_per_sample, endpoint=True):
                audio, target = self.get_sample(index, ss)
                chunks += [audio]
            chunks = torch.stack(chunks)
            return chunks, target
        else:
            raise ValueError('Unknown dataset mode.')

    def __len__(self):
        if self.mode == 'clip':
            return self.samples.shape[0]
        else:
            return self.labels.size

    def __repr__(self):
        head = "DCASE datat"
        body = ["Subset: {}".format(self.subset)]
        body += ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        if self.transform is not None:
            body += [repr(self.transform)]
        lines = [head] + [" " + line for line in body]
        return '\n'.join(lines)

    def extra_repr(self):
        return ""

    def get_sample(self, clip_idx, ss, duration=None):
        if duration is None:
            duration = self.clip_duration

        # Load audio
        path = '{}/{}'.format(self.root, self.filenames[clip_idx].decode('utf-8'))
        rate = self.rate * random.uniform(3./4., 4./3.) if self.augment else self.rate
        sig, rate = wav_loader(path, rate, start_time=ss, duration=duration*2)

        rate = self.rate
        sig = np.concatenate((sig, sig), 1)
        sig = sig[:, :int(duration*rate)]

        # Preprocess audio
        if self.transform is not None:
            if isinstance(self.transform, list):
                for t in self.transform:
                    sig, rate = t(sig, rate, duration)
            else:
                sig, _ = self.transform(sig, rate, duration)

        # Get label
        target = self.labels[clip_idx]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sig, target


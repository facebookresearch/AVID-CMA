import torch.utils.data as data
import csv
import numpy as np
import torch
from utils.ioutils import av_wrappers
import random

ROOT = '/checkpoint/pmorgado/ESC-50'
META = '/checkpoint/pmorgado/ESC-50/meta'


def wav_loader(filename, rate=None, start_time=0, duration=None):
    _, (data, sr) =  av_wrappers.av_loader(filename, audio_fps=rate, start_time=start_time, duration=duration, return_video=False, return_audio=True)
    return data, sr


class ESC50(data.Dataset):
    def __init__(self,
                 subset,
                 rate=48000,
                 clip_duration=1.,
                 transform=None,
                 target_transform=None,
                 augment=False,
                 mode='clip',
                 clips_per_sample=1):
        super(ESC50, self).__init__()
        self.root = ROOT
        self.subset = subset.split('-')[0]
        self.fold = int(subset.split('-')[1])
        self.rate = rate
        self.clip_duration = clip_duration
        self.mode = mode
        self.clips_per_sample = clips_per_sample
        self.augment = augment

        self.transform = transform if transform is not None else lambda x: x
        self.target_transform = target_transform

        meta = list(csv.DictReader(open(META+'/esc50.csv')))
        if self.subset == 'train':
            meta = [m for m in meta if int(m['fold']) != self.fold]
        elif self.subset == 'test':
            meta = [m for m in meta if int(m['fold']) == self.fold]
        else:
            raise ValueError('unknown subset')

        classes = {int(m['target']): m['category'] for m in meta}
        self.classes = [classes[lbl] for lbl in range(50)]

        fns = ['{}/audio/{}'.format(ROOT, m['filename']) for m in meta]
        lbl = [int(m['target']) for m in meta]
        self.filenames = np.array([fn.encode('utf-8') for fn in fns])
        self.labels = np.array(lbl).astype(int)

        if mode == 'clip':
            self.resample()

    def resample(self):
        import random
        ss = 0
        max_t = 5.-self.clip_duration-0.1
        samples = []
        for i in range(self.labels.size):
            samples.extend([(float(i), random.uniform(ss, max_t)) for _ in range(self.clips_per_sample)])
        self.samples = np.array(samples)

    def get_random_window(self, index):
        clip_idx = index % self.labels.size
        ss = random.uniform(0, 4.9)
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
            for ss in np.linspace(0, 4.9, self.clips_per_sample, endpoint=True):
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
        head = "ESC50 dataset"
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
        path = self.filenames[clip_idx].decode('utf-8')
        rate = self.rate * random.uniform(3./4., 4./3.) if self.augment else self.rate
        # rate = self.rate
        sig, rate = wav_loader(path, rate, start_time=0, duration=4.9)

        rate = self.rate
        sig = np.concatenate((sig, sig), 1)
        sig = sig[:, int(ss*rate):int((ss+duration)*rate)]

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





def benchmark(batch_size=16, num_workers=4):
    import time
    import torch.utils.data as data
    from datasets import preprocessing
    import numpy as np

    CLIP_DURATION = 1.
    AUDIO_FPS = 48000
    SPECTROGRAM_FPS = 64.

    audio_preproc = preprocessing.AudioPreprocessingLibrosa(trim_pad=False, augment=True)
    spectrogram = preprocessing.SpectrogramLibrosa(inp_sr=AUDIO_FPS, spect_db=True, normalize=True, hop_size=1./SPECTROGRAM_FPS)
    def audio_transform(data, rate):
        return spectrogram(audio_preproc(data, rate, CLIP_DURATION))

    dataset = ESC50('train-1',
                    rate=AUDIO_FPS,
                    transform=audio_transform,
                    clip_duration=CLIP_DURATION,
                    mode='video')
    print(dataset)

    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True)

    tt = time.time()
    read_times = []
    frames_per_clip = int(CLIP_DURATION)
    for idx, batch in enumerate(loader):
        if idx == 0:
            for k in batch:
                print('{}'.format(k.shape))
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
        for bs in [16]:
            print('=' * 60)
            benchmark(batch_size=bs, num_workers=w)

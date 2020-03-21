import random
import torch
import numpy as np
import torch.utils.data as data
from utils.ioutils import av_wrappers
from collections import defaultdict
import scipy.sparse


def chararray(fn_list):
    charr = np.chararray(len(fn_list), itemsize=max([len(fn) for fn in fn_list]))
    for i in range(len(fn_list)):
        charr[i] = fn_list[i]
    return charr


class VideoDataset(data.Dataset):
    def __init__(self,
                 return_video=True,
                 video_root=None,
                 video_fns=None,
                 video_clip_duration=1.,
                 video_fps=25,
                 video_shape=(3, 8, 224, 224),
                 video_fps_out=None,
                 video_transform=None,
                 return_audio=True,
                 audio_root=None,
                 audio_fns=None,
                 audio_clip_duration=1.,
                 audio_fps=None,
                 audio_shape=(1, 64, 257),
                 audio_fps_out=None,
                 audio_transform=None,
                 return_labels=False,
                 labels=None,
                 return_index=False,
                 mode='clip',
                 clips_per_video=1,
                 max_offsync_augm=0,
                 time_lims=None,
                 missing_audio_as_zero=False,
                 ):
        super(VideoDataset, self).__init__()

        self.num_samples = 0
        self.return_video = return_video
        self.video_root = video_root
        if return_video:
            self.video_fns = chararray(video_fns)
            self.num_samples = self.video_fns.shape[0]
        self.video_fps = video_fps
        self.video_fps_out = video_fps_out
        self.video_shape = video_shape
        if video_transform is not None:
            if not isinstance(video_transform, list):
                video_transform = [video_transform]
        self.video_transform = video_transform

        self.return_audio = return_audio
        self.audio_root = audio_root
        if return_audio:
            self.audio_fns = chararray(audio_fns)
            self.num_samples = self.audio_fns.shape[0]
        self.audio_fps = audio_fps
        self.audio_fps_out = audio_fps_out
        self.audio_shape = audio_shape
        self.audio_transform = audio_transform

        self.return_labels = return_labels
        if return_labels:
            self.labels = np.array(labels)
        self.return_index = return_index

        self.video_clip_duration = video_clip_duration
        self.audio_clip_duration = audio_clip_duration
        self.max_offsync_augm = max_offsync_augm
        self.clips_per_video = clips_per_video
        self.time_lims = np.array(time_lims)     # (Video SS, Video FF, Audio SS, Audio FF)
        self.missing_audio_as_zero = missing_audio_as_zero
        self.mode = mode

    def __getitem__(self, index):
        if self.mode == 'clip':
            try:
                sample_idx = index % self.num_samples
                video_start_time, video_duration, audio_start_time, audio_duration = self.sample_snippet(sample_idx)
                sample = self.get_clip(int(sample_idx), video_start_time, audio_start_time, video_clip_duration=video_duration, audio_clip_duration=audio_duration)
                if sample is None:
                    return self[(index+1) % len(self)]

                return sample
            except Exception:
                return self[(index+1) % len(self)]

        else:
            vs, vf, ss, sf = self.time_lims[index]
            if self.return_audio:
                start_time = max(vs, ss) if ss < 0 else vs
                final_time = min(vf, sf) if ss < 0 else vf
            else:
                start_time = vs
                final_time = vf

            if final_time <= start_time:
                final_time = start_time + max(self.video_clip_duration, self.audio_clip_duration)

            video_dur = final_time - start_time
            sample = self.get_clip(index, start_time, start_time, video_clip_duration=video_dur, audio_clip_duration=video_dur)
            chunks = defaultdict(list)
            if self.return_video:
                nf = sample['frames'].shape[1]
                chunk_size = int(self.video_clip_duration * self.video_fps)
                if chunk_size >= nf:
                    chunks['frames'] = torch.stack([sample['frames'] for _ in range(self.clips_per_video)])
                else:
                    chunks['frames'] = torch.stack([sample['frames'][:, ss:ss+chunk_size]
                                                    for ss in np.linspace(0, max(nf-chunk_size, 1),
                                                                          self.clips_per_video).astype(int)])

            if self.return_audio:
                nf = sample['audio'].shape[1]
                chunk_size = int(self.audio_clip_duration * self.audio_fps_out)
                if chunk_size >= nf:
                    chunks['audio'] = torch.stack([sample['audio'] for _ in range(self.clips_per_video)])
                else:
                    chunks['audio'] = torch.stack([sample['audio'][:, ss:ss+chunk_size]
                                                   for ss in np.linspace(0, max(nf-chunk_size, 1),
                                                                         self.clips_per_video).astype(int)])

            if self.return_labels:
                chunks['label'] = sample['label']

            if self.return_index:
                ts = torch.from_numpy(np.linspace(start_time, final_time-self.video_clip_duration, self.clips_per_video))
                chunks['index'] = torch.stack([sample['index'][:1].repeat(self.clips_per_video), ts.float()], dim=1)

            return chunks

    def __len__(self):
        if self.mode == 'clip':
            return self.num_samples * self.clips_per_video
        else:
            return self.num_samples

    def __repr__(self):
        desc = "{}\n - Root: {}\n - Subset: {}\n - Num videos: {}\n - Num samples: {}\n".format(
            self.name, self.root, self.subset, self.num_videos, self.num_videos * self.clips_per_video)
        if self.return_video:
            desc += " - Example video: {}/{}\n".format(self.video_root, self.video_fns[0].decode())
        if self.return_audio:
            desc += " - Example audio: {}/{}\n".format(self.audio_root, self.audio_fns[0].decode())
        return desc

    def sample_snippet(self, idx):
        video_st, video_ft, audio_st, audio_ft = self.time_lims[idx]

        if not self.return_audio:
            video_duration = video_ft - video_st
            if self.video_clip_duration > video_duration:
                return 0., video_duration, 0., video_duration
            else:
                min_d, max_d = self.video_clip_duration, min(self.video_clip_duration, video_duration)
                duration = random.uniform(min_d, max_d)
                sample_ss_v = random.uniform(video_st, video_ft - duration)
                return sample_ss_v, duration, sample_ss_v, duration

        else:
            min_ss = max(audio_st, video_st)
            max_ss = min(audio_ft - self.audio_clip_duration, video_ft - self.video_clip_duration)
            assert max_ss > min_ss
            if self.audio_clip_duration > self.video_clip_duration:
                sample_ss_a = random.uniform(min_ss, max_ss)
                sample_tt_a = sample_ss_a + self.audio_clip_duration

                win_min = max(sample_ss_a - self.max_offsync_augm, video_st)
                win_max = min(sample_tt_a + self.max_offsync_augm - self.video_clip_duration, video_ft)
                sample_ss_v = random.uniform(win_min, win_max)
                return sample_ss_v, self.video_clip_duration, sample_ss_a, self.audio_clip_duration
            else:
                sample_ss_v = random.uniform(min_ss, max_ss)
                sample_tt_v = sample_ss_v + self.video_clip_duration

                win_min = max(sample_ss_v - self.max_offsync_augm, audio_st)
                win_max = min(sample_tt_v + self.max_offsync_augm - self.audio_clip_duration, audio_ft)
                sample_ss_a = random.uniform(win_min, win_max)
                return sample_ss_v, self.video_clip_duration, sample_ss_a, self.audio_clip_duration

    def get_clip(self, clip_idx, video_start_time, audio_start_time, video_clip_duration=None, audio_clip_duration=None, return_video=True, return_audio=True, transform=True):
        if video_clip_duration is None:
            video_clip_duration = self.video_clip_duration
        if audio_clip_duration is None:
            audio_clip_duration = self.audio_clip_duration

        sample = {}
        if self.return_video and return_video:
            video_fn = '{}/{}'.format(self.video_root, self.video_fns[clip_idx].decode())
            (frames, fps, start_time), _ = av_wrappers.av_loader2(
                video_fn,
                return_video=True,
                video_fps=self.video_fps,
                return_audio=False,
                start_time=video_start_time,
                duration=video_clip_duration,
            )
            if transform and self.video_transform is not None:
                for t in self.video_transform:
                    frames = t(frames)

            sample['frames'] = frames
            sample['frame_rate'] = fps
            audio_start_time = audio_start_time - (video_start_time - start_time)

        if self.return_audio and return_audio:
            audio_fn = '{}/{}'.format(self.audio_root, self.audio_fns[clip_idx].decode())
            _, (samples, sr) = av_wrappers.av_loader2(
                audio_fn,
                return_video=False,
                return_audio=True,
                audio_fps=self.audio_fps,
                start_time=audio_start_time,
                duration=audio_clip_duration,
            )
            if self.missing_audio_as_zero and samples is None:
                sample['audio'] = torch.zeros(self.audio_shape)
                sample['audio_rate'] = self.audio_fps_out
            else:
                if transform and self.audio_transform is not None:
                    if isinstance(self.audio_transform, list):
                        sig, rate = samples, sr
                        for t in self.audio_transform:
                            sig, rate = t(sig, rate, audio_clip_duration)
                    else:
                        sig, rate = self.audio_transform(samples, sr)
                    sample['audio'] = sig
                    sample['audio_rate'] = rate
                else:
                    sample['audio'] = samples
                    sample['audio_rate'] = sr

        if self.return_labels:
            lbl = self.labels[clip_idx]
            if isinstance(lbl, np.ndarray):
                sample['label'] = torch.from_numpy(lbl)
            else:
                sample['label'] = lbl

        if self.return_index:
            sample['index'] = torch.tensor([float(clip_idx), video_start_time, video_clip_duration, audio_start_time, audio_clip_duration])

        return sample

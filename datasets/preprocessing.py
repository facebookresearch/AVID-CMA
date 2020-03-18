import torch
import numpy as np
import random
import librosa
import torchaudio


class VideoPrep_MSC_CJ(object):
    def __init__(self,
                 crop=(224, 224),
                 color=(0.4, 0.4, 0.4, 0.2),
                 min_area=0.08,
                 augment=True,
                 normalize=True,
                 totensor=True,
                 num_frames=8,
                 pad_missing=False,
                 ):
        from utils.videotransforms import video_transforms, volume_transforms, tensor_transforms
        self.crop = crop
        self.augment = augment
        self.num_frames = num_frames
        self.pad_missing = pad_missing
        if normalize:
            assert totensor

        if augment:
            transforms = [
                video_transforms.RandomResizedCrop(crop, scale=(min_area, 1.)),
                video_transforms.RandomHorizontalFlip(),
                video_transforms.ColorJitter(*color),
            ]
        else:
            transforms = [
                video_transforms.Resize(int(crop[0]/0.875)),
                video_transforms.CenterCrop(crop),
            ]

        if totensor:
            transforms += [volume_transforms.ClipToTensor()]
            if normalize:
                transforms += [tensor_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        self.transform = video_transforms.Compose(transforms)

    def __call__(self, frames):
        frames = self.transform(frames)
        if self.pad_missing:
            while True:
                n_missing = self.num_frames - frames.shape[1]
                if n_missing > 0:
                    frames = torch.cat((frames, frames[:, :n_missing]), 1)
                else:
                    break
        return frames


class VideoPrep_Crop_CJ(object):
    def __init__(self,
                 resize=(256, 256),
                 crop=(224, 224),
                 color=(0.4, 0.4, 0.4, 0.2),
                 num_frames=8,
                 pad_missing=False,
                 augment=True,
                 normalize=True,
                 totensor=True,
                 ):
        self.resize = resize
        self.crop = crop
        self.augment = augment
        self.num_frames = num_frames
        self.pad_missing = pad_missing
        if normalize:
            assert totensor

        from utils.videotransforms import video_transforms, volume_transforms, tensor_transforms

        if augment:
            transforms = [
                video_transforms.Resize(resize),
                video_transforms.RandomCrop(crop),
                video_transforms.RandomHorizontalFlip(),
                video_transforms.ColorJitter(*color),
            ]
        else:
            transforms = [
                video_transforms.Resize(resize),
                video_transforms.CenterCrop(crop),
            ]
        if totensor:
            transforms += [volume_transforms.ClipToTensor()]
            if normalize:
                transforms += [tensor_transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        self.transform = video_transforms.Compose(transforms)

    def __call__(self, frames):
        if isinstance(frames[0], list):
            return torch.stack([self(f) for f in frames])
        frames = self.transform(frames)
        if self.pad_missing:
            while True:
                n_missing = self.num_frames - frames.shape[1]
                if n_missing > 0:
                    frames = torch.cat((frames, frames[:, :n_missing]), 1)
                else:
                    break
        return frames


class AudioPreprocessing(object):
    def __init__(self, num_frames, augment=False):
        import torchaudio
        self.augment = augment
        self.trimpad = torchaudio.transforms.PadTrim(num_frames, channels_first=True)

    def __call__(self, sig, sr):
        # Downmix to mono
        sig = sig.mean(0).astype(np.float32)

        # Augment by changing volume +/- 10%
        if self.augment:
            vol_coeff = (random.random() * 0.2) + 0.9
            sig *= vol_coeff

        sig = torch.tensor(sig).unsqueeze(0)    # 1 x N
        return self.trimpad(sig)


def normalize_spectrogram(tensor, mean, std):
    mean = torch.as_tensor(mean, dtype=torch.float32, device=tensor.device)
    std = torch.as_tensor(std, dtype=torch.float32, device=tensor.device)
    tensor.sub_(mean[None, None, :]).div_(std[None, None, :])
    return tensor


class Spectrogram(object):
    def __init__(self, n_fft=512, hop_size=0.005, inp_sr=48000, spect_db=True, normalize=False):
        self.n_fft = n_fft
        self.hop_size = int(hop_size*inp_sr)
        self.inp_sr = inp_sr
        self.spect_db = spect_db
        self.normalize = normalize

        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft, hop=self.hop_size, power=2., window=torch.hann_window, normalize=False)
        if self.spect_db:
            self.toDB = torchaudio.transforms.SpectrogramToDB("power", top_db=100)

        if self.normalize:
            if self.spect_db:
                stats = np.load('/checkpoint/pmorgado/data_cache/audioset/audio-spectDB-norm-stats.npz')
            else:
                stats = np.load('/checkpoint/pmorgado/data_cache/audioset/audio-spect-norm-stats.npz')
            self.mean, self.std = stats['mean'], stats['std']

    def __call__(self, sig):
        spect = self.spectrogram(sig)[:, :-1]
        if self.spect_db:
            spect = self.toDB(spect)
        if self.normalize:
            spect = normalize_spectrogram(spect, self.mean, self.std)
        return spect


class AudioPrepLibrosa(object):
    def __init__(self, trim_pad=True, duration=None, missing_as_zero=False, augment=False, to_tensor=False, volume=0.1):
        self.trim_pad = trim_pad
        self.missing_as_zero = missing_as_zero
        self.augment = augment
        self.to_tensor = to_tensor
        self.volume = volume
        if trim_pad:
            assert duration is not None
            self.duration = duration

    def __call__(self, sig, sr, duration=None):
        if duration is None:
            duration = self.duration
        num_frames = int(duration*sr)

        # Check if audio is missing
        if self.missing_as_zero and sig is None:
            sig = np.zeros((1, num_frames), dtype=np.float32)

        # Downmix to mono
        sig = sig.mean(0).astype(np.float32)

        # Trim or pad to constant shape
        if self.trim_pad:
            if sig.shape[0] > num_frames:
                sig = sig[:num_frames]
            elif sig.shape[0] < num_frames:
                n_pad = num_frames - sig.shape[0]
                sig = np.pad(sig, (0, n_pad), mode='constant', constant_values=(0., 0.))

        # Augment by changing volume +/- 10%
        if self.augment:
            sig *= random.uniform(1.-self.volume, 1.+self.volume)

        sig = sig[np.newaxis]
        if self.to_tensor:
            sig = torch.from_numpy(sig)

        return sig, sr


class SpectrogramLibrosa(object):
    def __init__(self, n_fft=512, win_length=None, hop_size=0.005, spect_db=True, normalize=False):
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_size = hop_size
        self.rate = 1./hop_size
        self.spect_db = spect_db
        self.normalize = normalize

        if self.normalize:
            if self.spect_db:
                if n_fft == 512:
                    stats = np.load('/checkpoint/pmorgado/data_cache/audioset/audio-spectDB-librosa-norm-stats.npz')
                elif n_fft == 256:
                    stats = np.load('/checkpoint/pmorgado/data_cache/audioset/audio-spectDB-24k-257-norm-stats.npz')
            else:
                stats = np.load('/checkpoint/pmorgado/data_cache/audioset/audio-spect-librosa-norm-stats.npz')
            self.mean, self.std = stats['mean'], stats['std']

    def __call__(self, sig, sr, duration=None):
        hop_length = int(self.hop_size * sr)
        spect = np.abs(librosa.stft(sig[0], n_fft=self.n_fft, win_length=self.win_length, hop_length=hop_length))**2.
        if duration is not None:
            num_frames = int(duration * self.rate)
            spect = spect[:, :num_frames]

        if self.spect_db:
            spect = librosa.core.power_to_db(spect, top_db=100)
        if self.normalize:
            spect = (spect - self.mean[:, np.newaxis]) / (self.std[:, np.newaxis] + 1e-5)
        spect_tensor = torch.from_numpy(spect)
        spect_tensor = torch.transpose(spect_tensor, 0, 1).unsqueeze(0)
        return spect_tensor, self.rate


class SpectrogramLibrosa2(object):
    def __init__(self, fps, n_fft=512, hop_size=0.005, spect_db=True, normalize=False):
        self.inp_fps = fps
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.rate = 1./hop_size
        self.spect_db = spect_db
        self.normalize = normalize

        if self.normalize:
            if self.spect_db:
                if n_fft == 512 and fps == 24000:
                    stats = np.load('/checkpoint/pmorgado/data_cache/audioset/audio-spectDB-24k-513-norm-stats.npz')
                elif n_fft == 256 and fps == 24000:
                    stats = np.load('/checkpoint/pmorgado/data_cache/audioset/audio-spectDB-24k-257-norm-stats.npz')
            else:
                stats = np.load('/checkpoint/pmorgado/data_cache/audioset/audio-spect-librosa-norm-stats.npz')
            self.mean, self.std = stats['mean'], stats['std']

    def __call__(self, sig, sr, duration=None):
        hop_length = int(self.hop_size * sr)
        spect = np.abs(librosa.stft(sig[0], n_fft=self.n_fft*2, hop_length=hop_length))**2.
        spect = np.concatenate([spect[:1], spect[1:].reshape(self.n_fft//2, 2, -1).mean(1)], 0)
        if duration is not None:
            num_frames = int(duration * self.rate)
            spect = spect[:, :num_frames]

        if self.spect_db:
            spect = librosa.core.power_to_db(spect, top_db=100)
        if self.normalize:
            spect = (spect - self.mean[:, np.newaxis]) / (self.std[:, np.newaxis] + 1e-5)
        spect_tensor = torch.from_numpy(spect)
        spect_tensor = torch.transpose(spect_tensor, 0, 1).unsqueeze(0)
        return spect_tensor, self.rate


class MelSpectrogramLibrosa(object):
    def __init__(self, n_mels=128, n_fft=2048, hop_size=0.01, spect_db=True, normalize=False):
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.hop_size = hop_size
        self.rate = 1. / hop_size
        self.spect_db = spect_db
        self.normalize = normalize

        if self.normalize:
            if self.spect_db:
                stats = np.load('/checkpoint/pmorgado/data_cache/audioset/melspect-db-fs40960-fft2048-mels40-norm-stats.npz')
            else:
                stats = np.load('/checkpoint/pmorgado/data_cache/audioset/melspect-fs40960-fft2048-mels128-norm-stats.npz')
            self.mean, self.std = stats['mean'], stats['std']

    def __call__(self, sig, sr, duration=None):
        hop_size = int(self.hop_size * sr)

        spect = librosa.feature.melspectrogram(sig[0], sr=sr, n_fft=self.n_fft, hop_length=hop_size, n_mels=self.n_mels)
        if duration is not None:
            num_frames = int(duration * self.rate)
            spect = spect[:, :num_frames]

        if self.spect_db:
            spect = librosa.core.power_to_db(spect, top_db=100)
        if self.normalize:
            spect = (spect - self.mean[:, np.newaxis]) / (self.std[:, np.newaxis] + 1e-5)
        spect_tensor = torch.from_numpy(spect.astype(np.float32))
        spect_tensor = torch.transpose(spect_tensor, 0, 1).unsqueeze(0)
        return spect_tensor, self.rate


def main():
    import datasets
    import torch
    import torch.utils.data as data
    import numpy as np

    audio_fps = 24000
    n_fft = 512
    clip_duration = 2.
    spect_db = True
    audio_transform = [
        AudioPrepLibrosa(duration=clip_duration, trim_pad=True, augment=True),
        SpectrogramLibrosa(n_fft=n_fft, hop_size=1./64, spect_db=spect_db, normalize=False)
    ]
    dataset = datasets.AudioSet(
            subset='unbalanced_train-100k',
            return_video=False,
            video_fps=1,
            video_fps_out=1,
            return_audio=True,
            audio_clip_duration=clip_duration,
            audio_fps=audio_fps,
            audio_fps_out=64,
            audio_shape=(1, int(64*clip_duration), n_fft//2+1),
            audio_transform=audio_transform,
        )

    loader = data.DataLoader(dataset, batch_size=100, num_workers=20, pin_memory=True, shuffle=True)
    audio_samples = []
    for ii, sample in enumerate(loader):
        audio_samples.append(sample['audio'][:, 0])
        if ii == 500:
            break
        if ii % 10 == 0:
            print(audio_samples[-1].shape, audio_samples[-1][0].min(), audio_samples[-1][0].max())
            print(ii)
    audio_samples = torch.cat(audio_samples, 0)
    mean = torch.flatten(audio_samples, 0, 1).mean(0).cpu().numpy()
    std = torch.flatten(audio_samples, 0, 1).std(0).cpu().numpy()
    out_fn = f'/checkpoint/pmorgado/data_cache/audioset/audio-spect{"DB" if spect_db else ""}-{int(audio_fps/1000)}k-{n_fft+1}-norm-stats.npz'
    np.savez(out_fn, mean=mean, std=std)
    print(mean)
    print(std)


if __name__ == '__main__':
    main()
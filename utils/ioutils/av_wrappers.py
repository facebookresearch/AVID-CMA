import av
import numpy as np
from fractions import Fraction
av.logging.set_level(0)


def av_meta(inpt, video=None, audio=None, format=None):
    if isinstance(inpt, str):
        try:
            container = av.open(inpt, format=format)
        except av.AVError:
            return None, None
    else:
        container = inpt

    if video is not None and len(container.streams.video) > video:
        video_stream = container.streams.video[video]
        time_base = video_stream.time_base
        duration = video_stream.duration * time_base
        start_time = video_stream.start_time * time_base
        width = video_stream.width
        height = video_stream.height
        fps = video_stream.average_rate
        nframes = video_stream.frames
        video_meta = {'fps': fps, 'size': (width, height), 'start_time': start_time, 'duration': duration, 'nframes': nframes}
    else:
        video_meta = None

    if audio is not None and len(container.streams.audio) > audio:
        audio_stream = container.streams.audio[audio]
        time_base = audio_stream.time_base
        duration = audio_stream.duration * time_base
        start_time = audio_stream.start_time * time_base
        channels = audio_stream.channels
        fps = audio_stream.rate
        chunk_size = audio_stream.frame_size
        chunks = audio_stream.frames
        audio_meta = {'channels': channels, 'fps': fps, 'start_time': start_time, 'duration': duration, 'chunks': chunks, 'chunk_size': chunk_size}
    else:
        audio_meta = None

    return video_meta, audio_meta


def av_loader(video_path, audio_path, meta, video_fps=-1, audio_fps=-1, start_time=0, duration=-1, return_video=True, return_audio=True):
    # video_meta, _ = av_meta(video_container, video=0)
    # _, audio_meta = av_meta(audio_container, audio=0)

    # Check loader arguments
    video_end = meta['video_meta']['duration'] + meta['video_meta']['start_time']
    if return_audio:
        audio_end = meta['audio_meta']['duration'] + meta['audio_meta']['start_time']
        end_time = min(video_end, audio_end)
    else:
        end_time = video_end
    if duration == -1:
        duration = end_time - start_time
    duration = min(duration, end_time - start_time)

    if video_fps == -1:
        video_fps = meta['video_meta']['fps']
    if return_audio and audio_fps == -1:
        audio_fps = meta['audio_meta']['fps']

    end_time = start_time + duration

    # Extract video frames
    vframes = None
    if return_video:
        container = av.open(video_path, format=meta['video_fmt'])
        video_stream = container.streams.video[0]

        video_start_time = int(start_time * video_stream.average_rate) / video_stream.average_rate
        outp_times = [t for t in np.arange(video_start_time, video_start_time + duration - 0.5/video_fps, 1./video_fps)]
        outp_vframes = [int((t - meta['video_meta']['start_time']) * meta['video_meta']['fps']) for t in outp_times]

        # container.seek(int(video_start_time / video_stream.time_base), stream=video_stream)
        container.seek(int(video_start_time * av.time_base))
        vframes = []
        for frame in container.decode(video=0):
            frame_id = frame.pts * frame.time_base * video_stream.rate
            if frame_id < outp_vframes[len(vframes)]:
                continue
            pil_img = frame.to_image()
            while frame_id >= outp_vframes[len(vframes)]:
                vframes.append((frame.time, pil_img))
                if len(vframes) == len(outp_vframes):
                    break
            if len(vframes) == len(outp_vframes):
                break

        vframes = [vf[1] for vf in vframes]

    # Extract audio
    aframes = None
    if return_audio:
        audio_resampler = av.audio.resampler.AudioResampler(
            format="s16p", layout="mono", rate=audio_fps)

        container = av.open(audio_path, format=meta['audio_fmt'])
        audio_stream = container.streams.audio[0]

        container.seek(int(start_time / audio_stream.time_base), stream=audio_stream)
        aframes = []
        num_aframes = 0
        for frame in container.decode(audio=0):
            frame_pts = frame.pts * frame.time_base
            frame_end_pts = frame_pts + Fraction(frame.samples, frame.rate)
            if frame_end_pts < start_time:   # Skip until start time
                continue
            if frame_pts > end_time:       # Exit if clip has been extracted
                break
            frame.pts = num_aframes
            num_aframes += frame.samples
            np_snd = audio_resampler.resample(frame).to_ndarray()
            aframes.append((frame_pts, np_snd))

        audio_start_time = aframes[0][0]
        aframes = np.concatenate([af[1] for af in aframes], 1)
        ss = int((start_time - audio_start_time) * audio_fps)
        t = int(duration * audio_fps)
        if ss < 0:
            aframes = np.pad(aframes, ((0, 0), (-ss, 0)), 'constant', constant_values=0)
            ss = 0
        if t > aframes.shape[1]:
            aframes = np.pad(aframes, ((0, 0), (0, t-aframes.shape[1])), 'constant', constant_values=0)
        aframes = aframes[:, ss: ss+t]

    return vframes, aframes


def av_loader2(path, video_fps=None, audio_fps=None, start_time=0, duration=None, return_video=True, return_audio=True):
    container = av.open(path)

    # Extract video frames
    vframes = None
    if return_video and len(container.streams.video)>=1:
        video_stream = container.streams.video[0]
        vid_ss = video_stream.start_time * video_stream.time_base
        vid_dur = video_stream.duration * video_stream.time_base
        vid_ff = vid_ss + vid_dur
        vid_fps = video_stream.average_rate

        if video_fps is None:
            video_fps = vid_fps

        if duration is None:
            duration = vid_ff - start_time

        outp_times = [t for t in np.arange(start_time, min(start_time + duration - 0.5/vid_fps, vid_ff), 1./video_fps)][:int(duration*video_fps)]
        outp_vframes = [int((t - vid_ss) * vid_fps) for t in outp_times]
        start_time = outp_vframes[0] / float(vid_fps)

        container.seek(int(start_time * av.time_base))
        vframes = []
        for frame in container.decode(video=0):
            if len(vframes) == len(outp_vframes):
                break
            frame_id = frame.pts * frame.time_base * video_stream.average_rate
            if frame_id < outp_vframes[len(vframes)]:
                continue
            pil_img = frame.to_image()
            while frame_id >= outp_vframes[len(vframes)]:
                vframes.append((frame.time, pil_img))
                if len(vframes) == len(outp_vframes):
                    break

        vframes = [vf[1] for vf in vframes]

    # Extract audio
    aframes = None
    if return_audio and len(container.streams.audio)>=1:
        audio_stream = container.streams.audio[0]
        snd_ss = audio_stream.start_time * audio_stream.time_base if audio_stream.start_time is not None else 0.
        snd_dur = audio_stream.duration * audio_stream.time_base
        snd_ff = snd_ss + snd_dur
        snd_fps = audio_stream.rate

        if audio_fps is None:
            resample = False
            audio_fps = snd_fps
        else:
            resample = True
            audio_resampler = av.audio.resampler.AudioResampler(
                format="s16p", layout="mono", rate=audio_fps)

        if duration is None:
            duration = snd_ff - start_time
        duration = min(duration, snd_ff - start_time)
        end_time = start_time + duration

        container.seek(int(start_time / av.time_base))
        aframes = []
        first_frame_pts = None
        for frame in container.decode(audio=0):
            frame_pts = frame.pts * frame.time_base
            frame_end_pts = frame_pts + Fraction(frame.samples, frame.rate)
            if frame_end_pts < start_time:   # Skip until start time
                continue
            if frame_pts > end_time:       # Exit if clip has been extracted
                break
            if first_frame_pts is None:
                first_frame_pts = frame.pts

            try:
                frame.pts = None
                if resample:
                    np_snd = audio_resampler.resample(frame).to_ndarray()
                else:
                    np_snd = frame.to_ndarray()
                aframes.append((frame_pts, np_snd))
            except AttributeError:
                break

        if aframes:
            audio_start_time = aframes[0][0]
            aframes = np.concatenate([af[1] for af in aframes], 1)
            ss = int((start_time - audio_start_time) * audio_fps)
            t = int(duration * audio_fps)
            if ss < 0:
                aframes = np.pad(aframes, ((0, 0), (-ss, 0)), 'constant', constant_values=0)
                ss = 0
            if t > aframes.shape[1]:
                aframes = np.pad(aframes, ((0, 0), (0, t-aframes.shape[1])), 'constant', constant_values=0)
            aframes = aframes[:, ss: ss+t]
            aframes = aframes / np.iinfo(aframes.dtype).max
        else:
            aframes  = np.zeros((1, 0), dtype=np.float32)

    container.close()

    return (vframes, video_fps, start_time), (aframes, audio_fps)


def av_load_video(path, video_fps=None, start_time=0, duration=None):
    container = av.open(path)

    if len(container.streams.video)==0:
        return None

    # Extract video frames
    video_stream = container.streams.video[0]
    vid_ss = video_stream.start_time * video_stream.time_base
    vid_dur = video_stream.duration * video_stream.time_base
    vid_ff = vid_ss + vid_dur
    vid_fps = video_stream.average_rate

    if video_fps is None:
        video_fps = vid_fps

    if duration is None:
        duration = vid_ff - start_time

    outp_times = np.arange(start_time, start_time + duration, 1./video_fps)
    outp_times = [t for t in outp_times if t < vid_ff]
    outp_vframes = [int((t - vid_ss) * vid_fps) for t in outp_times]

    vframes = []
    container.seek(int(outp_times[len(vframes)] * av.time_base))
    for frame in container.decode(video=0):
        if len(vframes) == len(outp_vframes):
            break
        frame_id = frame.pts * frame.time_base * video_stream.rate
        if frame_id < outp_vframes[len(vframes)]:
            continue
        pil_img = frame.to_image()
        while frame_id >= outp_vframes[len(vframes)]:
            vframes.append((frame.time, pil_img))
            if len(vframes) == len(outp_vframes):
                break

    vframes = [vf[1] for vf in vframes]
    return vframes, video_fps


def av_load_all_video(path, video_fps=None):
    container = av.open(path)

    if len(container.streams.video)==0:
        return None

    # Extract video frames
    video_stream = container.streams.video[0]
    vid_ss = video_stream.start_time * video_stream.time_base
    vid_dur = video_stream.duration * video_stream.time_base
    vid_ff = vid_ss + vid_dur
    vid_fps = video_stream.average_rate

    if video_fps is None:
        video_fps = vid_fps

    outp_times = np.arange(vid_ss, vid_ff, 1./video_fps)
    outp_times = [t for t in outp_times if t < vid_ff]
    outp_vframes = [int((t - vid_ss) * vid_fps) for t in outp_times]

    vframes = []
    container.seek(int(outp_times[len(vframes)] * av.time_base))
    for frame in container.decode(video=0):
        if len(vframes) == len(outp_vframes):
            break
        frame_id = frame.pts * frame.time_base * video_stream.rate
        if frame_id < outp_vframes[len(vframes)]:
            continue
        pil_img = frame.to_image()
        while frame_id >= outp_vframes[len(vframes)]:
            vframes.append((frame.time, pil_img))
            if len(vframes) == len(outp_vframes):
                break

    vframes = [vf[1] for vf in vframes]
    return vframes, video_fps


def av_load_all_audio(path, audio_fps=None):
    container = av.open(path)

    if len(container.streams.audio) == 0:
        return None

    audio_stream = container.streams.audio[0]
    snd_fps = audio_stream.rate

    if audio_fps is None:
        resample = False
        audio_fps = snd_fps
    else:
        resample = True
        audio_resampler = av.audio.resampler.AudioResampler(format="s16p", layout="mono", rate=audio_fps)

    aframes = []
    for frame in container.decode(audio=0):
        frame_pts = frame.pts * frame.time_base
        if resample:
            np_snd = audio_resampler.resample(frame).to_ndarray()
        else:
            np_snd = frame.to_ndarray()
        aframes.append((frame_pts, np_snd))

    aframes = np.concatenate([af[1] for af in aframes], 1)
    aframes = aframes / np.iinfo(aframes.dtype).max

    return aframes, audio_fps


def av_write_video(fn, video_frames, fps):
    assert fn.endswith('.mp4')

    container = av.open(fn, mode='w')
    stream = container.add_stream('mpeg4', rate=fps)
    stream.width = video_frames.shape[2]
    stream.height = video_frames.shape[1]
    stream.pix_fmt = 'yuv420p'

    for img in video_frames:
        img = np.round(img).astype(np.uint8)
        img = np.clip(img, 0, 255)

        frame = av.VideoFrame.from_ndarray(img, format='rgb24')
        for packet in stream.encode(frame):
            container.mux(packet)

    # Flush stream
    for packet in stream.encode():
        container.mux(packet)

    # Close the file
    container.close()


if __name__ == '__main__':
    av_load_all_video('/datasets01_101/ucf101/112018/data/v_ApplyLipstick_g01_c01.avi', video_fps=8,)
    av_load_all_audio('/datasets01_101/ucf101/112018/data/v_ApplyLipstick_g01_c01.avi', audio_fps=48000, )
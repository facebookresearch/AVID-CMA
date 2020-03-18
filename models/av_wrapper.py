import random
import torch
import torch.nn as nn


__all__ = [
    'av_wrapper'
]


class Head(nn.Module):
    def __init__(self, input_dim, proj_dims):
        super(Head, self).__init__()
        if not isinstance(proj_dims, list):
            proj_dims = [proj_dims]

        projection = []
        for i, d in enumerate(proj_dims):
            projection += [nn.Linear(input_dim, d)]
            input_dim = d
            if i < len(proj_dims)-1:
                projection += [nn.ReLU(inplace=True)]
        self.projection = nn.Sequential(*projection)
        self.out_dim = proj_dims[-1]

    def forward(self, x):
        return self.projection(x)


class AV_Wrapper(nn.Module):
    def __init__(self, video_model, audio_model, proj_dim=128):
        super(AV_Wrapper, self).__init__()
        self.video_model = video_model
        self.audio_model = audio_model

        self.use_linear_proj = proj_dim is not None
        if proj_dim is not None:
            self.video_proj = Head(video_model.out_dim, proj_dim)
            self.audio_proj = Head(audio_model.out_dim, proj_dim)
            self.out_dim = self.video_proj.out_dim
        else:
            self.out_dim = video_model.out_dim

    def forward(self, video, audio):
        video_emb = self.video_model(video)
        video_emb = video_emb.view(video_emb.shape[0], video_emb.shape[1])
        if self.use_linear_proj:
            video_emb = self.video_proj(video_emb)

        audio_emb = self.audio_model(audio)
        audio_emb = audio_emb.view(audio_emb.shape[0], audio_emb.shape[1])
        if self.use_linear_proj:
            audio_emb = self.audio_proj(audio_emb)

        return video_emb, audio_emb


def av_wrapper(video_backbone, video_backbone_args, audio_backbone, audio_backbone_args, proj_dim=128, checkpoint=None):
    import models
    assert video_backbone in models.__dict__, 'Unknown model architecture'
    assert audio_backbone in models.__dict__, 'Unknown model architecture'
    video_model = models.__dict__[video_backbone](**video_backbone_args)
    audio_model = models.__dict__[audio_backbone](**audio_backbone_args)

    model = AV_Wrapper(video_model, audio_model, proj_dim=proj_dim)
    if checkpoint is not None:
        ckp = torch.load(checkpoint, map_location='cpu')
        nn.DataParallel(model).load_state_dict(ckp['model'])

    return model


def main():
    import utils.main_utils as utils
    import yaml
    import sys
    import GPUtil
    sys.path.insert(0, '.')

    cfg = yaml.safe_load(open('configs/main/mc-avc/l3-mc100-Fcat-C1-100k.yaml'))
    model = av_wrapper(**cfg['model']['args'])
    model.cuda()
    model.train()
    print(model)
    print(utils.parameter_description(model))

    # Dummy data
    dummy_video = torch.ones((100, 8, 3, 112, 112)).cuda()
    dummy_audio = torch.ones((100, 1, 200, 257)).cuda()

    logits, labels = model(dummy_video, dummy_audio)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(logits, labels)
    print(GPUtil.getGPUs()[0].memoryUsed)
    loss.backward()
    print(logits.shape)
    print(labels.shape)


if __name__ == '__main__':
    main()
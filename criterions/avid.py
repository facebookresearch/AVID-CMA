import torch
from torch import nn
import torch.distributed as dist
import pprint
from utils.main_utils import AliasMethod, _gather_from_all
from criterions.cross_entropy import xEntCriterion
from criterions.nce import NCECriterion

__all__ = ['AVID']


class AVIDSimilarityMemoryBank(nn.Module):
    def __init__(self,
                 memory_size,
                 embedding_dim,
                 xModal=True,
                 wModal=False,
                 num_negatives=1024,
                 momentum=0.5,
                 device=0
                 ):
        super(AVIDSimilarityMemoryBank, self).__init__()
        self.num_negatives = num_negatives
        self.temperature = 0.07
        if not isinstance(momentum, (list, tuple)):
            momentum = [momentum]*2
        self.momentum = momentum
        self.device = device

        self.multinomial = AliasMethod(torch.ones(memory_size-1))
        self.xModal = xModal
        self.wModal = wModal
        self.distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.distributed else 0

        self.init_memory(memory_size, embedding_dim)

    def forward(self, video_emb, audio_emb, y):
        K = int(self.num_negatives)

        # Normalize embeddings
        bs, dim = video_emb.shape
        video_emb = nn.functional.normalize(video_emb, p=2, dim=1).view(bs, dim, 1)
        audio_emb = nn.functional.normalize(audio_emb, p=2, dim=1).view(bs, dim, 1)

        # Sample memories
        with torch.no_grad():
            video_pos_mem = self.view1_mem[y].view(bs, 1, dim)
            audio_pos_mem = self.view2_mem[y].view(bs, 1, dim)

            idx = self.random_sampling(y, num_negatives=K).to(video_emb.device)
            video_neg_mem = self.view1_mem[idx].view(bs, K, dim)
            audio_neg_mem = self.view2_mem[idx].view(bs, K, dim)

        # Compute scores
        def compute_scores(context_emb, target_embs, T):
            return [torch.bmm(trg, context_emb).squeeze(-1) / T for trg in target_embs]

        scores = {}
        if self.xModal:
            scores['v2a'] = compute_scores(video_emb, [audio_pos_mem, audio_neg_mem], self.temperature)
            scores['a2v'] = compute_scores(audio_emb, [video_pos_mem, video_neg_mem], self.temperature)

        if self.wModal:
            scores['v2v'] = compute_scores(video_emb, [video_pos_mem, video_neg_mem], self.temperature)
            scores['a2a'] = compute_scores(audio_emb, [audio_pos_mem, audio_neg_mem], self.temperature)

        # Update memory bank
        self.update_memory(video_emb.squeeze(-1), audio_emb.squeeze(-1), y)

        # Compute scores for tensorboard
        with torch.no_grad():
            v2a = compute_scores(video_emb, [audio_pos_mem, audio_neg_mem], 1.)
            a2v = compute_scores(audio_emb, [video_pos_mem, video_neg_mem], 1.)
            v2v = compute_scores(video_emb, [video_pos_mem, video_neg_mem], 1.)
            a2a = compute_scores(audio_emb, [audio_pos_mem, audio_neg_mem], 1.)
            loss_debug = {
                'Scores/V2A/Pos': v2a[0].mean(),
                'Scores/V2A/Neg': v2a[1].mean(),
                'Scores/A2V/Pos': a2v[0].mean(),
                'Scores/A2V/Neg': a2v[1].mean(),
                'Scores/V2V/Pos': v2v[0].mean(),
                'Scores/V2V/Neg': v2v[1].mean(),
                'Scores/A2A/Pos': a2a[0].mean(),
                'Scores/A2A/Neg': a2a[1].mean(),
            }
        return scores, loss_debug

    def random_sampling(self, y, num_negatives):
        bs = y.shape[0]
        idx = self.multinomial.draw(bs * num_negatives).view(bs, -1).to(y.device)
        idx = idx + (idx >= y.unsqueeze(1)).long() # Avoid self
        return idx

    def init_memory(self, num_items, embedding_dim):
        self.register_buffer('view1_mem', torch.randn(num_items, embedding_dim))
        self.register_buffer('view2_mem', torch.randn(num_items, embedding_dim))

        self.view1_mem = nn.functional.normalize(self.view1_mem, p=2, dim=1)
        sample1_norm = self.view1_mem[:10].norm(dim=1).mean()
        self.view1_mem = self.view1_mem.cuda(self.device)

        self.view2_mem = nn.functional.normalize(self.view2_mem, p=2, dim=1)
        sample2_norm = self.view2_mem[:10].norm(dim=1).mean()
        self.view2_mem = self.view2_mem.cuda(self.device)

        if self.distributed:
            dist.broadcast(self.view1_mem, 0)
            dist.broadcast(self.view2_mem, 0)
            dist.barrier()

        mem_info = f"Init memory\n" \
            f"View1: {self.view1_mem.shape}; norm: {sample1_norm}\n" \
            f"View2: {self.view2_mem.shape}; norm: {sample2_norm}"
        print(f"Rank: {self.rank} - {mem_info}")

    def update_memory(self, video_emb, audio_emb, y):
        video_mom = float(self.momentum[0])
        audio_mom = float(self.momentum[1])
        # gather embeddings from all gpus
        if self.distributed:
            video_emb_gathered = _gather_from_all(video_emb)
            audio_emb_gathered = _gather_from_all(audio_emb)
            y_gathered = _gather_from_all(y)
        else:
            video_emb_gathered = video_emb
            audio_emb_gathered = audio_emb
            y_gathered = y

        # update memory
        with torch.no_grad():
            l1_pos = torch.index_select(self.view1_mem, 0, y_gathered.view(-1))
            l1_pos.mul_(video_mom)
            l1_pos.add_(torch.mul(video_emb_gathered, 1 - video_mom))
            updated_l1 = nn.functional.normalize(l1_pos, p=2, dim=1)
            self.view1_mem.index_copy_(0, y_gathered, updated_l1)

            l2_pos = torch.index_select(self.view2_mem, 0, y_gathered.view(-1))
            l2_pos.mul_(audio_mom)
            l2_pos.add_(torch.mul(audio_emb_gathered, 1 - audio_mom))
            updated_l2 = nn.functional.normalize(l2_pos, p=2, dim=1)
            self.view2_mem.index_copy_(0, y_gathered, updated_l2)

    def __repr__(self):
        num_negatives = int(self.num_negatives)
        view1_mom = float(self.momentum[0])
        view2_mom = float(self.momentum[1])
        repr_dict = {
            'name': self._get_name(),
            'num_negatives': num_negatives,
            'momentum': [view1_mom, view2_mom],
            'view1_buffer_size': self.view1_mem.shape,
            'view2_buffer_size': self.view2_mem.shape,
        }
        return pprint.pformat(repr_dict, indent=2)


class AVID(nn.Module):
    def __init__(self, num_data, embedding_dim,
                 num_negatives=4096,
                 momentum=0.9,
                 loss_type='nce',
                 xModal=True,
                 wModal=False,
                 loss_coeff=(1., 0.),
                 device=0):
        super(AVID, self).__init__()

        self.nce_average = AVIDSimilarityMemoryBank(
            memory_size=num_data,
            embedding_dim=embedding_dim,
            num_negatives=num_negatives,
            momentum=momentum,
            xModal=xModal,
            wModal=wModal,
            device=device
        )
        self.nce_average = self.nce_average.cuda(device)

        self.xModal = xModal
        self.wModal = wModal
        self.loss_type = loss_type
        self.loss_coeff = [c/sum(loss_coeff) for c in loss_coeff]
        self.criterion = NCECriterion(num_data)

    def forward(self, emb1, emb2, target):
        loss_debug_all = {}

        # Compare output embeddings to memory bank embeddings and get scores
        # scores given as: {task: [scores_positives, scores_negatives]}
        scores, loss_debug = self.nce_average(emb1, emb2, target)
        for mt in loss_debug:
            loss_debug_all[mt] = loss_debug[mt]

        # Compute loss
        xModal_loss = torch.tensor([0.], device=emb1.device)
        wModal_loss = torch.tensor([0.], device=emb1.device)
        for k in scores:
            loss, loss_debug = self.criterion(*scores[k])
            if k in {'v2a', 'a2v'}:
                xModal_loss += loss / 2. * self.loss_coeff[0]
            elif k in {'v2v', 'a2a'}:
                wModal_loss += loss / 2. * self.loss_coeff[1]

            # Tensorboard metrics
            with torch.no_grad():
                loss_debug_all[f'Loss/{k}'] = loss
                for mt in loss_debug:
                    loss_debug_all[f'{k}/{mt}'] = loss_debug[mt]

        # Tensorboard metrics
        loss_debug_all['xModal/Avg'] = xModal_loss
        loss_debug_all['wModal/Avg'] = wModal_loss

        # Final loss
        total_loss = xModal_loss + wModal_loss
        return total_loss, loss_debug_all

    def set_epoch(self, epoch):
        pass

    def __repr__(self):
        repr_dict = {
            'loss_type': self.loss_type,
            'nce_average': self.nce_average,
        }
        return pprint.pformat(repr_dict, indent=2)

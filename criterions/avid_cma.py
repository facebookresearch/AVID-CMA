import multiprocessing as mp
mp.set_start_method('spawn', force=True)

import torch
from torch import nn
import torch.distributed as dist
import numpy as np
import pprint
from utils.main_utils import AliasMethod, _gather_from_all

__all__ = ['AVID_CMA']


class CMASampler:
    def __init__(self, video_mem, audio_mem, sampling_args):
        self.video_mem = video_mem
        self.audio_mem = audio_mem
        self.sampling_args = sampling_args

    def sample_instance(self, gpu, q_job, q_data):
        video_mem = self.video_mem.cuda(gpu)
        audio_mem = self.audio_mem.cuda(gpu)

        while True:
            query_idx = q_job.get()
            if query_idx is None:
                break

            # Compute video and audio cosine similarities
            video_sim = torch.mm(video_mem, video_mem[query_idx].t())
            audio_sim = torch.mm(audio_mem, audio_mem[query_idx].t())

            # Compute agreement score
            if self.sampling_args['type'] == 'consensus':
                similarity = torch.stack([video_sim, audio_sim], 0).min(dim=0)[0]
            elif self.sampling_args['type'] == 'union':
                similarity = torch.stack([video_sim, audio_sim], 0).max(dim=0)[0]
            elif self.sampling_args['type'] == 'video':
                similarity = video_sim
            elif self.sampling_args['type'] == 'audio':
                similarity = audio_sim
            else:
                raise ValueError

            # Return top-k correspondences
            pos_sim, pos_idx = torch.topk(similarity, self.sampling_args['pos_k']+1, dim=0, sorted=True)
            pos_index = pos_idx[1:].t().cpu().numpy()   # Avoid self
            pos_index = np.sort(pos_index, axis=1)      # Sort indexes so that negative sampling can be done efficiently

            q_data.put((query_idx, pos_index))
        q_data.put((None, None))

    def sample_dispatcher(self, q_job, workers=80):
        num_items = self.video_mem.shape[0]
        batch_size = 64
        for i in range(0, num_items, batch_size):
            query_idx = list(range(i, min(i+batch_size, num_items)))
            q_job.put(query_idx)

        for _ in range(workers):
            q_job.put(None)

    def sample_gather(self, q_data, workers=80):
        num_items = self.video_mem.shape[0]
        positive_index = np.zeros((num_items, self.sampling_args['pos_k'])).astype(int)
        workers_done, done = 0, 0
        while workers_done < workers:
            query_idx, pos_idx = q_data.get()
            if query_idx is None:
                workers_done += 1
            else:
                done += pos_idx.shape[0]
                positive_index[query_idx] = pos_idx
                if done % (64*1000) == 0:
                    print(f"Done {done}/{num_items}")
        return positive_index

    def sample(self):
        # Compute CMA correspondences. Runs on only one process. Distributes work over all gpus.
        num_workers = torch.cuda.device_count()
        q_job = mp.Queue(maxsize=1000)
        q_data = mp.Queue(maxsize=1000)

        # Start job launcher
        disp = mp.Process(target=self.sample_dispatcher, args=(q_job, num_workers), daemon=True)
        disp.start()

        # Start workers
        workers = []
        for gpu in range(num_workers):
            w = mp.Process(target=self.sample_instance, args=(gpu, q_job, q_data), daemon=True)
            w.start()
            workers += [w]

        # Gather results from workers
        pos_index = self.sample_gather(q_data, num_workers)

        # Wait for all jobs to finish
        [w.join() for w in workers]
        disp.join()
        return pos_index


class AVIDSimilarityPositiveExpansion(nn.Module):
    def __init__(self,
                 memory_size,
                 embedding_dim,
                 xModal=True,
                 wModal=False,
                 num_negatives=1024,
                 num_positives=8,
                 num_negatives_within=None,
                 sampling_args=None,
                 momentum=0.5,
                 device=0,
                 ):
        super(AVIDSimilarityPositiveExpansion, self).__init__()
        self.num_negatives = num_negatives
        self.num_positives = num_positives
        self.num_negatives_within = num_negatives_within
        self.sampling_args = sampling_args
        self.temperature = 0.07
        if not isinstance(momentum, (list, tuple)):
            momentum = [momentum]*2
        self.momentum = momentum
        self.device = device

        self.pos_sampler = AliasMethod(torch.ones(sampling_args['pos_k']))
        self.neg_sampler = AliasMethod(torch.ones(memory_size-sampling_args['pos_k']))
        self.xModal = xModal
        self.wModal = wModal
        self.distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.distributed else 0

        self.init_memory(memory_size, embedding_dim)

    def forward(self, video_emb, audio_emb, y):
        bs, dim = video_emb.shape
        video_emb = nn.functional.normalize(video_emb, p=2, dim=1).view(bs, dim, 1)
        audio_emb = nn.functional.normalize(audio_emb, p=2, dim=1).view(bs, dim, 1)

        # Sample memories
        with torch.no_grad():
            video_self_mem = self.view1_mem[y].view(bs, 1, dim)
            audio_self_mem = self.view2_mem[y].view(bs, 1, dim)

            pos_idx, neg_idx = self.memory_sampling(y)
            if pos_idx is None:
                pos_idx = y.unsqueeze(-1)
            video_pos_mem = self.view1_mem[pos_idx]
            audio_pos_mem = self.view2_mem[pos_idx]
            video_neg_mem = self.view1_mem[neg_idx]
            audio_neg_mem = self.view2_mem[neg_idx]

        # Compute scores
        def compute_scores(context_emb, target_embs, T):
            return [torch.bmm(trg, context_emb).squeeze(-1) / T for trg in target_embs]

        scores = {}
        if self.xModal: # Cross-modal discrimination
            # Positives: Use own and positive memories
            xm_video_pos_mem = torch.cat([video_self_mem, video_pos_mem], 1)
            xm_audio_pos_mem = torch.cat([audio_self_mem, audio_pos_mem], 1)
            scores['v2a'] = compute_scores(video_emb, [xm_audio_pos_mem, audio_neg_mem], self.temperature)
            scores['a2v'] = compute_scores(audio_emb, [xm_video_pos_mem, video_neg_mem], self.temperature)

        if self.wModal: # Within-modal discrimination
            # Positives: Use positive set only
            wm_video_pos_mem, wm_audio_pos_mem = video_pos_mem, audio_pos_mem
            # Negatives: Potentially reduce number of negatives for within-modal discrimination
            wm_video_neg_mem, wm_audio_neg_mem = video_neg_mem, audio_neg_mem
            if self.num_negatives_within is not None:
                wm_video_neg_mem = video_neg_mem[:, :self.num_negatives_within]
                wm_audio_neg_mem = audio_neg_mem[:, :self.num_negatives_within]
            scores['v2v'] = compute_scores(video_emb, [wm_video_pos_mem, wm_video_neg_mem], self.temperature)
            scores['a2a'] = compute_scores(audio_emb, [wm_audio_pos_mem, wm_audio_neg_mem], self.temperature)

        # Update memory
        self.update_memory(video_emb.squeeze(-1), audio_emb.squeeze(-1), y)

        # Scores for tensorboard
        with torch.no_grad():
            loss_debug = {
                'Scores/V2A/Pos': compute_scores(video_emb, [audio_pos_mem], 1.)[0].mean(),
                'Scores/V2A/Neg': compute_scores(video_emb, [audio_neg_mem], 1.)[0].mean(),
                'Scores/A2V/Pos': compute_scores(audio_emb, [video_pos_mem], 1.)[0].mean(),
                'Scores/A2V/Neg': compute_scores(audio_emb, [video_neg_mem], 1.)[0].mean(),
                'Scores/V2V/Pos': compute_scores(video_emb, [video_pos_mem], 1.)[0].mean(),
                'Scores/V2V/Neg': compute_scores(video_emb, [video_neg_mem], 1.)[0].mean(),
                'Scores/A2A/Pos': compute_scores(audio_emb, [audio_pos_mem], 1.)[0].mean(),
                'Scores/A2A/Neg': compute_scores(audio_emb, [audio_neg_mem], 1.)[0].mean(),
            }
        return scores, loss_debug

    def memory_sampling(self, y):
        # Draw positives
        positive_indexes = None
        if self.num_positives > 0:
            resmp_idx =  self.pos_sampler.draw(y.shape[0] * self.num_positives).view(y.shape[0], -1).to(y.device)
            positive_indexes = self.positive_set[y].gather(1, resmp_idx).long()

        # Draw negatives
        bs = y.shape[0]
        rand_idx = self.neg_sampler.draw(bs * self.num_negatives).view(bs, -1).to(y.device)
        # Avoid positives while sampling negatives (Positive list is sorted.)
        if positive_indexes is not None:
            pos_idx = self.positive_set[y].long()
            ref = pos_idx - torch.range(0, pos_idx.shape[1]-1, dtype=pos_idx.dtype).to(pos_idx.device).unsqueeze(0)
            negative_indexes = rand_idx + (rand_idx.unsqueeze(2) >= ref.unsqueeze(1)).sum(2)
        else:
            negative_indexes = rand_idx

        return positive_indexes, negative_indexes

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
        if self.distributed:
            # gather all embeddings
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

    def find_correspondences(self):
        if self.sampling_args['pos_k'] <= 0:
            return

        # Find CMA correspondences. Only do this on one process if running in distributed mode and sync at the end.
        positive_set = np.zeros((self.view1_mem.shape[0], self.sampling_args['pos_k'])).astype(int)
        if not self.distributed or self.distributed and self.rank == 0:
            positive_set = CMASampler(self.view1_mem, self.view2_mem, self.sampling_args).sample()

        # Find CMA correspondences. Only do this on one process if running in distributed mode and sync at the end.
        if positive_set is not None:
            self.register_buffer('positive_set', torch.from_numpy(positive_set).int())
            self.positive_set = self.positive_set.cuda(self.device)
            if self.distributed:
                dist.broadcast(self.positive_set, 0)

        if self.distributed:
            dist.barrier()

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


class AVID_CMA(nn.Module):
    def __init__(self, num_data, embedding_dim,
                 num_negatives=1024,
                 num_positives=8,
                 num_negatives_within=None,
                 momentum=0.5,
                 loss_type='nce',
                 xModal=True,
                 wModal=False,
                 xModalCoeff=1.,
                 wModalCoeff=1.,
                 sampling_args=None,
                 checkpoint=None,
                 resample_freq=-1,
                 device=0):
        super(AVID_CMA, self).__init__()

        # first setup the NCEAverage method to get the scores of the output wrt. memory bank negatives
        self.nce_average = AVIDSimilarityPositiveExpansion(
            memory_size=num_data,
            embedding_dim=embedding_dim,
            num_negatives=num_negatives,
            num_positives=num_positives,
            num_negatives_within=num_negatives_within,
            momentum=momentum,
            xModal=xModal,
            wModal=wModal,
            sampling_args=sampling_args,
            device=device
        )
        self.nce_average = self.nce_average.cuda(device)

        self.loss_type = loss_type
        self.xModal = xModal
        self.wModal = wModal
        self.xModalCoeff = xModalCoeff
        self.wModalCoeff = wModalCoeff

        # Setup loss function
        from criterions.nce import NCECriterion
        self.criterion = NCECriterion(num_data)

        # Restore memory bank and partition function from AVID checkpoint
        # Needs to be done before finding correspondences
        if checkpoint is not None:
            ckp = torch.load(checkpoint, map_location='cpu')['train_criterion']
            state_dict = self.state_dict()
            # Restore memory banks
            state_dict['nce_average.view1_mem'] = ckp['nce_average.view1_mem']
            state_dict['nce_average.view2_mem'] = ckp['nce_average.view2_mem']
            # Restore partition function
            Z = torch.stack([ckp[k] for k in ckp if 'avg_exp_score' in k]).mean()
            for k in state_dict:
                if 'avg_exp_score' in k:
                    state_dict[k] = Z
            self.load_state_dict(state_dict)

        # Find CMA correspondences
        self.resample_freq = resample_freq
        self.nce_average.find_correspondences()

    def forward(self, emb1, emb2, target):
        loss_debug_all = {}

        # Compare output embeddings to memory bank embeddings and get scores
        scores, loss_debug = self.nce_average(emb1, emb2, target)
        for mt in loss_debug:
            loss_debug_all[mt] = loss_debug[mt]

        # Compute cross/within modal discrimination losses
        xModal_loss = torch.tensor([0.], device=emb1.device)
        wModal_loss = torch.tensor([0.], device=emb1.device)
        for k in scores:
            loss, loss_debug = self.criterion(*scores[k])
            if k in {'v2a', 'a2v'}:
                xModal_loss += loss / 2. * self.xModalCoeff
            if k in {'v2v', 'a2a'}:
                wModal_loss += loss / 2. * self.wModalCoeff

            # Metrics for tensorboard
            with torch.no_grad():
                loss_debug_all[f'Loss/{k}'] = loss
                for mt in loss_debug:
                    loss_debug_all[f'{k}/{mt}'] = loss_debug[mt]

        # Metrics for tensorboard
        loss_debug_all['xModal/Avg'] = xModal_loss
        loss_debug_all['wModal/Avg'] = wModal_loss

        # Compute final loss
        total_loss = xModal_loss + wModal_loss
        return total_loss, loss_debug_all

    def set_epoch(self, epoch):
        # Recompute CMA correspondences every resample_freq epochs
        if self.resample_freq > 0 and epoch > 0 and epoch % self.resample_freq == 0:
            self.nce_average.find_correspondences()

    def __repr__(self):
        repr_dict = {
            'loss_type': self.loss_type,
            'nce_average': self.nce_average,
        }
        return pprint.pformat(repr_dict, indent=2)


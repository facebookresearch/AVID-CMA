import torch
from torch import nn
from utils.main_utils import AliasMethod

__all__ = ['L3Criterion']


class L3Criterion(nn.Module):
    def __init__(self, num_data, embedding_dim, filters, device):
        super(L3Criterion, self).__init__()

        # first setup the NCEAverage method to get the scores of the output wrt. memory bank negatives
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim*2, filters[0]),
            nn.ReLU(inplace=True),
            nn.Linear(filters[0], 1),
        ).cuda(device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.sampler = None

    def sample_negative_pairs(self, emb1, emb2):
        bs = emb1.shape[0]
        if self.sampler is None:
            self.sampler = AliasMethod(torch.ones(bs - 1))

        idx = self.sampler.draw(bs)
        idx = (idx >= torch.range(0, bs-1).long()).long() + idx
        return emb1, emb2[idx]

    def set_epoch(self, epoch):
        pass


    def forward(self, emb1, emb2, target):
        bs = emb1.shape[0]
        emb1_neg, emb2_neg = self.sample_negative_pairs(emb1, emb2)
        embs_pos = torch.cat([emb1, emb2], 1)
        embs_neg = torch.cat([emb1_neg, emb2_neg], 1)
        emb_all = torch.cat([embs_pos, embs_neg], 0)
        y = torch.cat([torch.ones(bs), torch.zeros(bs)], 0).to(emb_all.device)

        pred = self.classifier(emb_all).squeeze(-1)
        loss = self.criterion(pred, y)
        loss_debug_all = {
            'xModal/Avg': loss,
            'wModal/Avg': torch.tensor([0.])}
        return loss, loss_debug_all

    def __repr__(self):
        from utils import main_utils
        return main_utils.parameter_description(self)


def main():
    # import torch.multiprocessing as mp
    # ngpus_per_node = torch.cuda.device_count()
    # mp.spawn(main_worker_distributed, nprocs=ngpus_per_node, args=(ngpus_per_node, ))
    main_worker_distributed(0, 1)


def main_worker_distributed(gpu, ngpus_per_node):
    # dist.init_process_group(backend='nccl', init_method='tcp://localhost:1234', world_size=ngpus_per_node, rank=gpu)
    loss = L3Criterion(10000, 128).cuda(gpu)
    print(loss)

    x1 = torch.randn((32, 128)).cuda(gpu)
    x2 = torch.randn((32, 128)).cuda(gpu)
    y = torch.randint(10000, (32,)).cuda(gpu)
    l = loss(x1, x2, y)
    print(l)


if __name__ == '__main__':
    main()

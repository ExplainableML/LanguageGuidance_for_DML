import torch

import batchminer
import criteria

ALLOWED_MINING_OPS = None
REQUIRES_BATCHMINER = False
REQUIRES_OPTIM = True


class Criterion(torch.nn.Module):
    def __init__(self, opt):
        """
        Module ProxyAnchor/NCA loss.
        """
        super(Criterion, self).__init__()

        self.opt = opt
        self.num_proxies = opt.n_classes
        self.embed_dim = opt.embed_dim

        self.pars = {
            'pos_alpha': opt.loss_oproxy_pos_alpha,
            'pos_delta': opt.loss_oproxy_pos_delta,
            'neg_alpha': opt.loss_oproxy_neg_alpha,
            'neg_delta': opt.loss_oproxy_neg_delta
        }

        self.mode = opt.loss_oproxy_mode
        self.name = 'proxynca' if self.mode == 'nca' else 'proxyanchor'
        self.dim = 1 if self.mode == 'nca' else 0

        self.class_idxs = torch.arange(self.num_proxies)
        self.proxies = torch.randn(self.num_proxies, self.embed_dim) / 8
        self.proxies = torch.nn.Parameter(self.proxies)
        self.optim_dict_list = [{
            'params': self.proxies,
            'lr': opt.lr * opt.loss_oproxy_lrmulti
        }]

        self.ALLOWED_MINING_OPS = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM = REQUIRES_OPTIM

    def forward(self, batch, labels, **kwargs):
        """
        Args:
            batch:   torch.Tensor: Input of embeddings with size (BS x DIM)
            labels: nparray/list: For each element of the batch assigns a class [0,...,C-1], shape: (BS x 1)
        """
        bs = len(batch)
        batch = torch.nn.functional.normalize(batch, dim=-1)

        self.labels = labels.unsqueeze(1)
        self.u_labels = self.labels.view(-1)
        self.same_labels = (self.labels.T == self.labels.view(-1, 1)).to(
            batch.device).T
        self.diff_labels = (self.class_idxs.unsqueeze(1) != self.labels.T).to(
            torch.float).to(batch.device).T

        return self.compute_proxyloss(batch)

    def compute_proxyloss(self, batch):
        proxies = self.prep(self.proxies)
        pos_sims = batch.mm(proxies[self.u_labels].T)
        sims = batch.mm(proxies.T)
        w_pos_sims = -self.pars['pos_alpha'] * (pos_sims -
                                                self.pars['pos_delta'])
        w_neg_sims = self.pars['neg_alpha'] * (sims - self.pars['neg_delta'])
        pos_s = self.masked_logsumexp(w_pos_sims,
                                      mask=self.same_labels.type(torch.bool),
                                      dim=self.dim)
        neg_s = self.masked_logsumexp(w_neg_sims,
                                      mask=self.diff_labels.type(torch.bool),
                                      dim=self.dim)
        return pos_s.mean() + neg_s.mean()

    def masked_logsumexp(self, sims, dim=0, mask=None):
        # Adapted from https://github.com/KevinMusgrave/pytorch-metric-learning/\
        # blob/master/src/pytorch_metric_learning/utils/loss_and_miner_utils.py.
        if mask is not None:
            sims = sims.masked_fill(~mask, torch.finfo(sims.dtype).min)
        dims = list(sims.shape)
        dims[dim] = 1
        zeros = torch.zeros(dims, dtype=sims.dtype, device=sims.device)
        sims = torch.cat([sims, zeros], dim=dim)
        logsumexp_sims = torch.logsumexp(sims, dim=dim, keepdim=True)
        if mask is not None:
            logsumexp_sims = logsumexp_sims.masked_fill(
                ~torch.any(mask, dim=dim, keepdim=True), 0)
        return logsumexp_sims

import numpy as np
import torch

import batchminer

ALLOWED_MINING_OPS = None
REQUIRES_BATCHMINER = False
REQUIRES_OPTIM = False


class Criterion(torch.nn.Module):
    def __init__(self, opt, **kwargs):
        """
        Multisimilarity Loss.
        """
        super(Criterion, self).__init__()
        self.pars = opt

        self.n_classes = opt.n_classes
        self.pos_weight = opt.loss_multisimilarity_pos_weight
        self.neg_weight = opt.loss_multisimilarity_neg_weight
        self.margin = opt.loss_multisimilarity_margin
        self.pos_thresh = opt.loss_multisimilarity_pos_thresh
        self.neg_thresh = opt.loss_multisimilarity_neg_thresh
        self.name = 'multisimilarity'

        self.dim = opt.loss_multisimilarity_dim
        self.lr = opt.lr

        ####
        self.ALLOWED_MINING_OPS = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM = REQUIRES_OPTIM

    def forward(self, batch, labels, **kwargs):
        """
        Args:
            batch: torch.Tensor: Input of embeddings with size (BS x DIM)
            labels: nparray/list: For each element of the batch assigns a
                    class [0,...,C-1], shape: (BS x 1)
        """
        batch_sims = self.smat(batch, batch)

        labels = labels.unsqueeze(1)
        bsame_labels = (labels.T == labels.view(-1, 1)).to(batch.device).T
        bdiff_labels = (labels.T != labels.view(-1, 1)).to(batch.device).T

        w_pos_sims = -self.pos_weight * (batch_sims - self.pos_thresh)
        w_neg_sims = self.neg_weight * (batch_sims - self.neg_thresh)

        pos_mask, neg_mask = self.sample_mask(batch_sims,
                                              bsame_labels,
                                              bdiff_labels,
                                              self.margin,
                                              dim=self.dim)
        pos_s = self.masked_logsumexp(w_pos_sims, mask=pos_mask, dim=self.dim)
        neg_s = self.masked_logsumexp(w_neg_sims, mask=neg_mask, dim=self.dim)

        ###
        return 1. / np.abs(self.pos_weight) * torch.nn.Softplus()(pos_s.mean(
        )) + 1. / np.abs(self.neg_weight) * torch.nn.Softplus()(neg_s.mean())

    ###
    def sample_mask(self, sims, pos_label_mat, neg_label_mat, margin, dim=-1):
        pos_label_mat = pos_label_mat.clone()
        pos_label_mat[torch.eye(len(sims), dtype=torch.bool)] = False

        pos_bounds = sims.clone().masked_fill(
            ~pos_label_mat,
            torch.finfo(sims.dtype).max).min(dim=dim).values
        neg_bounds = sims.clone().masked_fill(
            ~neg_label_mat,
            torch.finfo(sims.dtype).min).max(dim=dim).values
        pos_bounds = pos_bounds.unsqueeze(dim)
        neg_bounds = neg_bounds.unsqueeze(dim)

        pos_mask = pos_label_mat * ((sims - margin) < neg_bounds)
        neg_mask = neg_label_mat * ((sims + margin) > pos_bounds)
        return pos_mask, neg_mask

    ###
    def smat(self, A, B, mode='cosine'):
        if mode == 'cosine':
            return A.mm(B.T)
        elif mode == 'euclidean':
            return (A.mm(A.T).diag().unsqueeze(-1) +
                    B.mm(B.T).diag().unsqueeze(0) -
                    2 * A.mm(B.T)).clamp(min=1e-20).sqrt()

    ###
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

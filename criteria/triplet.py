import torch

import batchminer

ALLOWED_MINING_OPS = list(batchminer.BATCHMINING_METHODS.keys())
REQUIRES_BATCHMINER = True
REQUIRES_OPTIM = False


class Criterion(torch.nn.Module):
    def __init__(self, opt, batchminer):
        """
        Triplet Loss.
        """
        super(Criterion, self).__init__()
        self.margin = opt.loss_triplet_margin
        self.batchminer = batchminer
        self.name = 'triplet'

        ####
        self.ALLOWED_MINING_OPS = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM = REQUIRES_OPTIM

    def triplet_distance(self, anchor, positive, negative):
        return torch.nn.functional.relu((anchor - positive).pow(2).sum() -
                                        (anchor - negative).pow(2).sum() +
                                        self.margin)

    def forward(self, batch, labels, distances=None, **kwargs):
        """
        Args:
            batch:   torch.Tensor: Input of embeddings with size (BS x DIM)
            labels:  nparray/list: For each element of the batch assigns a
                     class [0,...,C-1], shape: (BS x 1)
        """
        if isinstance(labels, torch.Tensor): labels = labels.cpu().numpy()
        if isinstance(batch, list):
            assert len(
                batch
            ) == 3, 'Need three batches for explicit triplet construction.'
            sampled_triplets = self.batchminer(batch[0],
                                               labels,
                                               distances=distances)
        else:
            sampled_triplets = self.batchminer(batch,
                                               labels,
                                               distances=distances)

        if isinstance(batch, list):
            loss = torch.stack([
                self.triplet_distance(batch[0][triplet[0], :],
                                      batch[1][triplet[1], :],
                                      batch[2][triplet[2], :])
                for triplet in sampled_triplets
            ])
        else:
            loss = torch.stack([
                self.triplet_distance(batch[triplet[0], :],
                                      batch[triplet[1], :],
                                      batch[triplet[2], :])
                for triplet in sampled_triplets
            ])

        return torch.mean(loss)

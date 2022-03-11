import numpy as np
from scipy.spatial import distance
from scipy.stats import entropy
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import torch


class Metric():
    def __init__(self, embed_dim, mode, feature_type, **kwargs):
        self.mode = mode
        self.embed_dim = embed_dim
        self.requires = [feature_type]
        self.name = 'rho_spectrum@' + str(mode) + '@' + feature_type

    def __call__(self, embeds=None, avg_features=None):
        if avg_features is not None:
            embeds = avg_features

        if isinstance(embeds, torch.Tensor):
            _, s, _ = torch.svd(embeds)
            s = s.cpu().numpy()
        else:
            #Features need to be clipped due to maximum histogram length for W&B of 512
            svd = TruncatedSVD(n_components=np.clip(
                np.clip(self.embed_dim - 1, None, embeds.shape[-1] - 1), None,
                511),
                               n_iter=7,
                               random_state=42)
            svd.fit(embeds)
            s = svd.singular_values_

        s = s[np.abs(self.mode) - 1:]
        s_norm = s / np.sum(s)
        uniform = np.ones(len(s)) / (len(s))
        kl = entropy(uniform, s_norm)

        return kl

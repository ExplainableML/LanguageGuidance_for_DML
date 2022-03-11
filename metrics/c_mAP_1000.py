import faiss
import numpy as np
import torch


class Metric():
    def __init__(self, **kwargs):
        #For all benchmarks, there is really no purpose to go beyond a recall of 1000.
        #In addition, faiss on gpu only supports k up to 1024.
        self.requires = ['cos_nearest_features@1000', 'target_labels']
        self.name = 'mAP_1000'

    def __call__(self, target_labels, k_closest_classes_cos):
        R = 1000
        labels, freqs = np.unique(target_labels, return_counts=True)
        avg_r_precisions = []

        for label, freq in zip(labels, freqs):
            rows_with_label = np.where(target_labels == label)[0]
            for row in rows_with_label:
                n_recalled_samples = np.arange(1, R + 1)
                target_label_occ_in_row = k_closest_classes_cos[
                    row, :] == label
                cumsum_target_label_freq_row = np.cumsum(
                    target_label_occ_in_row)
                avg_r_pr_row = np.sum(
                    cumsum_target_label_freq_row * target_label_occ_in_row /
                    n_recalled_samples) / freq
                avg_r_precisions.append(avg_r_pr_row)

        return np.mean(avg_r_precisions)

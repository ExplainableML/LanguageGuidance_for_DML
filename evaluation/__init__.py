import os

import faiss
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch


def evaluate(dataset,
             LOG,
             metric_computer,
             dataloaders,
             model,
             opt,
             evaltypes,
             device,
             aux_store=None,
             make_recall_plot=False,
             store_checkpoints=True,
             log_key='Test',
             compute_metrics_only=False,
             print_text=True):
    """
    Parent-Function to compute evaluation metrics, print summary string and
    store checkpoint files/plot sample recall plots.
    """
    computed_metrics, extra_infos = metric_computer.compute_standard(
        opt, model, dataloaders[0], evaltypes, device, mode=log_key)

    numeric_metrics = {}
    histogr_metrics = {}
    for main_key in computed_metrics.keys():
        for name, value in computed_metrics[main_key].items():
            if isinstance(value, np.ndarray):
                if main_key not in histogr_metrics:
                    histogr_metrics[main_key] = {}
                histogr_metrics[main_key][name] = value
            else:
                if main_key not in numeric_metrics:
                    numeric_metrics[main_key] = {}
                numeric_metrics[main_key][name] = value

    ###
    full_result_str = ''
    for evaltype in numeric_metrics.keys():
        full_result_str += 'Out-Type: {}:\n'.format(evaltype)
        for i, (metricname,
                metricval) in enumerate(numeric_metrics[evaltype].items()):
            full_result_str += '{0}{1}: {2:4.4f}'.format(
                ' | ' if i > 0 else '', metricname, metricval)
        full_result_str += '\n'

    if print_text:
        print(full_result_str)

    if not compute_metrics_only:
        ### Log Histogram-Style data with W&Bs
        if opt.log_online:
            for evaltype in histogr_metrics.keys():
                for eval_metric, hist in histogr_metrics[evaltype].items():
                    import wandb, numpy
                    hist = hist[:500]
                    wandb.log(
                        {
                            log_key + ': ' + evaltype + '_{}'.format(eval_metric):
                            wandb.Histogram(
                                np_histogram=(list(hist),
                                              list(np.arange(len(hist) + 1))))
                        },
                        step=opt.epoch)
                    wandb.log(
                        {
                            log_key + ': ' + evaltype + '_LOG-{}'.format(eval_metric):
                            wandb.Histogram(
                                np_histogram=(list(np.log(hist) + 20),
                                              list(np.arange(len(hist) + 1))))
                        },
                        step=opt.epoch)

        ###
        for evaltype in numeric_metrics.keys():
            for eval_metric in numeric_metrics[evaltype].keys():
                parent_metric = evaltype + '_{}'.format(
                    eval_metric.split('@')[0])
                LOG.progress_saver[log_key].log(
                    eval_metric,
                    numeric_metrics[evaltype][eval_metric],
                    group=parent_metric)

            ###
            if make_recall_plot:
                if opt.dataset != 'inshop':
                    recover_closest_standard(
                        extra_infos[evaltype]['features'],
                        extra_infos[evaltype]['image_paths'],
                        LOG.prop.save_path + '/sample_recoveries.png')
                else:
                    recover_closest_query_gallery(
                        extra_infos[evaltype]['query_features'],
                        extra_infos[evaltype]['gallery_features'],
                        extra_infos[evaltype]['gallery_image_paths'],
                        extra_infos[evaltype]['query_image_paths'],
                        LOG.prop.save_path + '/sample_recoveries.png')

        ###
        for evaltype in evaltypes:
            for storage_metric in opt.storage_metrics:
                parent_metric = evaltype + '_{}'.format(
                    storage_metric.split('@')[0])
                ref_mets = LOG.progress_saver[log_key].groups[parent_metric][
                    storage_metric]['content'][:-1]
                if not len(ref_mets): ref_mets = [-np.inf]

                if numeric_metrics[evaltype][storage_metric] > np.max(
                        ref_mets) and opt.store_improvements:
                    print('Saved improved checkpoint for {}: {}\n'.format(
                        log_key, parent_metric))
                    set_checkpoint(model,
                                   opt,
                                   LOG,
                                   LOG.prop.save_path +
                                   '/checkpoint_{}_{}_{}.pth.tar'.format(
                                       log_key, evaltype, storage_metric),
                                   aux=aux_store)
    else:
        return numeric_metrics


def set_checkpoint(model, opt, progress_saver, savepath, aux=None):
    torch.save(
        {
            'state_dict': model.state_dict(),
            'opt': opt,
            'progress': progress_saver,
            'aux': aux
        }, savepath)


def recover_closest_standard(feature_matrix_all,
                             image_paths,
                             save_path,
                             n_image_samples=10,
                             n_closest=3):
    image_paths = np.array([x[0] for x in image_paths])
    sample_idxs = np.random.choice(np.arange(len(feature_matrix_all)),
                                   n_image_samples)

    faiss_search_index = faiss.IndexFlatL2(feature_matrix_all.shape[-1])
    faiss_search_index.add(feature_matrix_all)
    _, closest_feature_idxs = faiss_search_index.search(
        feature_matrix_all, n_closest + 1)

    sample_paths = image_paths[closest_feature_idxs][sample_idxs]

    f, axes = plt.subplots(n_image_samples, n_closest + 1)
    for i, (ax, plot_path) in enumerate(
            zip(axes.reshape(-1), sample_paths.reshape(-1))):
        ax.imshow(np.array(Image.open(plot_path)))
        ax.set_xticks([])
        ax.set_yticks([])
        if i % (n_closest + 1):
            ax.axvline(x=0, color='g', linewidth=13)
        else:
            ax.axvline(x=0, color='r', linewidth=13)
    f.set_size_inches(10, 20)
    f.tight_layout()
    f.savefig(save_path)
    plt.close()

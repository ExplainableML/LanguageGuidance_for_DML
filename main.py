import argparse
import collections
import contextlib
import copy
import json
import os
import random
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import termcolor
from tqdm import tqdm

import parameters as par
import utilities.misc as misc

### ---------------------------------------------------------------
### INPUT ARGUMENTS
parser = argparse.ArgumentParser()
parser = par.basic_training_parameters(parser)
parser = par.batch_creation_parameters(parser)
parser = par.batchmining_specific_parameters(parser)
parser = par.loss_specific_parameters(parser)
parser = par.wandb_parameters(parser)
### Additional, non-default parameters.
parser = par.language_guidance_parameters(parser)
opt = parser.parse_args()

### ---------------------------------------------------------------
# The following setting is useful when logging to wandb and running multiple
# seeds per setup: By setting the savename to <group_plus_seed>, the savename
# will instead comprise the group and the seed!
if opt.savename == 'group_plus_seed':
    if opt.log_online:
        opt.savename = opt.group + '_s{}'.format(opt.seed)
    else:
        opt.savename = ''

if opt.completed:
    print('\n\nTraining Run already completed!')
    exit()

# If wandb-logging is turned on, initialize the wandb-run here:
if opt.log_online:
    import wandb
    _ = os.system('wandb login {}'.format(opt.wandb_key))
    os.environ['WANDB_API_KEY'] = opt.wandb_key
    opt.unique_run_id = wandb.util.generate_id()
    wandb.init(id=opt.unique_run_id,
               resume='allow',
               project=opt.project,
               group=opt.group,
               name=opt.savename,
               dir=opt.source_path,
               settings=wandb.Settings(start_method="fork"))
    wandb.config.update(opt)

### ---------------------------------------------------------------
# Load Remaining Libraries that neeed to be loaded after wandb
import torch, torch.nn as nn, torch.nn.functional as F
import torch.multiprocessing
import torchvision

torch.multiprocessing.set_sharing_strategy('file_system')
import architectures as archs
import datasampler as dsamplers
import datasets as dataset_library
import criteria as criteria
import metrics as metrics
import batchminer as bmine
import evaluation as eval
from utilities import misc
from utilities import logger

full_training_start_time = time.time()

opt.source_path += '/' + opt.dataset
opt.save_path += '/' + opt.dataset

# Assert that the construction of the batch makes sense, i.e. the division into
# class-subclusters.
assert_text = 'Batchsize needs to fit number of samples per class for distance '
assert_text += 'sampling and margin/triplet loss!'
assert not opt.bs % opt.samples_per_class, assert_text

opt.pretrained = not opt.not_pretrained
opt.evaluate_on_gpu = not opt.evaluate_on_cpu

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu[0])
misc.set_seed(opt.seed)

### ---------------------------------------------------------------
### Network model.
opt.device = torch.device('cuda')
model = archs.select(opt.arch, opt)

if hasattr(model, 'optim_dict_list') and len(model.optim_dict_list):
    to_optim = model.optim_dict_list
else:
    if opt.fc_lr < 0:
        to_optim = [{
            'params': model.parameters(),
            'lr': opt.lr,
            'weight_decay': opt.decay
        }]
    else:
        all_but_fc_params = [
            x[-1] for x in list(
                filter(lambda x: 'last_linear' not in x[0],
                       model.named_parameters()))
        ]
        fc_params = model.model.last_linear.parameters()
        to_optim = [{
            'params': all_but_fc_params,
            'lr': opt.lr,
            'weight_decay': opt.decay
        }, {
            'params': fc_params,
            'lr': opt.fc_lr,
            'weight_decay': opt.decay
        }]

_ = model.to(opt.device)

### Datasetse & Dataloaders.
datasets = dataset_library.select(opt.dataset, opt, opt.source_path)

dataloaders = {}
dataloaders['evaluation'] = torch.utils.data.DataLoader(
    datasets['evaluation'],
    num_workers=opt.kernels,
    batch_size=opt.bs,
    shuffle=False)
dataloaders['testing'] = torch.utils.data.DataLoader(datasets['testing'],
                                                     num_workers=opt.kernels,
                                                     batch_size=opt.bs,
                                                     shuffle=False)

if opt.use_tv_split:
    dataloaders['validation'] = torch.utils.data.DataLoader(
        datasets['validation'],
        num_workers=opt.kernels,
        batch_size=opt.bs,
        shuffle=False)

train_data_sampler = dsamplers.select(opt.data_sampler, opt,
                                      datasets['training'].image_dict,
                                      datasets['training'].image_list)
if train_data_sampler.requires_storage:
    train_data_sampler.create_storage(dataloaders['evaluation'], model,
                                      opt.device)

dataloaders['training'] = torch.utils.data.DataLoader(
    datasets['training'],
    num_workers=opt.kernels,
    batch_sampler=train_data_sampler)

opt.n_classes = len(dataloaders['training'].dataset.avail_classes)
opt.n_test_classes = len(dataloaders['testing'].dataset.avail_classes)

metric_evaluation_keys = ['testing', 'evaluation']
if opt.use_tv_split: metric_evaluation_keys.append('validation')

### Create logging setup.
sub_loggers = ['Train', 'Test', 'Model Grad']
if opt.use_tv_split: sub_loggers.append('Val')

LOG = logger.LOGGER(opt,
                    sub_loggers=sub_loggers,
                    start_new=True,
                    log_online=opt.log_online)

### Criterion.
batchminer = bmine.select(opt.batch_mining, opt)
criterion, to_optim = criteria.select(opt.loss, opt, to_optim, batchminer)

_ = criterion.to(opt.device)

### Optimizer.
if opt.optim == 'adam':
    optimizer = torch.optim.Adam(to_optim)
else:
    raise Exception('Optimizer <{}> not available!'.format(opt.optim))

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                 milestones=opt.tau,
                                                 gamma=opt.gamma)

### Metric Computer.
metric_computer = metrics.MetricComputer(opt.evaluation_metrics, opt)

### ---------------------------------------------------------------
### Summary.
data_text = 'Dataset:\t {}'.format(opt.dataset.upper())
setup_text = 'Objective:\t {}'.format(opt.loss.upper())
miner_text = 'Batchminer:\t {}'.format(
    opt.batch_mining if criterion.REQUIRES_BATCHMINER else 'N/A')
arch_text = 'Backbone:\t {} (#weights: {})'.format(opt.arch.upper(),
                                                   misc.gimme_params(model))
summary = data_text + '\n' + setup_text + '\n' + miner_text + '\n' + arch_text
print(summary)

### ---------------------------------------------------------------
### Initialize optional language guidance.
if opt.language_distill_w:
    import language_guidance
    import pretrainedmodels as ptm
    language_guide = language_guidance.LanguageGuide(
        opt.device,
        language_model=opt.language_model,
        activation_iter=opt.language_delay,
        language_shift=opt.language_shift,
        use_pseudoclasses=opt.language_pseudoclass,
        pseudoclass_topk=opt.language_pseudoclass_topk,
        distill_dir=opt.language_distill_dir,
        T=opt.language_temp)

    torch.cuda.empty_cache()
    language_guide.precompute_language_embeds(
        dataloaders['evaluation'], opt.device,
        ptm.__dict__['resnet50'](pretrained='imagenet').to(opt.device))

    del language_guide.language_model
    torch.cuda.empty_cache()

### ---------------------------------------------------------------
#### Main training.
print('\n' + termcolor.colored('-----', 'red', attrs=['bold']) + '\n')

iter_count = 0
loss_args = {
    'batch': None,
    'labels': None,
    'batch_features': None,
    'f_embed': None
}
init_lrs = [param_group['lr'] for param_group in optimizer.param_groups[1:]]

opt.epoch = 0
epochs = range(opt.epoch, opt.n_epochs)

scaler = torch.cuda.amp.GradScaler()

torch.cuda.empty_cache()
for epoch in epochs:
    opt.epoch = epoch
    if epoch < opt.warmup:
        print(termcolor.colored('-- WARMUP --', 'yellow', attrs=['bold']))

    # Set seeds for each epoch - this ensures reproducibility after resumption.
    misc.set_seed(opt.n_epochs * opt.seed + epoch)
    print('Running with learning rates {}...'.format(' | '.join(
        '{}'.format(x['lr']) for x in optimizer.param_groups)))
    epoch_start_time = time.time()

    # Train one epoch
    data_iterator = tqdm(dataloaders['training'],
                         desc='Epoch {} Training...'.format(epoch))
    loss_collect = []
    _ = model.train()

    for i, out in enumerate(data_iterator):
        optimizer.zero_grad(set_to_none=True)
        class_labels, input_dict, sample_indices = out

        context = torch.cuda.amp.autocast(
        ) if opt.use_float16 else contextlib.nullcontext()

        with context:
            input = input_dict['image'].to(opt.device)
            model_args = {
                'x': input.to(opt.device),
                'warmup': epoch < opt.warmup
            }
            out_dict = model(**model_args)
            embeds, avg_features, features = [
                out_dict[key]
                for key in ['embeds', 'avg_features', 'features']
            ]

            loss_args['input_batch'] = input
            loss_args['batch'] = embeds
            loss_args['labels'] = class_labels
            loss_args['f_embed'] = model.model.last_linear
            loss_args['batch_features'] = features
            loss_args['avg_batch_features'] = avg_features
            loss_args['model'] = model

            loss = criterion(**loss_args)

            if opt.language_distill_w:
                loss += opt.language_distill_w * language_guide.regularize(
                    embeds, class_labels, sample_indices)

        loss_collect.append(loss.item())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        iter_count += 1

        if 'resnet' in opt.arch.lower() and epoch >= opt.warmup:
            data_iterator.set_postfix_str('DML Loss: {0:.4f}'.format(
                np.mean(loss_collect)))
        else:
            data_iterator.set_postfix_str('Loss: {0:.4f}'.format(
                np.mean(loss_collect)))
        if i == len(dataloaders['training']) - 1:
            data_iterator.set_description(
                'Epoch (Train) {0}: Mean Loss [{1:.4f}]'.format(
                    epoch, np.mean(loss_collect)))

    result_metrics = {'loss': np.mean(loss_collect)}

    LOG.progress_saver['Train'].log('epochs', epoch)
    for metricname, metricval in result_metrics.items():
        LOG.progress_saver['Train'].log(metricname, metricval)
    LOG.progress_saver['Train'].log(
        'time', np.round(time.time() - epoch_start_time, 4))

    if opt.scheduler != 'none':
        scheduler.step()

    # Evaluate Metric for Training & Test (& Validation)
    _ = model.eval()
    aux_store = {
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'dataloaders': dataloaders,
        'datasets': datasets,
        'train_data_sampler': train_data_sampler
    }

    torch.cuda.empty_cache()
    if not opt.no_test_metrics:
        print('\n' + termcolor.colored(
            'Computing Testing Metrics...', 'green', attrs=['bold']))
        eval.evaluate(opt.dataset,
                      LOG,
                      metric_computer, [dataloaders['testing']],
                      model,
                      opt,
                      opt.evaltypes,
                      opt.device,
                      log_key='Test',
                      aux_store=aux_store)
    if opt.use_tv_split:
        print('\n' + termcolor.colored(
            'Computing Validation Metrics...', 'green', attrs=['bold']))
        eval.evaluate(opt.dataset,
                      LOG,
                      metric_computer, [dataloaders['validation']],
                      model,
                      opt,
                      opt.evaltypes,
                      opt.device,
                      log_key='Val',
                      aux_store=aux_store)
    if not opt.no_train_metrics:
        print('\n' + termcolor.colored(
            'Computing Training Metrics...', 'green', attrs=['bold']))
        eval.evaluate(opt.dataset,
                      LOG,
                      metric_computer, [dataloaders['evaluation']],
                      model,
                      opt,
                      opt.evaltypes,
                      opt.device,
                      log_key='Train',
                      aux_store=aux_store)

    import wandb
    LOG.update(all=True)

    print('\nTotal Epoch Runtime: {0:4.2f}s'.format(time.time() -
                                                    epoch_start_time))
    print('\n' + termcolor.colored('-----', 'red', attrs=['bold']) + '\n')

opt.completed = True
pkl.dump(opt, open(opt.save_path + "/hypa.pkl", "wb"))

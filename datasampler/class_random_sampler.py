import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm
import random



"""======================================================"""
REQUIRES_STORAGE = False

###
class Sampler(torch.utils.data.sampler.Sampler):
    """
    Plugs into PyTorch Batchsampler Package.
    """
    def __init__(self, opt, image_dict, image_list, **kwargs):
        self.pars = opt

        #####
        self.image_dict         = image_dict
        self.image_list         = image_list

        #####
        self.internal_split = opt.internal_split
        self.use_meta_split = self.internal_split!=1
        self.classes        = list(self.image_dict.keys())
        self.tv_split       = int(len(self.classes)*self.internal_split)
        self.train_classes  = self.classes[:self.tv_split]
        self.val_classes    = self.classes[self.tv_split:]

        ####
        self.batch_size         = opt.bs
        self.samples_per_class  = opt.samples_per_class
        self.sampler_length     = len(image_list)//opt.bs
        assert self.batch_size%self.samples_per_class==0, '#Samples per class must divide batchsize!'

        self.name             = 'class_random_sampler'
        self.requires_storage = False

        self.random_gen       = random.Random(opt.seed)


    def __iter__(self):
        for _ in range(self.sampler_length):
            subset = []
            ### Random Subset from Random classes
            if self.use_meta_split:
                train_draws = int((self.batch_size//self.samples_per_class)*self.internal_split)
                val_draws   = self.batch_size//self.samples_per_class-train_draws
            else:
                train_draws = self.batch_size//self.samples_per_class
                val_draws   = None

            if self.pars.data_ssl_set:
                for _ in range(train_draws//2):
                    class_key = random.choice(self.train_classes)
                    subset.extend([random.choice(self.image_dict[class_key])[-1] for _ in range(self.samples_per_class)])
                subset = subset + subset
            else:
                for _ in range(train_draws):
                    class_key = random.choice(self.train_classes)
                    class_ix_list = [random.choice(self.image_dict[class_key])[-1] for _ in range(self.samples_per_class)]
                    subset.extend(class_ix_list)

            if self.use_meta_split:
                for _ in range(val_draws):
                    class_key = random.choice(self.val_classes)
                    subset.extend([random.choice(self.image_dict[class_key])[-1] for _ in range(self.samples_per_class)])
            yield subset

    def __len__(self):
        return self.sampler_length

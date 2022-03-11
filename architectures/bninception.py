import contextlib

import pretrainedmodels as ptm
import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(torch.nn.Module):
    def __init__(self, opt):
        super(Network, self).__init__()

        self.pars = opt
        self.model = ptm.__dict__['bninception'](num_classes=1000,
                                                 pretrained='imagenet')
        self.model.last_linear = torch.nn.Linear(
            self.model.last_linear.in_features, opt.embed_dim)

        if 'frozen' in opt.arch:
            for module in filter(lambda m: type(m) == nn.BatchNorm2d,
                                 self.model.modules()):
                module.eval()
                module.train = lambda _: None

        self.pool_base = F.avg_pool2d
        self.pool_aux = F.max_pool2d if 'double' in opt.arch else None

        self.name = opt.arch

    def forward(self, x, warmup=False, **kwargs):
        context = torch.no_grad() if warmup else contextlib.nullcontext()
        with context:
            x = self.model.features(x)
            prepool_y = y = self.pool_base(x, kernel_size=x.shape[-1])
            if self.pool_aux is not None:
                y += self.pool_aux(x, kernel_size=x.shape[-1])
            if 'lp2' in self.pars.arch:
                y += F.lp_pool2d(x, 2, kernel_size=x.shape[-1])
            if 'lp3' in self.pars.arch:
                y += F.lp_pool2d(x, 3, kernel_size=x.shape[-1])

            y = y.view(len(x), -1)

        z = self.model.last_linear(y)

        if 'normalize' in self.name:
            z = F.normalize(z, dim=-1)

        return {
            'embeds': z,
            'avg_features': y,
            'features': x,
            'extra_embeds': prepool_y
        }

import contextlib

import pretrainedmodels as ptm
import torch
import torch.nn as nn


class Network(torch.nn.Module):
    def __init__(self, opt):
        super(Network, self).__init__()

        self.pars = opt
        self.model = ptm.__dict__['resnet50'](
            num_classes=1000,
            pretrained='imagenet' if not opt.not_pretrained else None)

        self.name = opt.arch

        if 'frozen' in opt.arch:
            for module in filter(lambda m: type(m) == nn.BatchNorm2d,
                                 self.model.modules()):
                module.eval()
                module.train = lambda _: None

        opt.penultimate_dim = self.model.last_linear.in_features

        self.model.last_linear = torch.nn.Linear(
            self.model.last_linear.in_features, opt.embed_dim)

        self.layer_blocks = nn.ModuleList([
            self.model.layer1, self.model.layer2, self.model.layer3,
            self.model.layer4
        ])

        self.pool_base = torch.nn.AdaptiveAvgPool2d(1)
        self.pool_aux = torch.nn.AdaptiveMaxPool2d(
            1) if 'double' in opt.arch else None

    def forward(self, x, warmup=False, **kwargs):
        context = torch.no_grad() if warmup else contextlib.nullcontext()
        with context:
            x = self.model.maxpool(
                self.model.relu(self.model.bn1(self.model.conv1(x))))
            for i, layerblock in enumerate(self.layer_blocks):
                x = layerblock(x)
            prepool_y = x
            if self.pool_aux is not None:
                y = self.pool_aux(x) + self.pool_base(x)
            else:
                y = self.pool_base(x)
            y = y.view(y.size(0), -1)

        z = self.model.last_linear(y)

        if 'normalize' in self.pars.arch:
            z = torch.nn.functional.normalize(z, dim=-1)

        return {
            'embeds': z,
            'avg_features': y,
            'features': x,
            'extra_embeds': prepool_y
        }

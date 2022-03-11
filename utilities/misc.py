import random

import numpy as np
import torch
import tqdm


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gimme_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def gimme_save_string(opt):
    varx = vars(opt)
    base_str = ''
    for key in varx:
        base_str += str(key)
        if isinstance(varx[key], dict):
            for sub_key, sub_item in varx[key].items():
                base_str += '\n\t' + str(sub_key) + ': ' + str(sub_item)
        else:
            base_str += '\n\t' + str(varx[key])
        base_str += '\n\n'
    return base_str


class DataParallel(torch.nn.Module):
    def __init__(self, model, device_ids, dim):
        super().__init__()
        self.model = model.model
        self.network = torch.nn.DataParallel(model, device_ids, dim)

    def forward(self, x):
        return self.network(x)


def adjust_text(input_text, maxlen=30):
    text = ''
    count = 0
    for p, c in enumerate(input_text.split(' ')):
        if p:
            text += ' '
        if count > maxlen and len(text) > 0:
            text += '\n'
            count -= maxlen
        text += c
        count += len(c)
    return text

import copy
import importlib
import os

import criteria


def select(loss, opt, to_optim=None, batchminer=None):
    losses = os.listdir(os.path.join(os.getcwd(), 'criteria'))
    losses = [x.split('.py')[0] for x in losses if '__' not in x]
    losses = {x: 'criteria.{}'.format(x) for x in losses}

    if loss not in losses:
        raise NotImplementedError('Loss {} not implemented!'.format(loss))

    loss_lib = importlib.import_module(format(losses[loss]))

    if loss_lib.REQUIRES_BATCHMINER:
        if batchminer is None:
            raise Exception(
                'Loss {} requires one of the following batch mining methods: {}'
                .format(loss, loss_lib.ALLOWED_MINING_OPS))
        else:
            if batchminer.name not in loss_lib.ALLOWED_MINING_OPS:
                raise Exception('{}-mining not allowed for {}-loss!'.format(
                    batchminer.name, loss))

    loss_par_dict = {'opt': opt}
    if loss_lib.REQUIRES_BATCHMINER:
        loss_par_dict['batchminer'] = batchminer

    criterion = loss_lib.Criterion(**loss_par_dict)

    if to_optim is not None:
        if loss_lib.REQUIRES_OPTIM:
            if hasattr(criterion, 'optim_dict_list'
                       ) and criterion.optim_dict_list is not None:
                to_optim += criterion.optim_dict_list
            else:
                to_optim += [{
                    'params': criterion.parameters(),
                    'lr': criterion.lr
                }]

        return criterion, to_optim
    else:
        return criterion

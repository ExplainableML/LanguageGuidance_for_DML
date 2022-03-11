import datasets.cars196
import datasets.cub200
import datasets.stanford_online_products


def select(dataset, opt, data_path, splitpath=None):
    if 'cub200' in dataset:
        return cub200.give_dataloaders(opt, data_path)
    elif 'cars196' in dataset:
        return cars196.give_dataloaders(opt, data_path)
    elif 'online_products' in dataset:
        return stanford_online_products.give_dataloaders(opt, data_path)
    else:
        raise NotImplementedError(
            'A dataset for {} is currently not implemented.\n\Currently available are : cub200, cars196 and online_products!'
            .format(dataset))

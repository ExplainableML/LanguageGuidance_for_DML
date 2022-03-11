from batchminer import rho_distance, softhard, random, semihard, distance

BATCHMINING_METHODS = {
    'random': random,
    'semihard': semihard,
    'softhard': softhard,
    'distance': distance,
    'rho_distance': rho_distance
}


def select(batchminername, opt):
    if batchminername not in BATCHMINING_METHODS:
        raise NotImplementedError(
            'Batchmining {} not available!'.format(batchminername))

    batchmine_lib = BATCHMINING_METHODS[batchminername]

    return batchmine_lib.BatchMiner(opt)

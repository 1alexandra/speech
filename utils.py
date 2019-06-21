import h5py


def flatten(x):
    for k, v in x.items():
        if isinstance(v, h5py.Dataset):
            yield v
        else:
            for v in flatten(v):
                yield v
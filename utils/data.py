from datasets import Dataset
import random

def shuffle_group(dataset: Dataset, group_size, seed=None) -> Dataset:

    if seed is not None:
        random.seed(seed)
    idx = list(range(len(dataset)//group_size))
    random.shuffle(idx)
    idx = [list(range(i*group_size, (i+1)*group_size)) for i in idx]
    idx = [i for j in idx for i in j]

    return dataset.select(idx)
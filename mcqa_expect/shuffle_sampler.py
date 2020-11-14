from typing import List, Iterable
from torch.utils import data

from allennlp.common.registrable import Registrable
from allennlp.data.samplers import Sampler
import torch.utils.data
import random

@Sampler.register("shuffle_sampler")
class ShuffleSampler(data.Sampler, Sampler):
    r"""Samples elements sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        super().__init__(data_source)
        self.data_source = data_source


    def __iter__(self):
        indices = list(range(len(self.data_source)))
        random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return len(self.data_source)
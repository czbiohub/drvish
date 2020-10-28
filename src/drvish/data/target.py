from torch.utils.data import DataLoader, TensorDataset

from drvish.data.cupy import CupySparseDataLoader, CupySparseDataset


class TargetMixin:
    """
    Mixin class that stores the last argument as "target" data, to be added along with
    every batch
    """

    def __init__(self, *args):
        super().__init__(*args[:-1])
        self.target = args[-1]


class TensorTargetDataset(TargetMixin, TensorDataset):
    ...


class CupySparseTargetDataset(TargetMixin, CupySparseDataset):
    ...


class TargetLoaderMixin:
    """
    When combined with a TargetMixin-derived class, this mixin will append the target
    data to every batch, which allows us to use aggregate/bulk data for training
    """

    def __iter__(self):
        for batch in super().__iter__():
            batch.append(self.dataset.target)
            yield batch


class DataTargetLoader(TargetLoaderMixin, DataLoader):
    ...


class CupySparseTargetLoader(TargetLoaderMixin, CupySparseDataLoader):
    ...

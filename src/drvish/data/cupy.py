from torch.utils.data import DataLoader, Dataset
from torch.utils.dlpack import from_dlpack


class CupySparseDataset(Dataset):
    """Exactly the same as TensorDataset except for sparse cupy arrays (or normal ones,
    but there's no reason to use those)
    """

    def __init__(self, *arrays):
        assert all(arrays[0].shape[0] == arr.shape[0] for arr in arrays)
        self.arrays = arrays

    def __getitem__(self, index):
        return tuple(array[index] for array in self.arrays)

    def __len__(self):
        return self.arrays[0].shape[0]


class CupySparseDataLoader(DataLoader):
    """DataLoader that converts from sparse cupy array to a dense tensor. For large
    datasets that only fit in GPU memory in sparse form, this can speed up training
    """
    def __iter__(self):
        for indices in iter(self.batch_sampler):
            yield [
                from_dlpack(t[indices].todense().toDlpack())
                for t in self.dataset.arrays
            ]

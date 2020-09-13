import torch.distributions

from torch.utils.data import DataLoader, TensorDataset


class MCVDataLoader(DataLoader):
    """A version of the standard DataLoader that generates a new noise2self split every
    time it is iterated on. The dataset passed in should be an instance of TensorDataset
    and contain a tensor of UMI counts. Any transformation(s) of the data must be done
    downstream of this class.
    """

    def __init__(self, dataset: TensorDataset, **kwargs):
        super().__init__(dataset=dataset, **kwargs)

    def __iter__(self):
        for indices in iter(self.batch_sampler):
            yield self.split_molecules(*(d[indices] for d in self.dataset.tensors))

    @staticmethod
    def split_molecules(
        umis, data_split, data_split_complement, overlap_factor, *args,
    ):
        x_data = torch.clamp(
            torch.distributions.Binomial(
                umis, probs=data_split - overlap_factor
            ).sample(),
            0,
            umis,
        )
        y_data = torch.clamp(
            torch.distributions.Binomial(
                umis - x_data,
                probs=(1 - data_split) / (1 - data_split + overlap_factor),
            ).sample(),
            0,
            umis - x_data,
        )
        overlap = umis - x_data - y_data

        return (
            x_data + overlap,
            y_data + overlap,
            data_split,
            data_split_complement,
            *args,
        )

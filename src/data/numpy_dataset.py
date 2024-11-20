import os
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import get_worker_info


class NumpyTupleDatasetCached(Dataset):
    """Dataset of a tuple of datasets but cached.
    Data do not fit in memory (~100GB)

    TODO: only works with one worker
        """

    def __init__(self, dataset_func, data_dir, chunks, transform=None):
        self.dataset_func = dataset_func
        self.data_dir = data_dir
        self._datasets = self.load_datasets(0, self.dataset_func, self.data_dir, transform)
        self.transform = transform
        self.chunks = chunks
        self.current_chunk_index = 0
        self._length = self.compute_total_length()
        print(f'Total dataset length is {self._length}')

    def load_datasets(self, index, dataset_fun, data_dir, transform):
        filename = dataset_fun(index)
        return NumpyTupleDataset.load(os.path.join(data_dir, filename), transform=transform)._datasets

    def __len__(self):
        return self._length

    def compute_total_length(self):
        ds = self.load_datasets(self.chunks[-1], self.dataset_func, self.data_dir, self.transform)
        last_length = ds[0].shape[0]
        return last_length + self.chunks[-1]

    def find_chunk_index(self, data_index):
        for i, c in enumerate(self.chunks):
            if data_index < c:
                return i-1
        return len(self.chunks) - 1

    def reload(self, data_index):
        next_index = self.current_chunk_index + 1
        if next_index == len(self.chunks):
            return
        if (data_index >= self.chunks[self.current_chunk_index]) and (data_index < self.chunks[next_index]):
            return
        else:
            chunk_ind = self.find_chunk_index(data_index)
            self._datasets = self.load_datasets(self.chunks[chunk_ind], self.dataset_func, self.data_dir, self.transform)
            self.current_chunk_index = chunk_ind

    def __getitem__(self, data_index):
        self.reload(data_index)
        index = data_index - self.chunks[self.current_chunk_index]
        batches = [dataset[index] for dataset in self._datasets]
        if isinstance(index, (slice, list, np.ndarray)):
            length = len(batches[0])
            batches = [tuple([batch[i] for batch in batches])
                    for i in range(length)]   # six.moves.range(length)]
        else:
            batches = tuple(batches)

        if self.transform:
            batches = self.transform(batches)
        return batches

    def get_datasets(self):
        return self._datasets


class NumpyTupleDataset(Dataset):
    """Dataset of a tuple of datasets.

        It combines multiple datasets into one dataset. Each example is represented
        by a tuple whose ``i``-th item corresponds to the i-th dataset.
        And each ``i``-th dataset is expected to be an instance of numpy.ndarray.

        Args:
            datasets: Underlying datasets. The ``i``-th one is used for the
                ``i``-th item of each example. All datasets must have the same
                length.

        """

    def __init__(self, datasets, transform=None):
        # Load dataset
        if not datasets:
            raise ValueError('no datasets are given')
        length = len(datasets[0])  # 133885
        for i, dataset in enumerate(datasets):
            if len(dataset) != length:
                raise ValueError(
                    'dataset of the index {} has a wrong length'.format(i))
        # Initialization
        self._datasets = datasets
        self._length = length
        # self._features_indexer = NumpyTupleDatasetFeatureIndexer(self)
        # self.filepath = filepath
        self.transform = transform

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        batches = [dataset[index] for dataset in self._datasets]
        if isinstance(index, (slice, list, np.ndarray)):
            length = len(batches[0])
            batches = [tuple([batch[i] for batch in batches])
                    for i in range(length)]   # six.moves.range(length)]
        else:
            batches = tuple(batches)

        if self.transform:
            batches = self.transform(batches)
        return batches

    def get_datasets(self):
        return self._datasets


    @classmethod
    def save(cls, filepath, numpy_tuple_dataset):
        """save the dataset to filepath in npz format

        Args:
            filepath (str): filepath to save dataset. It is recommended to end
                with '.npz' extension.
            numpy_tuple_dataset (NumpyTupleDataset): dataset instance

        """
        if not isinstance(numpy_tuple_dataset, NumpyTupleDataset):
            raise TypeError('numpy_tuple_dataset is not instance of '
                            'NumpyTupleDataset, got {}'
                            .format(type(numpy_tuple_dataset)))
        np.savez(filepath, *numpy_tuple_dataset._datasets)
        print('Save {} done.'.format(filepath))

    @classmethod
    def load(cls, filepath, transform=None):
        print('Loading file {}'.format(filepath))
        if not os.path.exists(filepath):
            raise ValueError('Invalid filepath {} for dataset'.format(filepath))
            # return None
        load_data = np.load(filepath, allow_pickle=True)
        result = []
        i = 0
        while True:
            key = 'arr_{}'.format(i)
            if key in load_data.keys():
                result.append(load_data[key])
                i += 1
            else:
                break
        return cls(result, transform)

import os
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import get_worker_info
from multiprocessing import shared_memory, Lock
import struct
import data.spectrum_processing as dt
from functools import partial
import pandas as pd


max_atoms = 200
n_bonds = 4
spec_max_mz = 2500
max_num_peaks = 100
min_intensity = 0.1


def transform_label(label):
    spectrum = label[0]
    spectrum = [(float(s.split()[0]), float(s.split()[1])) for s in spectrum.strip().split('\n')]
    mz, ints = zip(*spectrum)

    spectrum = pd.DataFrame({'m/z': mz, 'intensity': ints})
    spectrum = dt.FilterPeaks(spec_max_mz, min_intensity)(spectrum)
    spectrum = dt.Normalize(intensity=True, mass=False, rescale_intensity=True, max_mz=spec_max_mz)(spectrum)
    spectrum = dt.TopNPeaks(max_num_peaks)(spectrum)
    spectrum = dt.ToMZIntConcatAlt(max_num_peaks)(spectrum)

    spectrum = list(spectrum)
    exact_mass = label[-1]
    return np.array([float(exact_mass)] + spectrum)


class NumpyTupleDatasetCached(Dataset):
    """Dataset of a tuple of datasets but cached.
    Data do not fit in memory (~100GB)

    TODO: only works with one worker
        """

    def __init__(self, dataset_func, data_dir, chunks, transform=None):
        self.dataset_func = dataset_func
        self.data_dir = data_dir
        self.transform = transform
        self._datasets = None
        self._next_datasets = None
        self.chunks = chunks
        self.lock = Lock()
        self._length = self.compute_total_length()
        print(f'Total dataset length is {self._length}')

    def load_datasets_into_shared_memory(self, chunk_index, to_store):
        """Load a single chunk into shared memory with the first dimension padded to 10,000."""
        max_first_dim = 10000  # Fixed size for the first dimension

        filename = self.dataset_func(self.chunks[chunk_index])
        file_path = os.path.join(self.data_dir, filename)
        print(f"Loading file {file_path} into shared memory")
        
        # Load datasets
        datasets = NumpyTupleDataset.load(file_path, transform=self.transform)._datasets
        if to_store == 'current':
            attrname = '_datasets'
        elif to_store == 'next':
            attrname = '_next_datasets'
        else:
            raise('Wrong value to store')
        attr = getattr(self, attrname)

        if attr is not None:
            # If shared memory already exists, copy new data into it
            for i, array in enumerate(datasets):
                # Convert object arrays to numeric types if needed
                if array.dtype == object:
                    array = np.array([transform_label(array[i]) for i in range(array.shape[0])])
                    array = np.array(array, dtype=np.float32)  # Convert to numeric (e.g., float32)

                # Access existing shared memory and reshape
                shared_array = np.frombuffer(attr[i][0].buf, dtype=attr[i][2])
                shared_array = shared_array.reshape(attr[i][1])  # Reshape to correct shape
                
                # Prepare padded array
                padded_shape = (max_first_dim,) + array.shape[1:]  # Keep other dimensions the same
                padded_array = np.zeros(padded_shape, dtype=array.dtype)
                copy_shape = (min(max_first_dim, array.shape[0]),) + array.shape[1:]  # Determine the copy size
                slices = tuple(slice(0, s) for s in copy_shape)  # Create slicing tuples
                padded_array[slices] = array[slices]  # Copy existing data into the padded array
                
                np.copyto(shared_array, padded_array)  # Directly copy without flattening
            
            # Update the chunk index in shared memory
            struct.pack_into("i", attr[-1].buf, 0, chunk_index)
        else:
            # Create shared memory for new buffers
            shared_memories = []
            for array in datasets:
                # Convert object arrays to numeric types if needed
                if array.dtype == object:
                    array = np.array([transform_label(array[i]) for i in range(array.shape[0])])
                    array = np.array(array, dtype=np.float32)  # Convert to numeric (e.g., float32)

                # Allocate shared memory with padded shape
                padded_shape = (max_first_dim,) + array.shape[1:]  # Fix first dimension, keep others
                shm = shared_memory.SharedMemory(create=True, size=np.prod(padded_shape) * array.itemsize)
                shared_array = np.frombuffer(shm.buf, dtype=array.dtype).reshape(padded_shape)

                # Pad and copy data
                padded_array = np.zeros(padded_shape, dtype=array.dtype)
                copy_shape = (min(max_first_dim, array.shape[0]),) + array.shape[1:]
                slices = tuple(slice(0, s) for s in copy_shape)
                padded_array[slices] = array[slices]  # Copy existing data into the padded array
                
                np.copyto(shared_array, padded_array)  # Directly copy the padded array
                shared_memories.append((shm, padded_shape, array.dtype))

            # Create shared memory for the current chunk index
            cur = shared_memory.SharedMemory(create=True, size=4)
            struct.pack_into("i", cur.buf, 0, chunk_index)
            shared_memories.append(cur)

            setattr(self, attrname, shared_memories)


    def retrieve_cur_chunk(self, to_retrieve):
        if to_retrieve == 'current':
            return struct.unpack_from("i", self._datasets[-1].buf, 0)[0]
        elif to_retrieve == 'next':
            return struct.unpack_from("i", self._next_datasets[-1].buf, 0)[0]
        else:
            raise('Wrong value to store')

    def retrieve_datasets(self, to_retrieve):
        res = []
        for i in range(3):
            if to_retrieve == 'current':
                shared_array = np.ndarray(self._datasets[i][1], dtype=self._datasets[i][2], buffer=self._datasets[i][0].buf)
            elif to_retrieve == 'next':
                shared_array = np.ndarray(self._next_datasets[i][1], dtype=self._next_datasets[i][2], buffer=self._next_datasets[i][0].buf)
            else:
                raise('Wrong value to store')
            res.append(shared_array)
        return res

    def load_datasets(self, index, dataset_fun, data_dir, transform):
        filename = dataset_fun(index)
        return NumpyTupleDataset.load(os.path.join(data_dir, filename), transform=transform)._datasets

    def __len__(self):
        return self._length

    def compute_total_length(self):
        with self.lock:
            self.load_datasets_into_shared_memory(len(self.chunks) - 1, 'current')
        last_length = self.retrieve_datasets('current')[0].shape[0]
        result = last_length + self.chunks[-1]
        with self.lock:
            self.load_datasets_into_shared_memory(0, 'current')
            self.load_datasets_into_shared_memory(1, 'next')
        return result

    def find_chunk_index(self, data_index):
        for i, c in enumerate(self.chunks):
            if data_index < c:
                return i-1
        return len(self.chunks) - 1

    def reload(self, data_index):
        chunk_ind = self.find_chunk_index(data_index)
        current_chunk_index = self.retrieve_cur_chunk('current')
        current_chunk_next_index = self.retrieve_cur_chunk('next')
        if chunk_ind == current_chunk_index:
            return 'current'
        if chunk_ind == current_chunk_next_index:
            return 'next'
        self.load_datasets_into_shared_memory(current_chunk_next_index, 'current')
        self.load_datasets_into_shared_memory(chunk_ind, 'next')
        return 'next'

    def __getitem__(self, data_index):
        with self.lock:
            retrieve_variable = self.reload(data_index)
            current_chunk_index = self.retrieve_cur_chunk(retrieve_variable)
        index = data_index - self.chunks[current_chunk_index]
        batches = [dataset[index] for dataset in self.retrieve_datasets(retrieve_variable)]
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

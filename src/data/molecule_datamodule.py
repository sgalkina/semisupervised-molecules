from typing import Any, Dict, Optional, Tuple, List

import os
import torch
import numpy as np
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from data.numpy_dataset import NumpyTupleDataset, NumpyTupleDatasetCached
import data.spectrum_processing as dt
from functools import partial
import pandas as pd
import glob
import re

max_atoms = 200
n_bonds = 4
spec_max_mz = 2500
max_num_peaks = 100
min_intensity = 0.1


def one_hot_hmdb(data, atomic_num_list, out_size=max_atoms):
    num_max_id = len(atomic_num_list)
    assert data.shape[0] == out_size
    b = np.zeros((out_size, num_max_id), dtype=np.float32)
    for i in range(out_size):
        ind = atomic_num_list.index(data[i])
        b[i, ind] = 1.
    return b


def transform_label(label):
    if label.dtype == np.float32:
        return label
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


def transform_fn_hmdb(atomic_num_list, data):
    node, adj, label = data
    node = one_hot_hmdb(node, atomic_num_list).astype(np.float32)
    # single, double, triple and no-bond. Note that last channel axis is not connected instead of aromatic bond.
    adj = np.concatenate([adj[:3], 1 - np.sum(adj[:3], axis=0, keepdims=True)],
                         axis=0).astype(np.float32)
    return node, adj, transform_label(label)


def worker_init_fn(worker_id):
    """Initialize worker by mapping shared memory."""
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset



class MoFlowDataModule(LightningDataModule):
    """`LightningDataModule` for the MNIST dataset.

    The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples.
    It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a
    fixed-size image. The original black and white images from NIST were size normalized to fit in a 20x20 pixel box
    while preserving their aspect ratio. The resulting images contain grey levels as a result of the anti-aliasing
    technique used by the normalization algorithm. the images were centered in a 28x28 image by computing the center of
    mass of the pixels, and translating the image so as to position this point at the center of the 28x28 field.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        atomic_list: List[int],
        data_dir: str = "data/",
        train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
        batch_size: int = 4,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `MNISTDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.atomic_list = atomic_list

        self.transform_fn = partial(transform_fn_hmdb, self.atomic_list)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.data_file_train = lambda c: f'molecules/mona_relgcn_chunk_{c}_kekulized_ggnp.npz'
        self.data_file_valid = 'molecules/mona_TEST_relgcn_kekulized_ggnp.npz'
        self.data_file_test = 'molecules/mona_TEST_relgcn_kekulized_ggnp.npz'

        self.batch_size_per_device = batch_size

    def all_chunks(self):
        file_pattern = os.path.join(self.hparams.data_dir, 'molecules/mona_relgcn_chunk_*_kekulized_ggnp.npz')
        files = glob.glob(file_pattern)
        c_values = []
        for file in files:
            match = re.search(os.path.join(self.hparams.data_dir, r'molecules/mona_relgcn_chunk_(\d+)_kekulized_ggnp\.npz'), file)
            if match:
                c_values.append(int(match.group(1)))
        c_values.sort()
        print('c_values', c_values)
        return c_values

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of MNIST classes (10).
        """
        return 10

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = NumpyTupleDatasetCached(self.data_file_train, self.hparams.data_dir, self.all_chunks(), transform=self.transform_fn)
            self.data_val = NumpyTupleDataset.load(os.path.join(self.hparams.data_dir, self.data_file_valid), transform=self.transform_fn)
            self.data_test = NumpyTupleDataset.load(os.path.join(self.hparams.data_dir, self.data_file_test), transform=self.transform_fn)

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            worker_init_fn=worker_init_fn,
            shuffle=False,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = MoFlowDataModule()

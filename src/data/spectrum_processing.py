import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import utils


def spectrum_split_string(spec):
    """
    Parse string representation of spectrum and convert it into sparse numerical representation.

    :param spec: Spectrum as string in format "mz1:int1 mz2:int2 ..."
    :return: List of pairs [mz, int]
    """
    return [[float(y) for y in x.split(':')] for x in spec.split(' ')]


def spectrum_to_dense(spec, max_mz, resolution):
    """
    Convert the sparse spectrum representation into dense vector representation.
    The mass is binned according to the resolution parameter, e.g. if resolution=1.0
    and max_mz=2500.0 then there are 2500 bins, each 1 m/z unit away.

    :param spec: Sparse spectrum representation
    :param max_mz: Maximum m/z value, e.g. 2500.0
    :param resolution: Mass resolution.
    :return: Dense binned representation of spectrum
    """
    numbers = np.arange(0, max_mz, step=resolution, dtype=np.float32)
    result = np.zeros(len(numbers), dtype=np.float32)
    for i in spec:
        idx = np.searchsorted(numbers, i[0])
        try:
            result[idx] = i[1]
        except IndexError:
            result[-1] = i[1]
    return result


def ion2int(mode):
    """
    Convert ionization mode from string to digit.

    :param mode: Mode as string
    :return: Numerical representation of mode, i.e. 0 for negative and 1 for positive.
    """
    return 1 if mode.lower() == 'positive' else 0


class Identity:
    def __call__(self, x):
        """
        Identity operator.

        :param x: Input sample
        :return: Pass forward input.
        """
        return x


class SplitSpectrum:
    def __call__(self, sample):
        """
        Convert string representation of spectrum to sparse numerical representation.

        :param sample: Row from data frame as Dict, or spectrum as string
        :return: Sparse representation of spectrum in 'spectrum' dict key, or direct value.
        """
        spec = sample['spectrum'] if isinstance(sample, dict) else sample
        res = spectrum_split_string(spec)
        if isinstance(sample, dict):
            if 'id' in sample and sample['id'] != sample['id']:
                sample['id'] = ''
            sample['spectrum'] = res
            return sample
        else:
            return res


class FilterPeaks:
    def __init__(self, max_mz, min_intensity=0.):
        self.max_mz = max_mz
        self.min_intensity = min_intensity

    def __call__(self, sample):
        """
        Filter peaks by minimum intensity, and maximum mass.

        :param sample: Row from data frame as Dict with sparse spectrum, or sparse spectrum
        :return: 
        """
        spec = sample['spectrum'] if isinstance(sample, dict) else sample
        spec = np.array(spec)
        spec = spec[np.where(spec[:,1] >= self.min_intensity)]
        spec = spec[np.where(spec[:,0] <= self.max_mz)]
        if isinstance(sample, dict):
            sample['spectrum'] = spec
            return sample
        else:
            return spec

class TopNPeaks:
    def __init__(self, n):
        self.n = n

    def __call__(self, sample):
        spec = sample['spectrum'] if isinstance(sample, dict) else sample
        spec = np.array(spec)
        idx = np.argsort(spec[:,1])[::-1][:self.n]
        spec = spec[idx]
        if isinstance(sample, dict):
            sample['spectrum'] = spec
            return sample
        else:
            return spec

class Normalize:
    def __init__(self, intensity=True, mass=True, rescale_intensity=False, min_intensity=0.1, max_mz=2500.):
        self.intensity = intensity
        self.mass = mass
        self.rescale_intensity = rescale_intensity
        self.min_intensity = min_intensity
        self.max_mz = max_mz

    def __call__(self, sample):
        spec = sample['spectrum'] if isinstance(sample, dict) else sample
        if spec.shape[0] > 0:
            spec = np.array(spec)
            if self.rescale_intensity:
                mx, mn = spec[:,1].max(), spec[:,1].min()
                if not np.isclose(mx, mn):
                    mn = mn - self.min_intensity
                    spec[:,1] = (spec[:,1] - mn) * 100. / (mx - mn)
            if self.intensity:
                spec[:,1] = spec[:,1] * 0.01
            if self.mass:
                spec[:,0] = spec[:,0] * (1. / self.max_mz)
        if isinstance(sample, dict):
            sample['spectrum'] = spec
            return sample
        else:
            return spec

class UpscaleIntensity:
    def __init__(self, max_mz=2500.):
        self.max_mz = max_mz

    def __call__(self, sample):
        spec = sample['spectrum'] if isinstance(sample, dict) else sample
        spec = np.array(spec)
        spec[:,1] = spec[:,1] * self.max_mz / 100.
        if isinstance(sample, dict):
            sample['spectrum'] = spec
            return sample
        else:
            return spec


class DeUpscaleIntensity:
    def __init__(self, max_mz=2500.):
        self.max_mz = max_mz

    def __call__(self, sample):
        spec = sample['spectrum'] if isinstance(sample, dict) else sample
        spec = np.array(spec)
        spec[:,1] = spec[:,1] / self.max_mz * 100.
        if isinstance(sample, dict):
            sample['spectrum'] = spec
            return sample
        else:
            return spec


class Denormalize:
    def __init__(self, intensity=True, mass=True, max_mz=2500.):
        self.intensity = intensity
        self.mass = mass
        self.max_mz = max_mz

    def __call__(self, sample):
        spec = sample['spectrum'] if isinstance(sample, dict) else sample
        spec = np.array(spec)
        if self.intensity:
            spec[:,1] = spec[:,1] * 100.
        if self.mass:
            spec[:,0] = spec[:,0] * self.max_mz
        if isinstance(sample, dict):
            sample['spectrum'] = spec
            return sample
        else:
            return spec

class ToMZIntConcat:
    def __init__(self, max_num_peaks, normalize=True):
        self.max_num_peaks = max_num_peaks

    def __call__(self, sample):
        spec = sample['spectrum'] if isinstance(sample, dict) else sample
        spec = np.array(spec)[:self.max_num_peaks]
        full = np.zeros((2 * self.max_num_peaks), dtype=np.float32)
        full[:spec.shape[0]] = spec[:,0]
        full[self.max_num_peaks:self.max_num_peaks + spec.shape[0]] = spec[:,1]
        if isinstance(sample, dict):
            sample['spectrum'] = full
            return sample
        else:
            return full

class ToMZIntConcatAlt:
    def __init__(self, max_num_peaks):
        self.max_num_peaks = max_num_peaks

    def __call__(self, sample):
        spec = sample['spectrum'] if isinstance(sample, dict) else sample
        spec = np.array(spec)[:self.max_num_peaks]
        full = np.zeros((2 * self.max_num_peaks), dtype=np.float32)
        idx = np.arange(0, 2 * self.max_num_peaks, 2)[:spec.shape[0]]
        full[idx] = spec[:,0]
        full[idx + 1] = spec[:,1]
        if isinstance(sample, dict):
            sample['spectrum'] = full
            return sample
        else:
            return full

class ToMZIntDeConcat:
    def __init__(self, max_num_peaks):
        self.max_num_peaks = max_num_peaks

    def __call__(self, sample):
        spec = sample['spectrum'] if isinstance(sample, dict) else sample
        mzs, ints = spec[:self.max_num_peaks], spec[self.max_num_peaks:]
        full = np.vstack((mzs, ints)).T
        if isinstance(sample, dict):
            sample['spectrum'] = full
            return sample
        else:
            return full

class ToMZIntDeConcatAlt:
    def __init__(self, max_num_peaks):
        self.max_num_peaks = max_num_peaks

    def __call__(self, sample):
        spec = sample['spectrum'] if isinstance(sample, dict) else sample
        idx = np.arange(0, 2 * self.max_num_peaks, 2)
        mzs, ints = spec[idx], spec[idx + 1]
        # mzs, ints = spec[:self.max_num_peaks], spec[self.max_num_peaks:]
        full = np.vstack((mzs, ints)).T
        if isinstance(sample, dict):
            sample['spectrum'] = full
            return sample
        else:
            return full

class ToDenseSpectrum:
    def __init__(self, resolution, max_mz):
        self.resolution = resolution
        self.max_mz = max_mz

    def __call__(self, sample):
        spec = sample['spectrum'] if isinstance(sample, dict) else sample
        result = spectrum_to_dense(spec, self.max_mz, self.resolution)
        if isinstance(sample, dict):
            # sample['spectrum'] = torch.from_numpy(result)
            sample['spectrum'] = result
            return sample
        else:
            return result

class ScaleSpectrum:
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, sample):
        spec = sample['spectrum'] if isinstance(sample, dict) else sample
        spec *= self.scale
        if isinstance(sample, dict):
            sample['spectrum'] = spec
            return sample
        else:
            return spec


class NormalizeSpectrum:
    def __init__(self, m=torch.Tensor([0.]), std=torch.Tensor([1.])):
        self.m = m
        self.std = std
        self.std += 1e-06 # avoid zeros

    def __call__(self, sample):
        spec = sample['spectrum'] if isinstance(sample, dict) else sample
        spec = (spec - self.m) / self.std
        if isinstance(sample, dict):
            sample['spectrum'] = spec
            return sample
        else:
            return spec


class ExpSpectrum:
    def __call__(self, sample):
        spec = sample['spectrum'] if isinstance(sample, dict) else sample
        spec = np.exp(spec) - 1.
        if isinstance(sample, dict):
            sample['spectrum'] = spec
            return sample
        else:
            return spec


class DenormalizeSpectrum:
    def __init__(self, m=torch.Tensor([0.]), std=torch.Tensor([1.])):
        self.m = m
        self.std = std

    def __call__(self, sample):
        spec = sample['spectrum'] if isinstance(sample, dict) else sample
        spec = spec * self.std + self.m
        if isinstance(sample, dict):
            sample['spectrum'] = spec
            return sample
        else:
            return spec


class ToSparseSpectrum:
    def __init__(self, resolution, max_mz):
        self.resolution = resolution
        self.max_mz = max_mz

    def __call__(self, sample):
        spectrum = sample['spectrum'] if isinstance(sample, dict) else sample
        idx = (spectrum > 0).nonzero()[0]
        res = [(idx.astype(np.float32) * self.resolution).flatten(), spectrum[idx].flatten()]
        if isinstance(sample, dict):
            sample['spectrum'] = res
            return sample
        else:
            return res


class ToString:
    def __call__(self, sample):
        spec = sample['spectrum'] if isinstance(sample, dict) else sample
        ints = spec[:, 1].tolist()
        strs = ' '.join(['{:.4f}:{:.4f}'.format(mz, ints[i]) 
                            for i, mz in enumerate(spec[:, 0].tolist())])
        if isinstance(sample, dict):
            sample['spectrum'] = strs
            return sample
        else:
            return strs

class SparseToString:
    def __call__(self, sample):
        spec = sample['spectrum'] if isinstance(sample, dict) else sample
        inte = spec[1].tolist()
        strs = ' '.join(['{:.4f}:{:.4f}'.format(mz, inte[i]) 
                            for i, mz in enumerate(spec[0].tolist())])
        if isinstance(sample, dict):
            sample['spectrum'] = strs
            return sample
        else:
            return strs


class Ion2Int:
    def __init__(self, one_hot=True):
        self.one_hot = one_hot

    def __call__(self, sample):
        mode = sample['ionization mode'] if isinstance(sample, dict) else sample
        mode = ion2int(mode) if isinstance(mode, str) else mode
        mode = int(np.nan_to_num(mode))
        if self.one_hot:
            mode = np.eye(2, dtype=np.float32)[mode]
        else:
            mode = float(mode)
        if isinstance(sample, dict):
            sample['ionization mode'] = mode
            return sample
        else:
            return mode


class Int2OneHot:
    def __init__(self, col_name, n_classes=2):
        self.col_name = col_name
        self.n_classes = n_classes

    def __call__(self, sample):
        mode = sample[self.col_name] if isinstance(sample, dict) else sample
        mode = int(np.nan_to_num(mode))
        mode = np.eye(self.n_classes, dtype=np.float32)[mode]
        if isinstance(sample, dict):
            sample[self.col_name] = mode
            return sample
        else:
            return mode


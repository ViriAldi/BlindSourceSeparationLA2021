"""Mic class"""
from __future__ import annotations
import numpy as np
from modelling.point import SoundPoint
from typing import Union
import matplotlib.pyplot as plt


class Mic(SoundPoint):
    """Model mic"""
    __mics_id = 0

    def __init__(self, x, y, noise_dist=None, **kwargs):
        super().__init__(x, y)
        self._sample = np.array([0.0], dtype=np.float)
        self._receiving = True
        self._time = None
        self._id = Mic.__mics_id
        Mic.__mics_id += 1
        self._noise = noise_dist
        self._noise_args = kwargs
        self._num_signals = 0

    def __lshift__(self, sample: Union[np.ndarray, list]):
        if not self._receiving:
            raise RuntimeError("mic finished reception")
        if type(sample) is list:
            sample = np.array(sample, dtype=np.float)
        len_diff = len(self._sample) - len(sample)
        if len_diff < 0:
            self._sample = np.pad(self._sample, (0, -len_diff), 'wrap')
        else:
            sample = np.pad(sample, (0, len_diff), 'wrap')
        self._sample += sample
        self._num_signals += 1
        return self

    def finalize(self):
        self.check_sr()
        self._time = np.arange(len(self._sample)) / self.get_rate()
        self._receiving = False
        self._sample /= self._num_signals
        if self._noise:
            self._sample += self._noise(size=len(self._sample),
                                        **self._noise_args)


    def _check_final(self):
        if self._receiving:
            raise RuntimeError("finalize input before sound processing")

    def show_time_domain(self):
        self._check_final()
        plt.plot(self._time, self._sample)
        plt.grid()
        plt.title(f"Time-domain for Mic {self._id}")
        plt.ylim(ymin=0)
        plt.ylabel("Amplitude")
        plt.xlabel('Time, seconds')
        plt.show()

    def __invert__(self):
        self._check_final()
        return self._sample

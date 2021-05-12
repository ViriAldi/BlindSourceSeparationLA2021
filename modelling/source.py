"""Source class"""
from __future__ import annotations
import numpy as np
from modelling.point import SoundPoint
from typing import Union


class Source(SoundPoint):
    """Model source."""
    __src_id = 0

    def __init__(self, x: Union[int, float], y: Union[int, float],
                 sample: Union[np.ndarray, list], loudness: float = 1.0):
        super().__init__(x, y)
        self._sample = loudness * (sample if (type(sample) is not list)
                                   else np.array(sample, dtype=np.float))
        self._id = Source.__src_id
        Source.__src_id += 1

    def __invert__(self):
        return self._sample.copy()

    def __le__(self, phase_count: Union[int, float]):
        phase_count = int(round(phase_count))
        return np.roll(self._sample, phase_count % len(self))

    def __len__(self):
        return len(self._sample)

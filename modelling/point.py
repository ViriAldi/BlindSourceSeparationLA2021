"""Point class."""
from __future__ import annotations
from typing import Union
import numpy as np


class SoundPoint:
    """Model signal source"""
    __sr = None

    @classmethod
    def set_rate(cls, env):
        cls.__sr = env.get_rate()

    @classmethod
    def get_rate(cls):
        return cls.__sr

    def __init__(self, x: Union[int, float], y: Union[int, float]):
        self.__data = np.array([x, y])

    @classmethod
    def check_sr(cls):
        if not cls.__sr:
            raise RuntimeError("enviroment not initialized")

    def __sub__(self, other: SoundPoint):
        return np.linalg.norm(self.__data - other.__data)

    def get_3d_coords(self):
        return np.append(self.__data, 1)

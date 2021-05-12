"""Represent a wall."""
from modelling.source import Source
import numpy as np


class Wall:
    """Wall in 2D."""

    def __init__(self, k: float, c: float, reflection_coef: float = 1.0):
        self._reflection_coef = reflection_coef
        self._reflection_matrix = np.array([
            [1-k**2, 2*k, -2*k*c],
            [2*k, k**2-1, 2*c],
            [0, 0, 1+k**2]
        ]) / (1+k**2)

    def reflect(self, source: Source):
        coords = self._reflection_matrix.dot(source.get_3d_coords())
        return Source(coords[0], coords[1], ~source,
                      loudness=self._reflection_coef)

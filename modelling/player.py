"""Player class."""
import simpleaudio as sa
import numpy as np
from modelling.microphone import Mic
from modelling.source import Source
from typing import Union


class Player:

    def __init__(self, env):
        self.__sr = env.get_rate()

    def play(self, data: Union[Mic, Source, list]):
        sample = ~data if isinstance(data, (Mic, Source)) else data
        play_sample = sample * (2 ** 15 - 1) / np.max(np.abs(sample))
        play_sample = play_sample.astype(np.int16)
        play_obj = sa.play_buffer(play_sample, 1, 2, self.__sr)
        play_obj.wait_done()

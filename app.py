import numpy as np
import matplotlib.pyplot as plt
import scipy

from modelling import Environment, Mic, Source
import librosa
from scipy.signal import spectrogram
import pyroomacoustics
from ilmra import BSS
import soundfile as sf
import os

if __name__ == '__main__':
    n_micro = 3
    e = Environment(mics=[Mic(11, 3), Mic(-5, 8), Mic(3, 17)],
                    sources=[Source(10, 4, librosa.load("audio/jojo.wav")[0][:150000]),
                             Source(-4, 7, librosa.load("les.wav")[0][:150000], loudness=100),
                             Source(2, 16, librosa.load("nobody.wav")[0][:150000], loudness=100)])
    e.transmit_all(real_shift=False)
    S = np.stack([e.get_src_at(idx) for idx in range(n_micro)], -1)
    X = np.stack([e.get_data_at(idx) for idx in range(n_micro)], -1)

    method = "their_mnmf"

    micros = [X[:, idx] for idx in range(X.shape[1])]
    f, t, _ = scipy.signal.stft(micros[0])
    sps = [scipy.signal.stft(mic)[2] for mic in micros]
    al = np.array(sps)

    solver = BSS()
    #nw = solver.iva(al)
    nw = pyroomacoustics.bss.fastmnmf(al.transpose((2, 1, 0)), n_src=3).transpose((2, 1, 0))

    #e.play(scipy.signal.istft(sps[0])[1])
    #e.play(scipy.signal.istft(nw[0])[1])
    #e.play(scipy.signal.istft(nw[1])[1])
    #e.play(scipy.signal.istft(nw[2])[1])

    sf.write(f"saved/exp1_mixed_{method}.wav", scipy.signal.istft(sps[0])[1], 22000)
    sf.write(f"saved/exp1_base1_{method}.wav", scipy.signal.istft(nw[0])[1], 22000)
    sf.write(f"saved/exp1_base2_{method}.wav", scipy.signal.istft(nw[1])[1], 22000)
    sf.write(f"saved/exp1_base3_{method}.wav", scipy.signal.istft(nw[2])[1], 22000)

    sources = [S[:, idx] for idx in range(S.shape[1])]
    sps_sources = [scipy.signal.stft(mic)[2] for mic in sources]

    idx = 1
    n = n_micro
    plt.figure(figsize=(20, 10))
    for i in range(n):
        plt.subplot(3, n, idx)
        plt.pcolormesh(t, f, np.log(np.abs(sps_sources[i])))
        idx += 1
    for i in range(n):
        plt.subplot(3, n, idx)
        plt.pcolormesh(t, f, np.log(np.abs(sps[i])))
        idx += 1
    for i in range(n):
        plt.subplot(3, n, idx)
        plt.pcolormesh(t, f, np.log(np.abs(nw[i])))
        idx += 1

    plt.savefig(f"saved/ex1_triple_{method}.png", dpi=200)

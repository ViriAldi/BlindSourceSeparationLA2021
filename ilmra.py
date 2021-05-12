import numpy
import numpy as np


class BSS:

    @staticmethod
    def projection_back(y, ref):

        num = np.sum(np.conj(ref[:, :, None]) * y, axis=0)
        denom = np.sum(np.abs(y) ** 2, axis=0)

        c = np.ones(num.shape, dtype=np.complex)
        I = denom > 0.0
        c[I] = num[I] / denom[I]

        return c

    @staticmethod
    def ilrma(x, n_comp=2, n_iter=20):
        # x.shape == (num_sources, num_freq, num_frames)
        num_sources, num_freq, num_frames = x.shape
        orig = x.copy()

        t = np.random.random((num_sources, num_freq, n_comp))
        v = np.random.random((num_sources, num_frames, n_comp))

        w = np.array([np.eye(num_sources) for _ in range(num_freq)], dtype=x.dtype)
        y = (w @ x.swapaxes(0, 1)).swapaxes(0, 1)
        lamd = np.zeros(num_sources)
        eyes = np.tile(np.eye(num_sources, num_sources), (num_freq, 1, 1))

        r = t @ v.swapaxes(1, 2)
        r[r < 1e-15] = 1e-15

        for idx in range(n_iter):
            # UPDATE T
            t *= np.sqrt(((abs(y)**2 / r**2) @ v) / (r**-1 @ v))
            t[t < 1e-15] = 1e-15
            # UPDATE V
            v *= np.sqrt(((abs(y)**2 / r**2).transpose(0, 2, 1) @ t) / ((r**-1).transpose(0, 2, 1) @ t))
            v[v < 1e-15] = 1e-15
            # UPDATE R
            r = t @ v.swapaxes(1, 2)
            r[r < 1e-15] = 1e-15

            # HEAD optimization auxiliary
            for m in range(num_sources):
                vx = (x.swapaxes(0, 1) / r[m, :, None, :]) @ np.conj(x.transpose((1, 2, 0))) / num_frames
                wv = w @ vx
                w[:, m, :] = np.conj(np.linalg.solve(wv, eyes[:, :, m]))
                w[:, m, :] /= np.sqrt((w[:, None, m, :] @ vx) @ np.conj(w[:, m, :, None]))[:, :, 0]

            y = (w @ x.swapaxes(0, 1)).swapaxes(0, 1)

            for m in range(num_sources):
                lamd[m] = 1 / np.sqrt(np.mean(abs(y[m, :, :])**2))
                lamd[lamd < 1e-15] = 1e-15
                w[:, :, m] *= lamd[m]
                r[m, :, :] *= lamd[m] ** 2
                t[m, :, :] *= lamd[m] ** 2

        z = BSS.projection_back(y.transpose(2, 1, 0), orig[0, :, :].T)
        y *= np.conj(z[None, :, :].transpose(2, 1, 0))

        return y

    @staticmethod
    def iva(x, n_iter=20):
        # x.shape == (num_sources, num_freq, num_frames)
        num_sources, num_freq, num_frames = x.shape
        orig = x.copy()
        w = np.array([np.eye(num_sources) for _ in range(num_freq)], dtype=x.dtype)
        y = (w @ x.swapaxes(0, 1)).swapaxes(0, 1)
        eyes = np.tile(np.eye(num_sources, num_sources), (num_freq, 1, 1))
        r = np.zeros((num_sources, num_frames))

        for idx in range(n_iter):

            # UPDATE R
            r[:, :] = 2.0 * np.linalg.norm(y.transpose(1, 0, 2), axis=0)
            r[r < 1e-15] = 1e-15

            # HEAD optimization auxiliary
            for m in range(num_sources):
                v = (x.swapaxes(0, 1) / r[None, m, None, :]) @ np.conj(x.transpose((1, 2, 0))) / num_frames
                wv = w @ v
                w[:, m, :] = np.conj(np.linalg.solve(wv, eyes[:, :, m]))
                w[:, m, :] /= np.sqrt((w[:, None, m, :] @ v) @ np.conj(w[:, m, :, None]))[:, :, 0]

            y = (w @ x.swapaxes(0, 1)).swapaxes(0, 1)

        z = BSS.projection_back(y.transpose(2, 1, 0), orig[0, :, :].T)
        y *= np.conj(z[None, :, :].transpose(2, 1, 0))

        return y

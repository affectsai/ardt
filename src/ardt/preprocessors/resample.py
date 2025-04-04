from .SignalPreprocessor import SignalPreprocessor

import math
from scipy.signal import medfilt, butter, filtfilt
from scipy.ndimage import median_filter
from scipy.signal import resample_poly
import numpy as np
import neurokit2 as nk

class Resample(SignalPreprocessor):
    """
    This signal processor models noise by apply 600ms and 200ms median filters sequentially, then subtracting the
    result for the original signal. Finally, a 12th order low-pass Butterworth filter is applied with 35hz cutoff.
    Optionally, if target_fs is not equal to fs, the signal is resampled to the target_fs.
    """
    def __init__(self, child_preprocessor=None, parent_preprocessor=None, fs=256, target_fs=256):
        super().__init__(child_preprocessor=child_preprocessor, parent_preprocessor=parent_preprocessor)
        self.fs=fs
        self.target_fs=target_fs

        gcd = math.gcd(self.fs, self.target_fs)
        self._up = self.target_fs // gcd
        self._down = self.fs // gcd

    def _filter(self, signal):
        if self.fs != self.target_fs:
            signal = resample_poly(signal, up=self._up, down=self._down)

        return signal

    def process_signal(self, ecg_signal):
        return np.array([self._filter(nk.ecg_invert(ecg_signal[c, :])[0]) for c in range(ecg_signal.shape[0])])


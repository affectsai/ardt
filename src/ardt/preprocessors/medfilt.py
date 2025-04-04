from .lowpass import LowPass
from .resample import Resample
from .SignalPreprocessor import SignalPreprocessor

import math
from scipy.signal import medfilt, butter, filtfilt
from scipy.ndimage import median_filter
from scipy.signal import resample_poly
import numpy as np
import neurokit2 as nk

class MedianFilterWith35HzLowPass(SignalPreprocessor):
    """
    This signal processor models noise by apply 600ms and 200ms median filters sequentially, then subtracting the
    result for the original signal. Finally, a 12th order low-pass Butterworth filter is applied with 35hz cutoff.
    Optionally, if target_fs is not equal to fs, the signal is resampled to the target_fs.
    """
    def __init__(self, child_preprocessor=None, parent_preprocessor=None, fs=256, target_fs=256):
        super().__init__(child_preprocessor=child_preprocessor, parent_preprocessor=parent_preprocessor)

        self.impl = MedianFilter(
            fs=fs,
            child_preprocessor=LowPass(
                fs=fs,
                freq=35.0,
                child_preprocessor=Resample(
                    fs=fs,
                    target_fs=target_fs,
                    child_preprocessor=child_preprocessor
                )
            )
        )

    def process_signal(self, ecg_signal):
        return self.impl.process_signal(ecg_signal)

class MedianFilter(SignalPreprocessor):
    """
    This signal processor models noise by apply 600ms and 200ms median filters sequentially, then subtracting the
    result for the original signal. Finally, a 12th order low-pass Butterworth filter is applied with 35hz cutoff.
    Optionally, if target_fs is not equal to fs, the signal is resampled to the target_fs.
    """
    def __init__(self, child_preprocessor=None, parent_preprocessor=None, fs=256):
        super().__init__(child_preprocessor=child_preprocessor, parent_preprocessor=parent_preprocessor)
        self.fs=fs

    def _median_filter(self, signal, window_ms):
        window = int(round((window_ms/1000) * self.fs))
        if window % 2 == 0:  # median filter window must be odd
            window += 1
        if window > len(signal):
            raise ValueError(f"Median filter window ({window}, {window_ms} ms) exceeds signal length ({len(signal)})")
        return median_filter(signal, size=window)

    def _filter(self, signal):
        noise = self._median_filter(signal, 200)
        noise = self._median_filter(noise, 600)
        result = signal - noise

        return result

    def process_signal(self, ecg_signal):
        return np.array([self._filter(nk.ecg_invert(ecg_signal[c, :])[0]) for c in range(ecg_signal.shape[0])])


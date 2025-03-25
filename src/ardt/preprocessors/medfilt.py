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
        self.fs=fs
        self.target_fs=target_fs

    def _median_filter(self, signal, window_ms):
        window = int(round((window_ms/1000) * self.fs))
        if window % 2 == 0:  # median filter window must be odd
            window += 1
        if window > len(signal):
            raise ValueError(f"Median filter window ({window}, {window_ms} ms) exceeds signal length ({len(signal)})")
        return median_filter(signal, size=window)

    def _low_pass(self, signal, cutoff):
        nyq = 0.5 * self.fs
        b, a = butter(N=12, Wn=cutoff / nyq, btype='low', analog=False)
        signal_filtered = filtfilt(b, a, signal)
        return signal_filtered

    def _filter(self, signal):
        noise = self._median_filter(signal, 600)
        noise = self._median_filter(noise, 200)
        result = signal - noise
        result = self._low_pass(result, 35)

        if self.fs != self.target_fs:
            gcd = math.gcd(self.fs, self.target_fs)
            up = self.target_fs // gcd
            down = self.fs // gcd
            result = resample_poly(result, up=up, down=down)

        return result

    def process_signal(self, ecg_signal):
        return np.array([self._filter(nk.ecg_invert(ecg_signal[c, :])[0]) for c in range(ecg_signal.shape[0])])


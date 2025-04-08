#  Licensed under the Creative Common CC BY-NC-SA 4.0 International License (the "License");
#  you may not use this file except in compliance with the License. The full text of the License is
#  provided in the included LICENSE file. If this file is not available, you may obtain a copy of the
#  License at
#
#       https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
#
#  Unless required by applicable law or agreed to in writing, software distributed under the License
#  is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#  express or implied. See the License for the specific language governing permissions and limitations
#  under the License.

from .SignalPreprocessor import SignalPreprocessor
from scipy.ndimage import median_filter
import numpy as np
import neurokit2 as nk


class MedianFilter(SignalPreprocessor):
    """
    This signal processor models noise by apply 600ms and 200ms median filters sequentially, then subtracting the
    result for the original signal. Finally, a 12th order low-pass Butterworth filter is applied with 35hz cutoff.
    Optionally, if target_fs is not equal to fs, the signal is resampled to the target_fs.

    Parameters
    ----------
    fs : int
        The sampling frequency of the signal.
    """
    def __init__(self, fs=256, child_preprocessor=None, parent_preprocessor=None):
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


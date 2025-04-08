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

from scipy.signal import medfilt, butter, filtfilt
import numpy as np
import neurokit2 as nk

class LowPass(SignalPreprocessor):
    """
    This signal processor applies a lowpass Butterworth filter with the specified cutoff frequency and order.

    Parameters
    ----------
    fs : int
        The sampling frequency of the signal.

    freq : int
        The cutoff frequency of the filter.

    order : int
        The order of the filter.
    """
    def __init__(self, fs=256, freq=35, order=12, child_preprocessor=None, parent_preprocessor=None):
        super().__init__(child_preprocessor=child_preprocessor, parent_preprocessor=parent_preprocessor)
        self.fs=fs
        self.freq=freq
        self.order=order

        self._nyq = 0.5 * self.fs
        self._b, self._a = butter(N=self.order, Wn=self.freq / nyq, btype='low', analog=False)

    def _filter(self, signal):
        result = filtfilt(self._b, self._a, signal)
        return result

    def process_signal(self, ecg_signal):
        return np.array([self._filter(nk.ecg_invert(ecg_signal[c, :])[0]) for c in range(ecg_signal.shape[0])])


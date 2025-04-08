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

import math
from scipy.signal import resample_poly
import numpy as np
import neurokit2 as nk

class Resample(SignalPreprocessor):
    """
    This signal processor will up- or down-sample the signal from its original sampling frequency, fs, to the new
    sampling frequency, target_fs, using a polynomial resampling from sklearn.

    Parameters
    ----------
    fs : int
        The sampling frequency of the original signal.

    target_fs : int
        The target sampling frequency for the result

    parent_preprocessor : SignalPreprocessor
        The parent preprocessor of this preprocessor.

    child_preprocessor : SignalPreprocessor
        The child preprocessor of this preprocessor.
    """
    def __init__(self, fs=256, target_fs=256, child_preprocessor=None, parent_preprocessor=None):
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


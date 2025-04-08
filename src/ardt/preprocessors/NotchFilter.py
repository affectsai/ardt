#  Copyright (c) 2024. Affects AI LLC
#
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

import numpy as np
from scipy.signal import iirnotch, filtfilt
from ardt.preprocessors import SignalPreprocessor


class NotchFilter(SignalPreprocessor):
    """
    Applies a notch filter to the signal at a specified frequency using scipy.signal.iirnotch.

    Parameters
    ----------
    fs : int
        The sampling frequency of the signal.

    freq : float
        The frequency to remove from the signal.

    quality : float
        The quality factor of the notch filter.

    parent_preprocessor : SignalPreprocessor
        The parent preprocessor of this preprocessor.

    child_preprocessor : SignalPreprocessor
        The child preprocessor of this preprocessor.
    """

    def __init__(self, fs, freq=60.0, quality=30.0, parent_preprocessor=None, child_preprocessor=None):
        """
        A notch filter preprocessor to remove powerline noise at a specified frequency.

        Parameters:
        - signal (np.ndarray): 1D input signal
        - fs (float): Sampling frequency in Hz
        - freq (float): Frequency to remove (default is 60Hz)
        - quality (float): Quality factor, higher = narrower notch (default is 30)

        Returns:
        - filtered_signal (np.ndarray): Signal after notch filtering
        """
        super().__init__(parent_preprocessor, child_preprocessor)
        self._b, self._a = iirnotch(w0=freq, Q=quality, fs=fs)

    def process_signal(self, signal):
        return np.array([filtfilt(self._b, self._a, signal[channel]) for channel in range(signal.shape[0])])

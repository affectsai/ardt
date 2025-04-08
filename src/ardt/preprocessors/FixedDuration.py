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

from .SignalPreprocessor import SignalPreprocessor


class FixedDuration(SignalPreprocessor):
    """
    Preprocesses the signal to a fixed duration. If signal is less than signal_duration, it will be padded on the
    left with the padding_value.

    Parameters
    ----------
    signal_duration : float
        Target signal length, in seconds

    sample_rate:
        target signal sample rate in Hz

    padding_value : float
        the value to pad incase signal is shorter than the target duration. If None, the mean signal value is used.

    parent_preprocessor : SignalPreprocessor
        The parent preprocessor of this preprocessor.

    child_preprocessor : SignalPreprocessor
        The child preprocessor of this preprocessor.
    """
    def __init__(self, signal_duration=45, sample_rate=256, padding_value=None,
                 parent_preprocessor=None, child_preprocessor=None, verbose=False):
        super().__init__(parent_preprocessor, child_preprocessor)
        self.signal_duration = signal_duration
        self.sample_rate = sample_rate
        self.default_padding_value = padding_value
        self._verbose = verbose

    def process_signal(self, signal):
        """
        Preprocesses the signal to a fixed duration. If signal is less than signal_duration, it will be padded on the
        left with the padding_value.

        :param signal: The signal to trim, with size NxM where N is the number of channels, and M is the number of samples.
        :return:
        """
        num_channels = signal.shape[0]
        num_samples = signal.shape[1]
        target_samples = self.signal_duration * self.sample_rate

        result = None
        if num_samples >= target_samples:
            result = signal[:, np.arange(num_samples - target_samples, num_samples)]
        else:
            padding_value = self.default_padding_value
            if padding_value is None:
                padding_value = np.mean(signal, axis=1)
            elif np.isscalar(padding_value):
                padding_value = np.ones(num_channels) * padding_value
            result = np.concatenate([np.ones((num_channels, target_samples - num_samples)) *
                                   padding_value.reshape(-1, 1), signal], axis=1)

        if self._verbose:
            print(result.shape)

        return result
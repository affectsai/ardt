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

import neurokit2 as nk2
import numpy as np

from ardt.preprocessors import SignalPreprocessor


class NK2SignalFilter(SignalPreprocessor):
    """
    Applies the Neurokit2 signal_filter method to the signal
    """

    def __init__(self, sampling_rate, lowcut=None, highcut=None, method='butterworth', order=2,
                 window_size='default', powerline=60, parent_preprocessor=None, child_preprocessor=None):
        super().__init__(parent_preprocessor, child_preprocessor)
        self._sampling_rate = sampling_rate
        self._lowcut = lowcut
        self._highcut = highcut
        self._method = method
        self._order = order
        self._window_size = window_size
        self._powerline = powerline

    def process_signal(self, signal):
        return np.array(
            nk2.signal_filter(signal,
                              sampling_rate=self._sampling_rate,
                              lowcut=self._lowcut,
                              highcut=self._highcut,
                              method=self._method,
                              order=self._order,
                              window_size=self._window_size,
                              powerline=self._powerline))

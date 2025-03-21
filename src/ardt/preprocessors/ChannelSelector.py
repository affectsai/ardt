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

from ardt.preprocessors import SignalPreprocessor


class ChannelSelector(SignalPreprocessor):
    """
    Use this to select specific channels from the signal. Useful for removing timestamp data, or narrowing down
    channels in use for high-channel data.
    """
    def __init__(self, child_preprocessor=None, parent_preprocessor=None, channels_first=True, channels=None):
        super().__init__(child_preprocessor=child_preprocessor, parent_preprocessor=parent_preprocessor)
        self.channels = channels
        self.channels_first = channels_first

    def process_signal(self, ecg_signal):
        if self.channels is None:
            self.channel = np.arange(1, signal.shape[0])

        if self.channels_first:
            return ecg_signal[self.channels, :]
        else:
            return ecg_signal[:, self.channels]


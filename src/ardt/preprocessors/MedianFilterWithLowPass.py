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

from .MedianFilter import MedianFilter
from .LowPass import LowPass
from .Resample import Resample
from .SignalPreprocessor import SignalPreprocessor


class MedianFilterWithLowPass(SignalPreprocessor):
    """
    This signal processor models noise by apply 600ms and 200ms median filters sequentially, then subtracting the
    result for the original signal. Finally, a 12th order low-pass Butterworth filter is applied with 35hz cutoff.
    Optionally, if target_fs is not equal to fs, the signal is resampled to the target_fs.

    This is a convenience wrapper around MedianFilter, LowPass and Resample and is the same as this::
        MedianFilter(
            fs=fs,
            child_preprocessor=LowPass(
                fs=fs,
                freq=freq,
                child_preprocessor=Resample(
                    fs=fs,
                    target_fs=target_fs,
                    child_preprocessor=child_preprocessor
                )
            )
        )


    """
    def __init__(self, fs=256, target_fs=256, freq=35.0, child_preprocessor=None, parent_preprocessor=None):
        super().__init__(child_preprocessor=child_preprocessor, parent_preprocessor=parent_preprocessor)

        self.impl = MedianFilter(
            fs=fs,
            child_preprocessor=LowPass(
                fs=fs,
                freq=freq,
                child_preprocessor=Resample(
                    fs=fs,
                    target_fs=target_fs,
                    child_preprocessor=child_preprocessor
                )
            )
        )

    def process_signal(self, ecg_signal):
        return self.impl.process_signal(ecg_signal)
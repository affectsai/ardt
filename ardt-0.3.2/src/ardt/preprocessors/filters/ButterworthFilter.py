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

from scipy.signal import butter, lfilter, sosfilt, zpk2sos

from ardt.preprocessors import SignalPreprocessor


class ButterworthFilter(SignalPreprocessor):
    def __init__(self, order, frequencies, btype, analog=False, output='ba', Fs=None,
                 parent_preprocessor=None, child_preprocessor=None):
        """
        See :scipy.signal.butter:`scipy.signal.butter` for parameter details.

        :param order: The order of the filter (N)
        :param frequencies: The critical frequencies (Wn)
        :param btype: The type of filter: 'lowpass','highpass','bandstop','bandpass'
        :param analog: When true, applies an analog filter, otherwise digital
        :param output: Type of output
        :param Fs: The sampling frequency of the digital system
        :param parent_preprocessor:
        """
        super().__init__(parent_preprocessor, child_preprocessor)
        self._order = order
        self._frequencies = frequencies
        self._btype = btype
        self._analog = analog
        self._output = output
        self._Fs = Fs

    def process_signal(self, signal):
        design = butter(
            self._order,
            self._frequencies,
            btype=self._btype,
            analog=self._analog,
            output=self._output,
            fs=self._Fs)

        if self._output == 'ba':
            return lfilter(design[0], design[1], signal, axis=1)
        elif self._output == 'sos':
            return sosfilt(design, signal, axis=1)
        elif self._output == 'zpk':
            sos = zpk2sos(design[0], design[1], design[2])
            return sosfilt(sos, signal, axis=1)
        else:
            raise ValueError('Unknown output type for butterworth filter: {}'.format(self._output))

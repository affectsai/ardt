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


class NK2ECGProcess(SignalPreprocessor):
    """
    Applies the Neurokit2 automated pipeline for preprocessing an ECG signal. See
    https://neuropsychology.github.io/NeuroKit/functions/ecg.html#ecg-process for more information.
    """

    def __init__(self, sampling_rate, method='neurokit', parent_preprocessor=None, child_preprocessor=None):
        super().__init__(parent_preprocessor, child_preprocessor)
        self._sampling_rate = sampling_rate
        self._method = method

    def process_signal(self, signal):
        """
        Applies the Neurokit2 ecg_process method to the signal. All results from the DataFrame returned by
        NeuroKit2 are available in the preprocessor context after this method returns.'

        :param signal:
        :return: The 'ECG_Clean' result from the DataFrame returned by Neurokit2
        """
        self.context.update(nk2.ecg_process(signal, sampling_rate=self._sampling_rate, method=self._method))
        return np.array(self.context['ECG_Clean'])

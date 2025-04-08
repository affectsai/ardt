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
import neurokit2 as nk

class NK2ECGCleaner(SignalPreprocessor):
    """
    Applies a signal preprocessor to the signal using NeuroKit2's ecg_clean function. See
    https://neuropsychology.github.io/NeuroKit/ for more.

    Parameters
    ----------
    fs : int
        The sampling frequency of the original signal.

    channels_first : bool
        True if the signal is channels first, False if channels last. Defaults to True

    powerline : int
        The AC Powerline frequency, used for filtering powerline interference from the signal

    method : str
        One of the following methods provided by NeuroKit2.

        * ``'neurokit'`` (default): 0.5 Hz high-pass butterworth filter (order = 5), followed by
          powerline filtering (see ``signal_filter()``). By default, ``powerline = 50``.
        * ``'biosppy'``: Method used in the BioSPPy package. A FIR filter ([0.67, 45] Hz; order = 1.5 *
          SR). The 0.67 Hz cutoff value was selected based on the fact that there are no morphological
          features below the heartrate (assuming a minimum heart rate of 40 bpm).
        * ``'pantompkins1985'``: Method used in Pan & Tompkins (1985). **Please help providing a better
          description!**
        * ``'hamilton2002'``: Method used in Hamilton (2002). **Please help providing a better
          description!**
        * ``'elgendi2010'``: Method used in Elgendi et al. (2010). **Please help providing a better
          description!**
        * ``'engzeemod2012'``: Method used in Engelse & Zeelenberg (1979). **Please help providing a
          better description!**
        * ``'vg'``: Method used in Visibility Graph Based Detection Emrich et al. (2023)
          and Koka et al. (2022). A 4.0 Hz high-pass butterworth filter (order = 2).
    parent_preprocessor : SignalPreprocessor
        The parent preprocessor of this preprocessor.

    child_preprocessor : SignalPreprocessor
        The child preprocessor of this preprocessor.
    """

    def __init__(self, fs=256, channels_first=True, powerline=60, method="neurokit2", child_preprocessor=None, parent_preprocessor=None):
        super().__init__(child_preprocessor=child_preprocessor, parent_preprocessor=parent_preprocessor)
        self.fs = fs
        self.channels_first = channels_first
        self.powerline = powerline
        self.method = method

    def process_signal(self, ecg_signal):
        if self.channels_first:
            filtered_signal = np.array([
                nk.ecg_clean(nk.ecg_invert(ecg_signal[c, :], sampling_rate=self.fs)[0], method=self.method,
                             sampling_rate=self.fs, powerline=self.powerline)
                for c in range(ecg_signal.shape[0])])
        else:
            filtered_signal = np.array([
                nk.ecg_clean(nk.ecg_invert(ecg_signal[:, c], sampling_rate=self.fs)[0], method=self.method,
                             sampling_rate=self.fs, powerline=self.powerline)
                for c in range(ecg_signal.shape[0])])

        return filtered_signal
from .SignalPreprocessor import SignalPreprocessor
import neurokit2 as nk

class NK2ECGCleaner(SignalPreprocessor):
    def __init__(self, child_preprocessor=None, parent_preprocessor=None, fs=256, channels_first=True, powerline=60, method="neurokit2"):
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
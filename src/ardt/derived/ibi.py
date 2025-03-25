
from ardt.datasets import AERTrial
from ardt.datasets.AERDerivedSignal import AERDerivedSignal

import numpy as np
import neurokit2 as nk

class IBISignal(AERDerivedSignal):
    def __init__(self, target_ibi_sample_rate=4):
        super(IBISignal, self).__init__(name="IBI", description="Interbeat interval derived from ECG", source_signal="ECG")
        self.ibi_sample_rate = target_ibi_sample_rate

    def to_ibi(self, channel_data, ecg_sample_rate, ecg_signal_duration):
        _, ecg_info = nk.ecg_process(channel_data, sampling_rate=ecg_sample_rate)
        rpeaks = ecg_info['ECG_R_Peaks']
        rpeak_times = rpeaks[1:] / ecg_sample_rate
        ibi_seconds = np.diff(rpeaks) / ecg_sample_rate
        interp_times = np.arange(0, ecg_signal_duration, 1 / self.ibi_sample_rate)
        ibi_timeseries = np.interp(interp_times, rpeak_times, ibi_seconds)
        return ibi_timeseries


    def get_derived_signal(self, trial: AERTrial):
        ecg = trial.load_signal_data(self.source_signal)
        ecg_metadata = trial.get_signal_metadata(self.source_signal)
        ecg_fs = ecg_metadata['sample_rate']
        ecg_n_channels = ecg_metadata['n_channels']
        ecg_signal_duration = len(ecg[0])/ecg_fs

        return np.array([
            self.to_ibi(
                channel_data=ecg[channel],
                ecg_sample_rate=ecg_fs,
                ecg_signal_duration=ecg_signal_duration
            ) for channel in range(ecg_n_channels)
        ])
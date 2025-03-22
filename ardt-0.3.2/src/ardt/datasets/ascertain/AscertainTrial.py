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

from datetime import datetime, timedelta

import numpy as np
import scipy.io

from ardt.datasets import AERTrial
from ardt.datasets.AERTrial import TruthType

ASCERTAIN_ECG_SAMPLE_RATE = 256
ASCERTAIN_ECG_N_CHANNELS = 2


class AscertainTrial(AERTrial):
    def __init__(self, dataset, participant_id, movie_id, quadrant):
        super().__init__(dataset, participant_id, movie_id)
        self._ecg_signal_duration = None
        self._truth = quadrant

    def load_ground_truth(self, truth=TruthType.QUADRANT):
        quad = self._truth
        response = quad
        if truth == TruthType.AROUSAL:
            response = AERTrial.quadrant_to_arousal(quad)
        elif truth == TruthType.VALENCE:
            response = AERTrial.quadrant_to_valence(quad)

        return response

    def get_signal_metadata(self, signal_type):
        dataset_meta = self.dataset.get_signal_metadata(signal_type)
        if signal_type == 'ECG':
            dataset_meta['duration'] = self._ecg_signal_duration
        return dataset_meta

    def load_raw_signal_data(self, signal_type):
        if signal_type not in self.signal_types:
            raise ValueError('load_signal_data not implemented for signal type {}'.format(signal_type))

        result = np.load(self.dataset.get_working_path(
            trial_participant_id=self.participant_id,
            trial_media_id=self.media_id,
            signal_type=signal_type
        ))
        self._trial_duration = result.shape[1] / ASCERTAIN_ECG_SAMPLE_RATE

        return result
        #
        # signal_data_file = self._signal_data_files[signal_type]
        # matfile = scipy.io.loadmat(signal_data_file)
        # print(signal_data_file)
        # if signal_type == 'ECG':
        #     data = self._load_ecg_signal_data(matfile)
        #     self._ecg_signal_duration = data.shape[1] / ASCERTAIN_ECG_SAMPLE_RATE
        #     return data
        # elif signal_type == 'GSR':
        #     return self._load_gsr_signal_data(matfile)
        # elif signal_type == 'EEG':
        #     return self._load_eeg_signal_data(matfile)
        # else:
        #     raise ValueError('_load_signal_data not implemented for signal type {}'.format(signal_type))



    @property
    def participant_response(self):
        self.load_ground_truth()

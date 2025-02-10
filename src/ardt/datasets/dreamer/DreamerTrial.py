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

from ardt.datasets import AERTrial
from pathlib import Path

from ardt.datasets.AERTrial import TruthType

DREAMER_ECG_SAMPLE_RATE = 256
DREAMER_ECG_N_CHANNELS = 2


class DreamerTrial(AERTrial):
    def __init__(self, dataset, participant_id, movie_id):
        super().__init__(dataset, participant_id, movie_id)

    def _to_quadrant(self,a,v):
        q=-1
        if a >= 3:  # A is high
            if v >= 3:  # A is High, V is Neg = Quad 0
                q = 1
            else:
                q = 2
        else:
            if v < 3:
                q = 3
            else:
                q = 4
        return q

    def load_raw_signal_data(self, signal_type):
        if signal_type == 'ECG':
            signal = np.load(self.dataset.get_working_path(self.participant_id, self.media_id, signal_type))
            time_steps = (np.arange(0, signal.shape[0]) * 1000 / 256).reshape(-1, 1)
            result = np.append(time_steps, signal, axis=1)
            return result.transpose()
        else:
            raise ValueError('load_signal_data not implemented for signal type {}'.format(signal_type))

    def get_signal_metadata(self, signal_type):
        dataset_meta = self.dataset.get_signal_metadata(signal_type)
        if signal_type == 'ECG':
            dataset_meta['duration'] = self._ecg_signal_duration
        return dataset_meta

    def load_ground_truth(self, truth=TruthType.QUADRANT):
        participant_path = self.dataset.get_working_path(self.participant_id)
        if participant_path is None:
            return 0

        ar=np.load(participant_path / Path('arousal.npy'))
        va=np.load(participant_path / Path('valence.npy'))
        quad = self._to_quadrant(ar[self.media_id - self.dataset.media_file_offset - 1], va[self.media_id - self.dataset.media_file_offset - 1])

        response = quad
        if truth == TruthType.AROUSAL:
            response = AERTrial.quadrant_to_arousal(quad)
        elif truth == TruthType.VALENCE:
            response = AERTrial.quadrant_to_valence(quad)

        return response

    def get_signal_metadata(self, signal_type):
        dataset_meta = self.dataset.get_signal_metadata(signal_type)
        return dataset_meta

    @property
    def participant_response(self):
        self.load_ground_truth()

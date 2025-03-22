#  Copyright (c) 2025. Affects AI LLC
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

SAMPLE_RATE = 256



class CuadsTrial(AERTrial):
    def __init__(self, dataset, participant_id, movie_id, truth, shared_cache=None):
        super().__init__(dataset, participant_id, movie_id)
        self._truth = truth
        self._trial_duration = 0
        self._shared_cache = shared_cache
        self.signal_types=['ECG','ECGHR','GSR','PPG','PPGHR']

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
        dataset_meta['duration'] = self._trial_duration
        return dataset_meta

    def load_raw_signal_data(self, signal_type):
        if signal_type not in self.signal_types:
            raise ValueError('load_signal_data not implemented for signal type {}'.format(signal_type))

        result = np.load(self.dataset.get_working_path(
            trial_participant_id=self.participant_id,
            dataset_media_name=self.media_name,
            signal_type=signal_type
        ))
        self._trial_duration = result.shape[1] / SAMPLE_RATE

        return result

    @property
    def participant_response(self):
        self.load_ground_truth()


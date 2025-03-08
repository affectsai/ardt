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


import logging
import os
from pathlib import Path

from tensorboard.plugins.projector.projector_plugin import LRUCache

from ardt import config
from ardt.datasets import AERDataset
from .CuadsTrial import CuadsTrial
import numpy as np

CONFIG = config['datasets']['cuads']
DEFAULT_DATASET_PATH = Path(CONFIG['path'])
CUADS_NUM_MEDIA_FILES   = 20
CUADS_NUM_PARTICIPANTS  = 38     # There are only 38, but they're still numbered 1 to 44.
CUADS_MAX_PARTICIPANT_NUM = 44
CUADS_NUM_TRIALS        = 714    # The real number of trials in the CUADS Data Set
CUADS_SAMPLE_RATE       = 256

logger = logging.getLogger('CuadsDataset')
logger.level = logging.DEBUG

expected_classifications = {
            'video_55': 2,
            'video_79': 1,
            'video_111': 3,
            'video_73': 2,
            'video_52': 4,
            'video_146': 3,
            'video_funny_f': 1,
            'video_80': 1,
            'video_cats_f': 1,
            'video_138': 3,
            'video_69': 2,
            'video_dallas_f': 4,
            'video_detroit_f': 4,
            'video_53': 4,
            'video_58': 4,
            'video_30': 2,
            'video_earworm_f': 2,
            'video_newyork_f': 4,
            'video_90': 1,
            'video_107': 2
        }

default_signal_metadata = {
            'ECG': {
                'sample_rate': CUADS_SAMPLE_RATE,
                'n_channels': 3,
            },
            'GSR': {
                'sample_rate': CUADS_SAMPLE_RATE,
                'n_channels': 2,
            },
            'PPG': {
                'sample_rate': CUADS_SAMPLE_RATE,
                'n_channels': 1,
            }
        }

CUADS_COLUMN_MAP = {
    "SEGMENT_ECG_TIMESTAMP": 0,
    "SEGMENT_GSR_TIMESTAMP": 1,
    "SEGMENT_ECG_LARA":     15,
    "SEGMENT_ECG_LLLA":     16,
    "SEGMENT_ECG_LLRA":     17,
    "SEGMENT_ECG_HR_LARA":  19,
    "SEGMENT_ECG_HR_LLLA":  20,
    "SEGMENT_ECG_HR_LLRA":  21,
    "SEGMENT_GSR_SC":       37,
    "SEGMENT_GSR_SR":       38,
    "SEGMENT_PPG":          45,
    "SEGMENT_PPG_IBI":      46,
    "SEGMENT_PPG_HR":       47,
}
class CuadsDataset(AERDataset):
    def __init__(self, dataset_path=None, participant_offset=0, mediafile_offset=0):
        """
        Construct a new CuadsDataset object..

        :param path: Path to the extracted CUADS dataset
        :param signals: A list of signals to load, e.g. ['ECG','EEG'] to load ECG and EEG data. If None, the folder
        given by dataset_path will be checked for subfolders named '<SIGNAL>Data', and each one found will be loaded.
        For example, if dreamer/ECGData/ is found, the ECG is automatically loaded.
        :param participant_offset: Constant value added to each participant identifier within this dataset. For example,
        if participant_offset is 32, then Participant 1 from this dataset's raw data will be returned as Participant 33.
        :param mediafile_offset: Constant value added to each media identifier within this dataset. For example, if
        mediafile_offset is 12, then Movie 1 from this dataset's raw data will be reported as Media ID 13.
        """
        signals = ['ECG', 'PPG', 'GSR', 'ECGHR', 'PPGHR']
        super().__init__(signals=signals,
                         participant_offset=participant_offset,
                         mediafile_offset=mediafile_offset,
                         signal_metadata=default_signal_metadata,
                         expected_responses=expected_classifications)

        if dataset_path is None:
            dataset_path = DEFAULT_DATASET_PATH

        if not os.path.exists(dataset_path):
            raise ValueError(
                f'Invalid path to DREAMER dataset: {dataset_path}. Please correct and try again.')

        logger.info(f'Loading CUADS from {dataset_path} with signals {signals}.')
        self.media_index_map = {}               # Maps media name to int index
        self.media_index_to_name = {}           # Maps media index back to name
        self.participant_id_map = {}            # Maps participant number to int index
        self.dataset_path = Path(dataset_path)
        self._trial_cache = LRUCache(10)


    def _preload_dataset(self):
        all_trials = {}
        for p in range(CUADS_MAX_PARTICIPANT_NUM):
            cuads_participant_number = p + 1
            participant_id = f'CUADS_{cuads_participant_number:03}'
            participant_folder = os.path.join(self.dataset_path, participant_id)
            response_file = os.path.join(participant_folder, 'responses.csv')
            if not os.path.exists(response_file):
                continue

            if cuads_participant_number not in self.participant_id_map:
                self.participant_id_map[cuads_participant_number] = len(self.participant_id_map) + 1
            dataset_participant_number = self.participant_id_map[cuads_participant_number] #+ self.participant_offset


            # Load this participant's responses...
            responses = np.loadtxt(response_file, delimiter=',', dtype=str, skiprows=1)
            for response_number, response in enumerate(responses):
                movie_name = response[0]
                segmented_data_filepath = os.path.join(participant_folder, 'segmented', f'{movie_name}_sessiondata.csv')
                segment_data = np.loadtxt(segmented_data_filepath, delimiter=',', dtype=str, skiprows=1)

                path = self.get_working_path(dataset_participant_id=dataset_participant_number, dataset_media_name=movie_name, signal_type='ECG')
                data = np.array(segment_data[:, [CUADS_COLUMN_MAP["SEGMENT_ECG_TIMESTAMP"],
                                                 CUADS_COLUMN_MAP["SEGMENT_ECG_LARA"],
                                                 CUADS_COLUMN_MAP["SEGMENT_ECG_LLLA"],
                                                 CUADS_COLUMN_MAP["SEGMENT_ECG_LLRA"]]], dtype=float)
                data = data.transpose()
                np.save(path, data)

                path = self.get_working_path(dataset_participant_id=dataset_participant_number, dataset_media_name=movie_name, signal_type='ECGHR')
                data = np.array(segment_data[:, [CUADS_COLUMN_MAP["SEGMENT_ECG_TIMESTAMP"],
                                                 CUADS_COLUMN_MAP["SEGMENT_ECG_HR_LARA"],
                                                 CUADS_COLUMN_MAP["SEGMENT_ECG_HR_LLLA"],
                                                 CUADS_COLUMN_MAP["SEGMENT_ECG_HR_LLRA"]]], dtype=float)
                data = data.transpose()
                np.save(path, data)

                path = self.get_working_path(dataset_participant_id=dataset_participant_number, dataset_media_name=movie_name, signal_type='GSR')
                data = np.array(segment_data[:, [CUADS_COLUMN_MAP["SEGMENT_GSR_TIMESTAMP"],
                                                 CUADS_COLUMN_MAP["SEGMENT_GSR_SC"],
                                                 CUADS_COLUMN_MAP["SEGMENT_GSR_SR"]]], dtype=float)
                data = data.transpose()
                np.save(path, data)

                path = self.get_working_path(dataset_participant_id=dataset_participant_number, dataset_media_name=movie_name, signal_type='PPG')
                data = np.array(segment_data[:, [CUADS_COLUMN_MAP["SEGMENT_GSR_TIMESTAMP"],
                                                 CUADS_COLUMN_MAP["SEGMENT_PPG"]]], dtype=float)
                data = data.transpose()
                np.save(path, data)

                path = self.get_working_path(dataset_participant_id=dataset_participant_number, dataset_media_name=movie_name, signal_type='PPGHR')
                data = np.array(segment_data[:, [CUADS_COLUMN_MAP["SEGMENT_GSR_TIMESTAMP"],
                                                 CUADS_COLUMN_MAP["SEGMENT_PPG_HR"]]], dtype=float)
                data = data.transpose()
                np.save(path, data)

    def _load_trials(self):
        response_movie_name = 0
        response_valence = 1
        response_arousal = 2

        def _to_quadrant(a, v):
            q = -1
            if a >= 5:
                if v >= 5:
                    q = 1
                else:
                    q = 2
            else:
                if v < 5:
                    q = 3
                else:
                    q = 4
            return q

        # Load trial data...
        all_trials = {}
        for p in range(CUADS_MAX_PARTICIPANT_NUM):
            cuads_participant_number = p + 1
            participant_id = f'CUADS_{cuads_participant_number:03}'
            participant_folder = os.path.join( self.dataset_path, participant_id )
            response_file = os.path.join( participant_folder, 'responses.csv' )
            if not os.path.exists(response_file):
                continue

            if cuads_participant_number not in self.participant_id_map:
                self.participant_id_map[cuads_participant_number] = len(self.participant_id_map) + 1
            dataset_participant_number = self.participant_id_map[cuads_participant_number] #+ self.participant_offset

            self.participant_ids.add(dataset_participant_number)
            if participant_id not in all_trials.keys():
                all_trials[participant_id] = {}

            # Load this participant's responses...
            responses = np.loadtxt(response_file, delimiter=',', dtype=str, skiprows=1)
            for response_number, response in enumerate(responses):
                movie_name = response[response_movie_name]
                segmented_data_filepath = os.path.join(participant_folder, 'segmented', f'{movie_name}_sessiondata.csv')

                if not os.path.exists(segmented_data_filepath):
                    continue

                if movie_name not in self.media_index_map:
                    self.media_index_map[movie_name] = len(self.media_index_map) + 1

                movie_id = self.media_index_map[movie_name] #+ self.media_file_offset
                self.media_ids.add(movie_id)
                self.media_index_to_name[movie_id] = movie_name

                trial = CuadsTrial(self,
                               dataset_participant_number,
                               movie_id,
                               _to_quadrant(float(response[response_arousal]), float(response[response_valence])),
                               shared_cache=self._trial_cache)
                trial.signal_preprocessors = self.signal_preprocessors
                self.trials.append(trial)

    def _post_load_trials(self):
        pass

    def get_media_name_by_movie_id(self, movie_id):
        return self.media_index_to_name[movie_id]


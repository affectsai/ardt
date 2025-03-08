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

import logging
import os.path
from pathlib import Path

import ijson
import numpy as np

from ardt import config
from ardt.datasets import AERDataset
from ardt.datasets.cuads.CuadsDataset import default_signal_metadata

from .DreamerTrial import DreamerTrial

CONFIG = config['datasets']['dreamer']
DEFAULT_DREAMER_PATH = Path(CONFIG['path'])
DEFAULT_DREAMER_FILENAME = Path(CONFIG['dreamer_data_filename'])
DREAMER_NUM_MEDIA_FILES = 18
DREAMER_NUM_PARTICIPANTS = 23
DREAMER_ALL_SIGNALS = {'ECG', 'EEG'}

logger = logging.getLogger('DreamerDataset')
logger.level = logging.DEBUG

expected_classifications = {
            # DREAMER: A Database for Emotion Recognition Through EEG and ECG Signals from Wireless Low-cost Off-the-Shelf Devices
            1: 2,   # Searching for Bobby Fisher
            2: 1,   # D.O.A
            3: 1,   # The Hangover
            4: 2,   # The Ring
            5: 1,   # 300
            6: 2,   # National Lampoon's Van Wilder
            7: 1,   # Wall-E
            8: 2,   # Crash
            9: 2,   # My Girl
            10: 2,  # The Fly
            11: 4,  # Pride and Prejudice
            12: 4,  # Modern Times
            13: 1,  # Remember the Titans
            14: 3,  # Gentlemans Agreement
            15: 2,  # Phsycho
            16: 1,  # The Bourne IDentity
            17: 3,  # The Shawshank Redemption
            18: 2,  # The Departed
        }

default_signal_metadata = {
    'ECG': {
        'sample_rate': 256,
        'n_channels': 2,
    }
}

class DreamerDataset(AERDataset):
    def __init__(self, dataset_path=None, signals=None, participant_offset=0, mediafile_offset=0,
                 dataset_fname=DEFAULT_DREAMER_FILENAME):
        """
        Construct a new DreamerDataset object using the given path.

        :param dataset_path: Path to the folder containing the dataset_fname file.
        :param signals: A list of signals to load, e.g. ['ECG','EEG'] to load ECG and EEG data. If None, all available
        signals are loaded.
        :param participant_offset: Constant value added to each participant identifier within this dataset. For example,
        if participant_offset is 32, then Participant 1 from this dataset's raw data will be returned as Participant 33.
        :param mediafile_offset: Constant value added to each media identifier within this dataset. For example, if
        mediafile_offset is 12, then Movie 1 from this dataset's raw data will be reported as Media ID 13.
        """
        if signals is None:
            signals = list(DREAMER_ALL_SIGNALS)
        else:
            for signal in signals:
                if signal not in DREAMER_ALL_SIGNALS:
                    raise ValueError(
                        f'{signal} signal does not exist in DREAMER. Please correct and try again.')

        super().__init__(signals=signals,
                         participant_offset=participant_offset,
                         mediafile_offset=mediafile_offset,
                         signal_metadata=default_signal_metadata,
                         expected_responses=expected_classifications)


        if dataset_path is None:
            dataset_path = CONFIG.get('path')

        if dataset_path is None or not os.path.exists(dataset_path):
            raise ValueError(
                f'Invalid path to DREAMER dataset: {dataset_path}. Please correct and try again.')

        logger.debug(f'Loading DREAMER from {dataset_path}/{dataset_fname} with signals {signals}.')
        self._dataset_file = Path(dataset_path) / Path(dataset_fname)

        if not self._dataset_file.exists():
            raise ValueError('Path to DREAMER dataset does not exist: {}'.format(self._dataset_file.resolve()))

        self.media_index_to_name = {}           # Maps media index back to name

    def _preload_dataset(self):
        participant_id = 0
        with open(self._dataset_file, 'rb') as f:
            participant_entries = ijson.items(f, 'item')
            for participant_entry in participant_entries:
                participant_id += 1
                participant_path = self.get_working_path(participant_id)
                participant_path.mkdir(parents=True, exist_ok=True)

                for signal in self.signals:
                    baseline_signal_data = participant_entry[signal]['baseline']
                    stimuli_signal_data = participant_entry[signal]['stimuli']

                    np.save(participant_path / Path('arousal.npy'), participant_entry['ScoreArousal'])
                    np.save(participant_path / Path('valence.npy'), participant_entry['ScoreValence'])

                    for c in range(DREAMER_NUM_MEDIA_FILES):
                        media_id = c + 1
                        media_path = self.get_working_path(participant_id, media_id)
                        media_path.mkdir(parents=True, exist_ok=True)

                        np.save(media_path / Path(f'{signal}_stimuli.npy'), stimuli_signal_data[c])
                        np.save(media_path / Path(f'{signal}_baseline.npy'), baseline_signal_data[c])

    def _load_trials(self):
        for p in range(DREAMER_NUM_PARTICIPANTS):
            participant_id = p+1
            for c in range(DREAMER_NUM_MEDIA_FILES):
                media_id = c+1
                self.media_index_to_name[media_id] = media_id  # no names, just ids... 1:1 map
                trial = DreamerTrial(self, participant_id, media_id)
                trial.signal_preprocessors = self.signal_preprocessors
                for signal in self.signals:
                    trial.signal_types.add(signal)
                    trial.signal_data_files[signal] = self.get_working_path(trial.participant_id, trial.media_id, signal)
                self.trials.append(trial)

    def _post_load_trials(self):
        pass

    def get_media_name_by_movie_id(self, movie_id):
        return None


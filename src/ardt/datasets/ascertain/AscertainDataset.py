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

import os
import logging
from pathlib import Path
from typing import Optional, List

import numpy as np
import scipy

from ardt import config
from ardt.datasets import AERDataset, AERTrialFilter
from .AscertainTrial import AscertainTrial
from datetime import datetime, timedelta


CONFIG = config['datasets']['ascertain']
DEFAULT_ASCERTAIN_PATH = Path(CONFIG['path'])
ASCERTAIN_RAW_FOLDER = Path(CONFIG['raw_data_path'])
ASCERTAIN_FEATURES_FOLDER = Path(CONFIG['features_data_path'])
ASCERTAIN_NUM_MEDIA_FILES = 36
ASCERTAIN_NUM_PARTICIPANTS = 58

logger = logging.getLogger('AscertainDataset')
logger.level = logging.DEBUG

expected_classifications = {
            # Taken from DECAF+ mediafile_offset: MEG-BASED MULTIMODAL DATABASE FOR DECODING AFFECTIVE PHYSIOLOGICAL RESPONSES
            # Amusing
            1: 1,   # Ace Ventura: Pet Detective
            2: 1,   # The Gods Must Be Crazy II
            4: 1,   # Airplane
            5: 1,   # When Harry Met Sally

            # Funny:
            3: 1,   # Liar Liar
            6: 1,   # The Gods Must Be Crazy
            7: 1,   # The Hangover
            9: 1,   # Hot Shots

            # Happy:
            8: 4,   # Up
            10: 4,  # August Rush
            11: 4,  # Truman Show
            12: 4,  # Wall-E
            13: 4,  # Love Actually
            14: 4,  # Remember the Titans
            16: 4,  # Life is Beautiful
            17: 4,  # Slumdog Millionaire
            18: 4,  # House of Flying Daggers

            # Exciting
            15: 1,  # Legally Blonde
            33: 1,  # The Untouchables

            # Angry
            19: 3,  # Ghandi
            21: 3,  # Lagaan
            23: 3,  # My Bodyguard
            35: 3,  # Crash

            # Disgusting
            28: 2,  # The Exorcist
            34: 2,  # Pink Flamingos

            # Fear:
            30: 2,  # The Shining
            36: 2,  # Black Swan

            # Sad
            20: 3,  # My Girl
            22: 3,  # Bambi
            24: 3,  # Up
            25: 3,  # Life is Beautiful
            26: 3,  # Remember the Titans
            27: 3,  # Titanic
            31: 3,  # Prestige

            # Shock
            29: 2,  # Mulholland Drive
            32: 2,  # Alien
        }

default_signal_metadata = {'ECG': {
                'sample_rate': 256,
                'n_channels': 2,
            }
}

class AscertainDataset(AERDataset):
    def __init__(self, ascertain_path=None, signals=None, participant_offset=0, mediafile_offset=0):
        """
        Construct a new AscertainDataset object for a given ascertainPath.

        :param ascertain_path: Path to the extracted ASCERTAIN dataset
        :param signals: A list of signals to load, e.g. ['ECG','EEG'] to load ECG and EEG data. If None, the folder
        given by ascertain_path will be checked for subfolders named '<SIGNAL>Data', and each one found will be loaded.
        For example, if ascertain/ECGData/ is found, the ECG is automatically loaded.
        :param participant_offset: Constant value added to each participant identifier within this dataset. For example,
        if participant_offset is 32, then Participant 1 from this dataset's raw data will be returned as Participant 33.
        :param mediafile_offset: Constant value added to each media identifier within this dataset. For example, if
        mediafile_offset is 12, then Movie 1 from this dataset's raw data will be reported as Media ID 13.
        """
        super().__init__(signals=signals,
                         participant_offset=participant_offset,
                         mediafile_offset=mediafile_offset,
                         signal_metadata=default_signal_metadata,
                         expected_responses=expected_classifications)

        if ascertain_path is None:
            ascertain_path = CONFIG.get('path')

        if ascertain_path is None or not os.path.exists(ascertain_path):
            raise ValueError(
                f'Invalid path to ASCERTAIN dataset: {ascertain_path}. Please correct and try again.')

        logger.debug(f'Loading ASCERTAIN from {ascertain_path} with signals {signals}.')

        self.ascertain_path = Path(ascertain_path)
        self.ascertain_raw_path = self.ascertain_path / ASCERTAIN_RAW_FOLDER
        self.ascertain_features_path = self.ascertain_path / ASCERTAIN_FEATURES_FOLDER
        self.media_index_to_name = {}           # Maps media index back to name

        if not self.ascertain_path.exists():
            raise ValueError('Path to ASCERTAIN does not exist: {}'.format(ascertain_path))

        # Load signals
        if signals is not None:
            for signal in signals:
                if not (self.ascertain_raw_path / f'{signal}Data').exists():
                    raise ValueError(
                        f'{signal}Data does not exist, unable to load {signal} Signal. Please correct and try again.')
        else:
            for p in sorted(self.ascertain_raw_path.rglob("*Data")):
                if p.is_dir():
                    self.signals.append(str(p.name).replace("Data", ""))

    @staticmethod
    def _load_eeg_signal_data(signal_data_file):
        return []

    @staticmethod
    def _load_ecg_signal_data(signal_data_file):
        """
        Loads the ECG signal data from the given ASCERTAIN matlab file, and returns it as a 3xN
        numpy array. The first row contains the signal timestamp data in UNIX Epoch time. The
        second and third rows contain the two ECG signal channels from the dataset.

        :param signal_data_file: The ECG_ClipXX.mat file for this trial
        :return:
        """
        start_time_arr = signal_data_file['timeECG'][0]
        start_time = datetime(
            int(start_time_arr[0]),
            int(start_time_arr[1]),
            int(start_time_arr[2]),
            int(start_time_arr[3]),
            int(start_time_arr[4]),
            int(start_time_arr[5]),
            int(1000 * (start_time_arr[5] % 1))
        )

        def convert_to_epoch(_timestamp, _start_time):
            return (_start_time + timedelta(milliseconds=_timestamp)).timestamp()

        timeconverter = np.vectorize(lambda _ts: convert_to_epoch(_ts, start_time))

        ecg_data = signal_data_file['Data_ECG']
        left_arm_idx = 1 if (len(ecg_data[0]) < 6) else 4
        right_arm_idx = 2 if (len(ecg_data[0]) < 6) else 5

        ecg = ecg_data[:, [0, left_arm_idx, right_arm_idx]]
        ts = np.apply_along_axis(
            func1d=timeconverter,
            axis=0,
            arr=ecg[:, 0])
        ts = ts.reshape(-1, 1)

        result = np.append(ts, ecg[:, [1, 2]], axis=1)
        return result.transpose()

    @staticmethod
    def _load_gsr_signal_data(signal_data_file):
        return []

    def _preload_dataset(self):
        # Load ascertain data files...
        # Map< participantId, Map< movieId, data_file_path >>
        dt_selfreports_path = os.path.join(self.ascertain_features_path, "Dt_SelfReports.mat")
        dt_selfreports = scipy.io.loadmat(dt_selfreports_path)

        for matlab_file in self.ascertain_raw_path.rglob("*Clip*.mat"):
            movie_folder = matlab_file.parents[0].name
            signal_folder = matlab_file.parents[1].name

            signal_type = signal_folder.replace("Data", "")
            if signal_type not in self._signals:
                continue

            dataset_participant_id = int(movie_folder.split("_P")[1])  # + self.participant_offset
            dataset_movie_id = int(matlab_file.name.upper().replace(f'{signal_type}_CLIP', '').replace('.MAT', ''))
            matfile_path = matlab_file.resolve()

            if signal_type not in self._signals:
                continue

            matlab_data = scipy.io.loadmat(matfile_path)

            data = None
            if signal_type == 'ECG':
                data = self._load_ecg_signal_data(matlab_data)
            elif signal_type == 'GSR':
                data = self._load_gsr_signal_data(matlab_data)
            elif signal_type == 'EEG':
                data = self._load_eeg_signal_data(matlab_data)

            preload_data_path = self.get_working_path(
                dataset_participant_id=dataset_participant_id,
                dataset_media_id=dataset_movie_id,
                signal_type=signal_type)
            np.save(preload_data_path, data)


    def _load_trials(self, trial_filters: Optional[List[AERTrialFilter]] = None):
        # Load ascertain data files...
        # Map< participantId, Map< movieId, data_file_path >>
        ascertain_datafiles = {}

        dt_selfreports_path = os.path.join(self.ascertain_features_path, "Dt_SelfReports.mat")
        dt_selfreports = scipy.io.loadmat(dt_selfreports_path)

        for matlab_file in self.ascertain_raw_path.rglob("*Clip*.mat"):
            movie_folder = matlab_file.parents[0].name
            signal_folder = matlab_file.parents[1].name

            signal_type = signal_folder.replace("Data", "")
            if signal_type not in self._signals:
                continue

            participant_id = int(movie_folder.split("_P")[1]) # + self.participant_offset
            movie_id = int(matlab_file.name.upper().replace(f'{signal_type}_CLIP', '').replace('.MAT', ''))
            # movie_id += self.media_file_offset

            self.media_index_to_name[movie_id] = movie_id   # no names, just ids... 1:1 map
            if participant_id not in ascertain_datafiles.keys():
                ascertain_datafiles[participant_id] = {}

            if movie_id not in ascertain_datafiles[participant_id].keys():
                ascertain_datafiles[participant_id][movie_id] = {}

            ascertain_datafiles[participant_id][movie_id][signal_type] = matlab_file.resolve()

        def _to_quadrant(a,v):
            q=-1
            if a >= 3:  # A is high
                if v >= 0:  # A is High, V is Neg = Quad 0
                    q = 1
                else:
                    q = 2
            else:
                if v < 0:
                    q = 3
                else:
                    q = 4
            return q

        for participant_id in ascertain_datafiles:
            for movie_id in ascertain_datafiles[participant_id]:
                arousal = dt_selfreports['Ratings'][0][participant_id - 1][movie_id - 1]
                valence = dt_selfreports['Ratings'][1][participant_id - 1][movie_id - 1]

                trial = AscertainTrial(self, participant_id, movie_id, _to_quadrant(arousal, valence))
                trial.signal_data_files = ascertain_datafiles[participant_id][movie_id]
                trial.signal_preprocessors = self.signal_preprocessors

                self.trials.append(trial)

    def _post_load_trials(self):
        pass

    def get_media_name_by_movie_id(self, movie_id):
        return None



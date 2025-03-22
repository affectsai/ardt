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
import random
import unittest
from pathlib import Path

import numpy as np
from ardt.datasets.AERTrial import TruthType

from ardt.datasets.ascertain import AscertainDataset
from ardt.datasets.ascertain.AscertainDataset import DEFAULT_ASCERTAIN_PATH, ASCERTAIN_NUM_MEDIA_FILES, \
    ASCERTAIN_NUM_PARTICIPANTS, ASCERTAIN_RAW_FOLDER

PARTICIPANT_OFFSET = 50
MEDIAFILE_OFFSET = 20


class AscertainDatasetTest(unittest.TestCase):
    def setUp(self):
        self.dataset = AscertainDataset(DEFAULT_ASCERTAIN_PATH, signals=['ECG'],
                                            participant_offset=PARTICIPANT_OFFSET, mediafile_offset=MEDIAFILE_OFFSET)
        self.dataset.preload()
        self.dataset.load_trials()
        self.dataset_path = (DEFAULT_ASCERTAIN_PATH / ASCERTAIN_RAW_FOLDER).resolve()

    def test_ascertain_paths(self):
        """
        Asserts that the expected paths for the ASCERTAIN dataset exist...
        :return:
        """
        dataset = AscertainDataset(DEFAULT_ASCERTAIN_PATH)
        dataset.preload()
        dataset.load_trials()

        for signal in dataset.signals:
            self.assertTrue(os.path.isdir(os.path.join(self.dataset_path, f'{signal}Data')))

        for path in np.array(sorted(os.listdir(self.dataset_path))):
            if os.path.isdir(os.path.join(DEFAULT_ASCERTAIN_PATH, ASCERTAIN_RAW_FOLDER, path)):
                if path.endswith('Data'):
                    self.assertIn(path.replace("Data", ""), dataset.signals)

    def test_ascertain_dataset_load(self):
        """
        Asserts that the ASCERTAIN dataset loads the expected number of participants, movie clips, and trials.
        :return:
        """
        self.assertEqual(len(self.dataset.participant_ids), ASCERTAIN_NUM_PARTICIPANTS)
        self.assertEqual(len(self.dataset.media_ids), ASCERTAIN_NUM_MEDIA_FILES)
        self.assertEqual(len(self.dataset.trials), ASCERTAIN_NUM_MEDIA_FILES * ASCERTAIN_NUM_PARTICIPANTS)

    def test_ascertain_path_limit_signals(self):
        """
        Asserts that the ASCERTAIN data set class works properly on a restricted list of signal types, using the
        dataset created in setUp
        :return:
        """
        self.assertTrue(True, 'We made it!')
        for signal in self.dataset.signals:
            self.assertTrue(os.path.isdir(os.path.join(DEFAULT_ASCERTAIN_PATH, ASCERTAIN_RAW_FOLDER, f'{signal}Data')))

        for path in np.array(sorted(os.listdir(self.dataset_path))):
            if os.path.isdir(os.path.join(DEFAULT_ASCERTAIN_PATH, ASCERTAIN_RAW_FOLDER, path)):
                if path.endswith('Data') and not path.startswith('ECG'):
                    self.assertNotIn(path.replace("Data", ""), self.dataset.signals)

        self.assertEqual(len(self.dataset.participant_ids), ASCERTAIN_NUM_PARTICIPANTS)
        self.assertEqual(len(self.dataset.media_ids), ASCERTAIN_NUM_MEDIA_FILES)

    @staticmethod
    def bad_signal():
        return AscertainDataset(DEFAULT_ASCERTAIN_PATH, signals=['XYZ'])

    def test_ascertain_path_invalid_signal(self):
        """
        Asserts that a ValueError is thrown if AscertainDataset is constructed with a signal type that does not
        exist on the filesystem.
        :return:
        """
        self.assertRaises(ValueError, AscertainDatasetTest.bad_signal)

    def test_signal_datafiles(self):
        """
        Asserts that the all the datafiles loaded by the AscertainDataset are found.
        :return:
        """
        num_trials = 0
        num_data_files = 0
        for trial in self.dataset.trials:
            num_trials += 1
            for data_file in trial.signal_data_files.values():
                num_data_files += 1
                self.assertTrue(Path(data_file).exists(), msg=f"File {data_file} does not exist")
        self.assertEqual(num_trials, ASCERTAIN_NUM_PARTICIPANTS * ASCERTAIN_NUM_MEDIA_FILES)

    def test_ecg_signal_load(self):
        """
        Asserts that we can properly load an ECG signal from one of the dataset's trials.
        :return:
        """
        trial = self.dataset.trials[random.randint(0, len(self.dataset.trials) - 1)]
        self.assertEqual(trial.load_signal_data('ECG').shape[0], 3)

    def test_participant_id_offsets(self):
        min_id = min(self.dataset.participant_ids)
        max_id = max(self.dataset.participant_ids)

        self.assertEqual(PARTICIPANT_OFFSET + 1, min_id)
        self.assertEqual(ASCERTAIN_NUM_PARTICIPANTS, max_id - min_id + 1)

    def test_media_id_offsets(self):
        min_id = 9999999
        max_id = -1

        for media_id in sorted(self.dataset.media_ids):
            min_id = min(min_id, media_id)
            max_id = max(max_id, media_id)

        self.assertEqual(self.dataset.media_file_offset + 1, min_id)
        self.assertEqual(ASCERTAIN_NUM_MEDIA_FILES, max_id - min_id + 1)

    def test_splits(self):
        trial_splits = self.dataset.get_trial_splits([.7, .3])
        split_1_participants = set([x.participant_id for x in trial_splits[0]])
        split_2_participants = set([x.participant_id for x in trial_splits[1]])

        self.assertEqual(len(trial_splits), 2)
        self.assertEqual(len(trial_splits[0]) + len(trial_splits[1]), len(self.dataset.trials))
        self.assertEqual(0, len(split_1_participants.intersection(split_2_participants)))

    def test_three_splits(self):
        trial_splits = self.dataset.get_trial_splits([.7, .15, .15])
        split_1_participants = set([x.participant_id for x in trial_splits[0]])
        split_2_participants = set([x.participant_id for x in trial_splits[1]])
        split_3_participants = set([x.participant_id for x in trial_splits[2]])

        self.assertEqual(len(trial_splits), 3)
        self.assertEqual(len(trial_splits[0]) + len(trial_splits[1]) + len(trial_splits[2]),
                         len(self.dataset.trials))
        self.assertEqual(0, len(split_1_participants.intersection(split_2_participants)))
        self.assertEqual(0, len(split_1_participants.intersection(split_3_participants)))
        self.assertEqual(0, len(split_2_participants.intersection(split_3_participants)))


    def test_split_datasets(self):
        datasets = self.dataset.get_dataset_splits([.7, .3])
        split_1_participants = set([x.participant_id for x in datasets[0].trials])
        split_2_participants = set([x.participant_id for x in datasets[1].trials])

        self.assertEqual(len(datasets), 2)
        self.assertEqual(len(datasets[0].trials) + len(datasets[1].trials), len(self.dataset.trials))
        self.assertEqual(0, len(split_1_participants.intersection(split_2_participants)))

    def test_three_split_datasets(self):
        datasets = self.dataset.get_dataset_splits([.7, .15, .15])
        split_1_participants = set([x.participant_id for x in datasets[0].trials])
        split_2_participants = set([x.participant_id for x in datasets[1].trials])
        split_3_participants = set([x.participant_id for x in datasets[2].trials])

        self.assertEqual(len(datasets), 3)
        self.assertEqual(len(datasets[0].trials) + len(datasets[1].trials) + len(datasets[2].trials), len(self.dataset.trials))
        self.assertEqual(0, len(split_1_participants.intersection(split_2_participants)))
        self.assertEqual(0, len(split_1_participants.intersection(split_3_participants)))
        self.assertEqual(0, len(split_2_participants.intersection(split_3_participants)))



    def test_participant_ids_are_sequential(self):
        participant_ids = sorted(self.dataset.participant_ids)
        for i in range(len(participant_ids)):
            self.assertEqual(participant_ids[i], i + 1 + self.dataset.participant_offset)

    def test_media_ids_are_sequential(self):
        media_ids = sorted(self.dataset.media_ids)
        for i in range(len(media_ids)):
            self.assertEqual(media_ids[i], i + 1 + self.dataset.media_file_offset)

    def test_expected_responses(self):
        media_ids = sorted(self.dataset.media_ids)
        self.assertEqual(len(media_ids), len(self.dataset.expected_media_responses))
        for trial in self.dataset.trials:
            self.assertIsNotNone(trial.expected_response)

    def test_arousal(self):
        for trial in self.dataset.trials:
            quad = trial.load_ground_truth()
            if quad == 1 or quad == 2:
                self.assertEqual(2, trial.load_ground_truth(truth=TruthType.AROUSAL))
            elif quad == 3 or quad == 4:
                self.assertEqual(1, trial.load_ground_truth(truth=TruthType.AROUSAL))
            else:
                self.fail("Unknown trial type: " + str(quad))

    def test_valence(self):
        for trial in self.dataset.trials:
            quad = trial.load_ground_truth()
            if quad == 1 or quad == 4:
                self.assertEqual(2, trial.load_ground_truth(truth=TruthType.VALENCE))
            elif quad == 2 or quad == 3:
                self.assertEqual(1, trial.load_ground_truth(truth=TruthType.VALENCE))
            else:
                self.fail("Unknown trial type: " + str(quad))

if __name__ == '__main__':
    unittest.main()

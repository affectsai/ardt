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

import random
import unittest

from ardt.datasets.AERTrial import TruthType
from ardt.datasets.dreamer.DreamerDataset import (DEFAULT_DREAMER_PATH, DEFAULT_DREAMER_FILENAME,
                                                  DREAMER_NUM_MEDIA_FILES, DREAMER_NUM_PARTICIPANTS)
from ardt.datasets.dreamer.DreamerDataset import DreamerDataset

PARTICIPANT_OFFSET = 50
MEDIAFILE_OFFSET = 20


class DreamerDatasetTest(unittest.TestCase):
    def setUp(self):
        self.dataset = DreamerDataset(DEFAULT_DREAMER_PATH, signals=['ECG'], participant_offset=PARTICIPANT_OFFSET,
                                          mediafile_offset=MEDIAFILE_OFFSET)
        self.dataset.preload()
        self.dataset.load_trials()
        self.dataset_path = (DEFAULT_DREAMER_PATH / DEFAULT_DREAMER_FILENAME).resolve()

    def test_dataset_load(self):
        """
        Asserts that the ASCERTAIN dataset loads the expected number of participants, movie clips, and trials.
        :return:
        """
        self.assertEqual(len(self.dataset.participant_ids), DREAMER_NUM_PARTICIPANTS)
        self.assertEqual(len(self.dataset.media_ids), DREAMER_NUM_MEDIA_FILES)
        self.assertEqual(len(self.dataset.trials), DREAMER_NUM_MEDIA_FILES * DREAMER_NUM_PARTICIPANTS)

    def test_participant_id_offsets(self):
        min_id = min(self.dataset.participant_ids)
        max_id = max(self.dataset.participant_ids)

        self.assertEqual(PARTICIPANT_OFFSET + 1, min_id)
        self.assertEqual(DREAMER_NUM_PARTICIPANTS, max_id - min_id + 1)

    def test_media_id_offsets(self):
        min_id = 9999999
        max_id = -1

        for media_id in sorted(self.dataset.media_ids):
            min_id = min(min_id, media_id)
            max_id = max(max_id, media_id)

        self.assertEqual(MEDIAFILE_OFFSET + 1, min_id)
        self.assertEqual(DREAMER_NUM_MEDIA_FILES, max_id - min_id + 1)

    def test_dataset_preload_files_exist(self):
        for trial in self.dataset.trials:
            for signal_type in trial.signal_types:
                self.assertTrue(trial.signal_data_files[signal_type].exists(),
                                f'Signal data file {trial.signal_data_files[signal_type]} does not exist')

    @staticmethod
    def bad_signal():
        return DreamerDataset(DEFAULT_DREAMER_PATH, signals=['XYZ'])

    def test_invalid_signal(self):
        """
        Asserts that a ValueError is thrown if DreamerDataset is constructed with a signal type that does not
        exist on the filesystem.
        :return:
        """
        self.assertRaises(ValueError, DreamerDatasetTest.bad_signal)

    def test_ecg_signal_load(self):
        """
        Asserts that we can properly load an ECG signal from one of the dataset's trials.
        :return:
        """
        for trial in self.dataset.trials:
            self.assertEqual(trial.load_signal_data('ECG').shape[0], 3)

    def test_splits(self):
        trial_splits = self.dataset.get_trial_splits([.7, .3])
        split_1_participants = set([trial.participant_id for trial in trial_splits[0]])
        split_2_participants = set([trial.participant_id for trial in trial_splits[1]])

        # Assert that we got two splits...
        self.assertEqual(len(trial_splits), 2)

        # Assert that the length of the splits sums to the total number of trials in the dataset.
        self.assertEqual(len(self.dataset.trials), len(trial_splits[0]) + len(trial_splits[1]))

        # Assert that no participant in the first split appears in the second split
        self.assertEqual(0, len(split_1_participants.intersection(split_2_participants)))

    def test_three_splits(self):
        trial_splits = self.dataset.get_trial_splits([.7, .15, .15])
        split_1_participants = set([trial.participant_id for trial in trial_splits[0]])
        split_2_participants = set([trial.participant_id for trial in trial_splits[1]])
        split_3_participants = set([trial.participant_id for trial in trial_splits[2]])

        # Assert that we got three splits...
        self.assertEqual(len(trial_splits), 3)

        # Assert that the length of the splits sums to the total number of trials in the dataset.
        self.assertEqual(len(trial_splits[0]) + len(trial_splits[1]) + len(trial_splits[2]),
                         len(self.dataset.trials))

        # Assert that no participant in the first split appears in the second split
        self.assertEqual(0, len(split_1_participants.intersection(split_2_participants)))

        # Assert that no participant in the first split appears in the third split
        self.assertEqual(0, len(split_1_participants.intersection(split_3_participants)))

        # Assert that no participant in the second split appears in the third split
        self.assertEqual(0, len(split_2_participants.intersection(split_3_participants)))

    def test_participant_ids_are_sequential(self):
        participant_ids = sorted(self.dataset.participant_ids)
        for i in range(len(participant_ids)):
            self.assertEqual(participant_ids[i], i + 1 + self.dataset.participant_offset)

    def test_expected_responses(self):
        media_ids = sorted(self.dataset.media_ids)
        self.assertEqual(len(media_ids), len(self.dataset.expected_media_responses))
        for trial in self.dataset.trials:
            self.assertIsNotNone(trial.expected_response)

    def test_arousal(self):
        for trial in self.dataset.trials:
            quad = trial.load_ground_truth()
            if quad == 1 or quad == 2:
                self.assertEqual(1, trial.load_ground_truth(truth=TruthType.AROUSAL))
            elif quad == 3 or quad == 4:
                self.assertEqual(0, trial.load_ground_truth(truth=TruthType.AROUSAL))
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

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

from ardt.datasets import AERTrialFilter
from ardt.datasets.AERTrial import TruthType

from ardt.datasets.cuads import CuadsDataset
from ardt.datasets.cuads.CuadsDataset import DEFAULT_DATASET_PATH, CUADS_NUM_MEDIA_FILES, \
    CUADS_NUM_PARTICIPANTS

PARTICIPANT_OFFSET = 5
MEDIAFILE_OFFSET = 5

CUADS_NUM_PARTICIPANTS = 38
CUADS_NUM_MEDIA_FILES = 20
CUADS_NUM_TRIALS = 714

class CuadsDatasetTest(unittest.TestCase):
    def setUp(self):
        self.dataset = CuadsDataset(None, PARTICIPANT_OFFSET, MEDIAFILE_OFFSET)
        self.dataset.preload()
        self.dataset.load_trials()

    def test_cuads_dataset_load(self):
        """
        Asserts that the ASCERTAIN dataset loads the expected number of participants, movie clips, and trials.
        :return:
        """
        self.assertEqual(len(self.dataset.participant_ids), CUADS_NUM_PARTICIPANTS)
        self.assertEqual(len(self.dataset.media_ids), CUADS_NUM_MEDIA_FILES)
        self.assertEqual(len(self.dataset.trials), CUADS_NUM_TRIALS)

    def test_ecg_signal_load(self):
        """
        Asserts that we can properly load an ECG signal from one of the dataset's trials.
        :return:
        """
        trial = self.dataset.trials[random.randint(0, len(self.dataset.trials) - 1)]
        self.assertEqual(trial.load_signal_data('ECG').shape[0], 4)

    def test_participant_id_offsets(self):
        min_id = min(self.dataset.participant_ids)
        max_id = max(self.dataset.participant_ids)

        self.assertEqual(PARTICIPANT_OFFSET + 1, min_id)
        self.assertEqual(PARTICIPANT_OFFSET + CUADS_NUM_PARTICIPANTS, max_id)

    def test_media_id_offsets(self):
        min_id = min(self.dataset.media_ids)
        max_id = max(self.dataset.media_ids)

        self.assertEqual(MEDIAFILE_OFFSET + 1, min_id)
        self.assertEqual(CUADS_NUM_MEDIA_FILES, max_id - min_id + 1)

    def test_splits(self):
        trial_splits = self.dataset.get_trial_splits([.7, .3])
        split_1_participants = set([x.participant_id for x in trial_splits[0]])
        split_2_participants = set([x.participant_id for x in trial_splits[1]])

        self.assertEqual(len(trial_splits), 2)
        self.assertEqual(len(trial_splits[0]) + len(trial_splits[1]), len(self.dataset.trials))
        self.assertEqual(0, len(split_1_participants.intersection(split_2_participants)))

    def test_filtered_splits(self):
        localds = CuadsDataset(None, 0, 0)
        localds.preload()

        num_media_ids = random.randint(2,5)
        num_participant_ids = random.randint(5,10)

        mediaids_to_filter_out = random.choices(np.arange(1,CUADS_NUM_MEDIA_FILES), k=num_media_ids)
        participants_to_filter_out = random.choices(np.arange(1,CUADS_NUM_PARTICIPANTS), k=num_participant_ids)

        participant_id_filter = AERTrialFilter(lambda x: x.participant_id not in participants_to_filter_out)
        media_id_filter = AERTrialFilter(lambda x: x.media_id not in mediaids_to_filter_out)
        localds.load_trials(trial_filters=[participant_id_filter, media_id_filter])

        trial_splits = localds.get_trial_splits([.7, .3])
        split_1_participants = set([x.participant_id for x in trial_splits[0]])
        split_2_participants = set([x.participant_id for x in trial_splits[1]])

        self.assertEqual(len(trial_splits), 2)
        self.assertEqual(len(trial_splits[0]) + len(trial_splits[1]), len(localds.trials))
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
                self.assertEqual(2, trial.load_ground_truth(truth=TruthType.AROUSAL))
            elif quad == 3 or quad == 4:
                self.assertEqual(1, trial.load_ground_truth(truth=TruthType.AROUSAL))

    def test_valence(self):
        for trial in self.dataset.trials:
            quad = trial.load_ground_truth()
            if quad == 1 or quad == 4:
                self.assertEqual(2, trial.load_ground_truth(truth=TruthType.VALENCE))
            elif quad == 2 or quad == 3:
                self.assertEqual(1, trial.load_ground_truth(truth=TruthType.VALENCE))
            else:
                self.fail("Unknown trial type: " + str(quad))

    def test_trial_filters(self):
        localds = CuadsDataset(None, 0, 0)
        localds.preload()

        num_media_ids = random.randint(2,5)
        num_participant_ids = random.randint(5,10)

        mediaids_to_filter_out = random.choices(np.arange(1,CUADS_NUM_MEDIA_FILES), k=num_media_ids)
        participants_to_filter_out = random.choices(np.arange(1,CUADS_NUM_PARTICIPANTS), k=num_participant_ids)

        participant_id_filter = AERTrialFilter(lambda x: x.participant_id not in participants_to_filter_out)
        media_id_filter = AERTrialFilter(lambda x: x.media_id not in mediaids_to_filter_out)
        localds.load_trials(trial_filters=[participant_id_filter, media_id_filter])

        self.assertEqual(1, min(localds.media_ids))
        self.assertEqual(1, min(localds.participant_ids))
        self.assertEqual(len(localds.media_ids), max(localds.media_ids))
        self.assertEqual(len(localds.participant_ids), max(localds.participant_ids))

        pass

if __name__ == '__main__':
    unittest.main()

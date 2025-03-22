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

from ardt.datasets.ascertain.AscertainDataset import DEFAULT_ASCERTAIN_PATH, ASCERTAIN_NUM_MEDIA_FILES, \
    ASCERTAIN_NUM_PARTICIPANTS, ASCERTAIN_RAW_FOLDER
from ardt.datasets.cuads import CuadsDataset
from ardt.datasets.cuads.CuadsDataset import CUADS_NUM_TRIALS, CUADS_NUM_PARTICIPANTS, CUADS_NUM_MEDIA_FILES
PARTICIPANT_OFFSET = 50
MEDIAFILE_OFFSET = 20


class DatasetSplitTest(unittest.TestCase):
    def setUp(self):
        self.cuads = CuadsDataset(participant_offset=PARTICIPANT_OFFSET,
                                  mediafile_offset=MEDIAFILE_OFFSET)
        self.cuads.preload()
        self.cuads.load_trials()

        self.datasets = self.cuads.get_dataset_splits([.7, .3])
        self.dataset = self.datasets[0]

    def test_split_counts(self):
        """
        Asserts that the ASCERTAIN dataset loads the expected number of participants, movie clips, and trials.
        :return:
        """
        self.assertEqual(len(self.dataset.participant_ids), int(CUADS_NUM_PARTICIPANTS*.7)+1)
        self.assertEqual(len(self.dataset.media_ids), CUADS_NUM_MEDIA_FILES)

    def test_expected_responses(self):
        media_ids = sorted(self.dataset.media_ids)
        self.assertEqual(len(media_ids), len(self.dataset.expected_media_responses))
        for trial in self.dataset.trials:
            self.assertIsNotNone(trial.expected_response)

    def test_ecg_signal_load(self):
        """
        Asserts that we can properly load an ECG signal from one of the dataset's trials.
        :return:
        """
        trial = self.dataset.trials[random.randint(0, len(self.dataset.trials) - 1)]
        self.assertEqual(trial.load_signal_data('ECG').shape[0], 4)


if __name__ == '__main__':
    unittest.main()

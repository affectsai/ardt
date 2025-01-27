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

from aardt.datasets import TFDatasetWrapper
from aardt.datasets.MultiDataset import MultiDataset
from aardt.datasets.ascertain import AscertainDataset
from aardt.datasets.ascertain.AscertainDataset import DEFAULT_ASCERTAIN_PATH, ASCERTAIN_NUM_PARTICIPANTS, \
    ASCERTAIN_NUM_MEDIA_FILES
from aardt.datasets.cuads import CuadsDataset
from aardt.datasets.cuads.CuadsDataset import DEFAULT_DATASET_PATH, CUADS_NUM_MEDIA_FILES, \
    CUADS_NUM_PARTICIPANTS
from aardt.datasets.dreamer import DreamerDataset
from aardt.datasets.dreamer.DreamerDataset import DEFAULT_DREAMER_PATH, DREAMER_NUM_PARTICIPANTS, \
    DREAMER_NUM_MEDIA_FILES
from aardt.preprocessors import FixedDurationPreprocessor
from aardt.preprocessors.ChannelSelector import ChannelSelector


class CuadsDatasetTest(unittest.TestCase):
    def setUp(self):
        fixed_duration = FixedDurationPreprocessor(45, 256,0)

        cuads_processor = ChannelSelector(retain_channels=[2,3],
                                          child_preprocessor=fixed_duration)
        ascertain_processor = ChannelSelector(retain_channels=[1,2],
                                          child_preprocessor=fixed_duration)
        dreamer_processor = ChannelSelector(retain_channels=[1,2],
                                          child_preprocessor=fixed_duration)

        self.ascertain_dataset = AscertainDataset(DEFAULT_ASCERTAIN_PATH, signals=['ECG'])
        self.ascertain_dataset.signal_preprocessors['ECG'] = ascertain_processor

        self.dreamer_dataset = DreamerDataset(DEFAULT_DREAMER_PATH,
                                              signals=['ECG'],
                                              participant_offset=ASCERTAIN_NUM_PARTICIPANTS,
                                              mediafile_offset=ASCERTAIN_NUM_MEDIA_FILES)
        self.dreamer_dataset.signal_preprocessors['ECG'] = dreamer_processor

        self.cuads_dataset = CuadsDataset( participant_offset=ASCERTAIN_NUM_PARTICIPANTS+DREAMER_NUM_PARTICIPANTS,
                                           mediafile_offset=ASCERTAIN_NUM_MEDIA_FILES+DREAMER_NUM_MEDIA_FILES)

        self.cuads_dataset.signal_preprocessors['ECG'] = cuads_processor

        self.multiset = MultiDataset([self.ascertain_dataset, self.dreamer_dataset, self.cuads_dataset])
        self.multiset.set_signal_metadata('ECG', {'n_channels':2})
        self.multiset.preload()
        self.multiset.load_trials()

    def test_multiset_trial_count(self):
        """
        Asserts that the number of trials in the multiset is the same as the sum of the number of trials in each dataset.
        :return:
        """
        self.assertEqual(len(self.ascertain_dataset.trials)+len(self.dreamer_dataset.trials)+len(self.cuads_dataset.trials), len(self.multiset.trials))
        self.assertNotEqual(0, len(self.multiset.trials))


    def test_ecg_signal_load(self):
        """
        Asserts that we can properly load an ECG signal from one of the dataset's trials.
        :return:
        """
        for trial in self.multiset.trials:
            signal = trial.load_preprocessed_signal_data('ECG')
            self.assertEqual(signal.shape[0], 2, f"{type(trial)} has shape {signal.shape}")

    def test_splits(self):
        trial_splits = self.multiset.get_trial_splits([.7, .3])
        split_1_participants = set([x.participant_id for x in trial_splits[0]])
        split_2_participants = set([x.participant_id for x in trial_splits[1]])

        self.assertEqual(len(trial_splits), 2)
        self.assertEqual(len(trial_splits[0]) + len(trial_splits[1]), len(self.multiset.trials))
        self.assertEqual(0, len(split_1_participants.intersection(split_2_participants)))

    def test_three_splits(self):
        trial_splits = self.multiset.get_trial_splits([.7, .15, .15])
        split_1_participants = set([x.participant_id for x in trial_splits[0]])
        split_2_participants = set([x.participant_id for x in trial_splits[1]])
        split_3_participants = set([x.participant_id for x in trial_splits[2]])

        self.assertEqual(len(trial_splits), 3)
        self.assertEqual(len(trial_splits[0]) + len(trial_splits[1]) + len(trial_splits[2]),
                         len(self.multiset.trials))
        self.assertEqual(0, len(split_1_participants.intersection(split_2_participants)))
        self.assertEqual(0, len(split_1_participants.intersection(split_3_participants)))
        self.assertEqual(0, len(split_2_participants.intersection(split_3_participants)))

    def test_tfdatasetwrapper(self):
        """
        Tests that the tf.data.dataset provided by the TFDataSetWrapper provides all the samples given in the dataset,
        the expected number of times.
        """
        repeat_count = random.randint(1, 3)
        tfdsw = TFDatasetWrapper(dataset=self.multiset)
        tfds = tfdsw(signal_type='ECG', batch_size=64, buffer_size=500, repeat=repeat_count)

        iteration = 0
        total_elems = 0

        # loop over the provided number of steps
        for batch in tfds:
            iteration += 1
            total_elems += len(batch[0])

        # stop the timer
        # return the difference between end and start times
        self.assertGreater(iteration, 0)
        self.assertEqual(len(self.multiset.trials) * repeat_count, total_elems)

if __name__ == '__main__':
    unittest.main()

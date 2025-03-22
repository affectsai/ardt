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

from ardt.datasets import TFDatasetWrapper
from ardt.datasets.MultiDataset import MultiDataset
from ardt.datasets.ascertain import AscertainDataset
from ardt.datasets.cuads import CuadsDataset
from ardt.datasets.dreamer import DreamerDataset
from ardt.preprocessors import FixedDurationPreprocessor
from ardt.preprocessors.ChannelSelector import ChannelSelector


class MultiDatasetTest(unittest.TestCase):
    def setUp(self):
        fixed_duration = FixedDurationPreprocessor(45, 256,0)

        cuads_processor = ChannelSelector(retain_channels=[2,3],
                                          child_preprocessor=fixed_duration)
        ascertain_processor = ChannelSelector(retain_channels=[1,2],
                                          child_preprocessor=fixed_duration)
        dreamer_processor = ChannelSelector(retain_channels=[1,2],
                                          child_preprocessor=fixed_duration)

        self.ascertain_dataset = AscertainDataset(signals=['ECG'])
        self.ascertain_dataset.signal_preprocessors['ECG'] = ascertain_processor

        self.dreamer_dataset = DreamerDataset(signals=['ECG'])
        self.dreamer_dataset.signal_preprocessors['ECG'] = dreamer_processor

        self.cuads_dataset = CuadsDataset( )

        self.cuads_dataset.signal_preprocessors['ECG'] = cuads_processor

        self.dataset = MultiDataset([self.ascertain_dataset, self.dreamer_dataset, self.cuads_dataset])
        self.dataset.set_signal_metadata('ECG', {'n_channels':2})
        self.dataset.preload()
        self.dataset.load_trials()

    def test_multiset_random_order(self):
        datasets = [self.ascertain_dataset, self.dreamer_dataset, self.cuads_dataset]
        for i in range(len(datasets)):
            local_dataset = MultiDataset(datasets)
            local_dataset.set_signal_metadata('ECG', {'n_channels':2})
            local_dataset.preload()
            local_dataset.load_trials()
            datasets.append(datasets.pop(0))

            for trial in local_dataset.trials:
                self.assertIn(trial.participant_id, trial.dataset.participant_ids)
                self.assertIn(trial.media_id, trial.dataset.media_ids)


    def test_multiset_trial_count(self):
        """
        Asserts that the number of trials in the multiset is the same as the sum of the number of trials in each dataset.
        :return:
        """
        self.assertEqual(len(self.ascertain_dataset.trials)+len(self.dreamer_dataset.trials)+len(self.cuads_dataset.trials), len(self.dataset.trials))
        self.assertNotEqual(0, len(self.dataset.trials))

    def test_multiset_participant_count(self):
        """
        Asserts that the number of trials in the multiset is the same as the sum of the number of trials in each dataset.
        :return:
        """
        self.assertNotEqual(0, len(self.dataset.participant_ids))
        self.assertEqual(len(self.ascertain_dataset.participant_ids)+len(self.dreamer_dataset.participant_ids)+len(self.cuads_dataset.participant_ids), len(self.dataset.participant_ids))

    def test_multiset_media_count(self):
        """
        Asserts that the number of trials in the multiset is the same as the sum of the number of trials in each dataset.
        :return:
        """
        self.assertNotEqual(0, len(self.dataset.media_ids))
        self.assertEqual(len(self.ascertain_dataset.media_ids)+len(self.dreamer_dataset.media_ids)+len(self.cuads_dataset.media_ids), len(self.dataset.media_ids))

    def test_ecg_signal_load(self):
        """
        Asserts that we can properly load an ECG signal from one of the dataset's trials.
        :return:
        """
        for trial in random.sample(self.dataset.trials, int(len(self.dataset.trials)*.1)):
            signal = trial.load_preprocessed_signal_data('ECG')
            self.assertEqual(signal.shape[0], 2, f"{type(trial)} has shape {signal.shape}")

    def test_splits(self):
        trial_splits = self.dataset.get_trial_splits([.7, .3])
        split_1_participants = set([x.participant_id for x in trial_splits[0]])
        split_2_participants = set([x.participant_id for x in trial_splits[1]])

        self.assertNotEqual(0, len(split_1_participants))
        self.assertNotEqual(0, len(split_2_participants))
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
        self.assertNotEqual(0, len(split_1_participants))
        self.assertNotEqual(0, len(split_2_participants))
        self.assertNotEqual(0, len(split_3_participants))
        self.assertEqual(0, len(split_1_participants.intersection(split_2_participants)))
        self.assertEqual(0, len(split_1_participants.intersection(split_3_participants)))
        self.assertEqual(0, len(split_2_participants.intersection(split_3_participants)))

    # def test_tfdatasetwrapper(self):
    #     """
    #     Tests that the tf.data.dataset provided by the TFDataSetWrapper provides all the samples given in the dataset,
    #     the expected number of times.
    #     """
    #     repeat_count = random.randint(1, 3)
    #     tfdsw = TFDatasetWrapper(dataset=self.dataset)
    #     tfds = tfdsw(signal_type='ECG', batch_size=2, buffer_size=4, repeat=repeat_count)
    #
    #     iteration = 0
    #     total_elems = 0
    #
    #     # loop over the provided number of steps
    #     for batch in tfds:
    #         iteration += 1
    #         total_elems += len(batch[0])
    #
    #     # stop the timer
    #     # return the difference between end and start times
    #     self.assertGreater(iteration, 0)
    #     self.assertEqual(len(self.dataset.trials) * repeat_count, total_elems)

    def test_participant_ids_are_sequential(self):
        participant_ids = sorted(self.dataset.participant_ids)
        for i in range(len(participant_ids)):
            self.assertEqual(participant_ids[i], i + 1 + self.dataset.participant_offset)

    def test_expected_responses(self):
        for trial in self.dataset.trials:
            self.assertIsNotNone(trial.expected_response)

if __name__ == '__main__':
    unittest.main()

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

from ardt.datasets import TFDatasetWrapper
from ardt.datasets.ascertain import AscertainDataset
from ardt.datasets.ascertain.AscertainDataset import DEFAULT_ASCERTAIN_PATH, ASCERTAIN_NUM_MEDIA_FILES, \
    ASCERTAIN_NUM_PARTICIPANTS
from ardt.datasets.dreamer import DreamerDataset
from ardt.datasets.dreamer.DreamerDataset import DEFAULT_DREAMER_PATH, DREAMER_NUM_PARTICIPANTS, \
    DREAMER_NUM_MEDIA_FILES
from ardt.datasets.cuads import CuadsDataset
from ardt.datasets.cuads.CuadsDataset import CUADS_NUM_TRIALS

from ardt.preprocessors import FixedDurationPreprocessor


class TFDataSetWrapperTest(unittest.TestCase):
    def setUp(self):
        self.preprocess_pipeline = FixedDurationPreprocessor(45, 256, 0)

        self.ascertain_dataset = AscertainDataset(DEFAULT_ASCERTAIN_PATH, signals=['ECG'])
        self.ascertain_dataset.signal_preprocessors['ECG'] = self.preprocess_pipeline
        self.ascertain_dataset.preload()
        self.ascertain_dataset.load_trials()

        self.dreamer_dataset = DreamerDataset(DEFAULT_DREAMER_PATH, signals=['ECG'])
        self.dreamer_dataset.signal_preprocessors['ECG'] = self.preprocess_pipeline
        self.dreamer_dataset.preload()
        self.dreamer_dataset.load_trials()

        self.cuads_dataset = CuadsDataset()
        self.cuads_dataset.signal_preprocessors['ECG'] = self.preprocess_pipeline
        self.cuads_dataset.preload()
        self.cuads_dataset.load_trials()

    def test_ascertain_dataset(self):
        """
        Tests that the tf.data.dataset provided by the TFDataSetWrapper provides all the samples given in the dataset,
        the expected number of times.
        """
        repeat_count = random.randint(1, 10)
        tfdsw = TFDatasetWrapper(dataset=self.ascertain_dataset)
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
        self.assertEqual(ASCERTAIN_NUM_PARTICIPANTS * ASCERTAIN_NUM_MEDIA_FILES * repeat_count, total_elems)

    def test_dreamer_dataset(self):
        """
        Tests that the tf.data.dataset provided by the TFDataSetWrapper provides all the samples given in the dataset,
        the expected number of times.
        """
        repeat_count = random.randint(1, 10)
        tfdsw = TFDatasetWrapper(dataset=self.dreamer_dataset)
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
        self.assertEqual(DREAMER_NUM_PARTICIPANTS * DREAMER_NUM_MEDIA_FILES * repeat_count, total_elems)

    def test_cuads_dataset(self):
        """
        Tests that the tf.data.dataset provided by the TFDataSetWrapper provides all the samples given in the dataset,
        the expected number of times.
        """
        repeat_count = random.randint(2, 5)
        tfdsw = TFDatasetWrapper(dataset=self.cuads_dataset)
        tfds = tfdsw(signal_type='ECG', batch_size=64, buffer_size=100, repeat=repeat_count)

        iteration = 0
        total_elems = 0

        # loop over the provided number of steps
        for batch in tfds:
            iteration += 1
            total_elems += len(batch[0])

        # stop the timer
        # return the difference between end and start times
        self.assertGreater(iteration, 0)
        self.assertEqual(CUADS_NUM_TRIALS * repeat_count, total_elems)



if __name__ == '__main__':
    unittest.main()

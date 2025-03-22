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
import random
import unittest

import torch

from ardt.datasets.ascertain import AscertainDataset
from ardt.datasets.ascertain.AscertainDataset import DEFAULT_ASCERTAIN_PATH
from ardt.datasets.cuads import CuadsDataset
from ardt.datasets.dreamer import DreamerDataset
from ardt.datasets.dreamer.DreamerDataset import DEFAULT_DREAMER_PATH
from ardt.datasets.ml import TorchDatasetWrapper
from ardt.preprocessors import FixedDurationPreprocessor


class TorchDatasetTest(unittest.TestCase):
    def setUp(self):
        self.preprocess_pipeline = FixedDurationPreprocessor(45, 256, 0)

        self.ascertain_dataset = AscertainDataset(DEFAULT_ASCERTAIN_PATH, signals=['ECG'])
        self.ascertain_dataset.signal_preprocessors['ECG'] = self.preprocess_pipeline
        self.ascertain_dataset.preload()
        self.ascertain_dataset.load_trials()


    def test_torch_dataset(self):
        """
        Tests that the tf.data.dataset provided by the TFDataSetWrapper provides all the samples given in the dataset,
        the expected number of times.
        """
        repeat_count = random.randint(1, 10)
        dataset = TorchDatasetWrapper(dataset=self.ascertain_dataset, signal_type='ECG')

        sample_count = 0
        for trial in self.ascertain_dataset.trials:
            sample_count += trial.get_signal_metadata('ECG')['n_channels']

        dataset_sample_count = 0
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
        for batch in dataloader:
            signals, labels = batch
            dataset_sample_count += signals.shape[0]

        self.assertEqual(sample_count, dataset_sample_count)

    def test_torch_dataset_repeat(self):
        """
        Tests that the tf.data.dataset provided by the TFDataSetWrapper provides all the samples given in the dataset,
        the expected number of times.
        """
        repeat_count = random.randint(1, 10)
        dataset = TorchDatasetWrapper(dataset=self.ascertain_dataset, signal_type='ECG')

        sample_count = 0
        for trial in self.ascertain_dataset.trials:
            sample_count += trial.get_signal_metadata('ECG')['n_channels']

        dataset_sample_count = 0
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
        for batch in dataloader:
            signals, labels = batch
            dataset_sample_count += signals.shape[0]

        self.assertEqual(sample_count, dataset_sample_count)
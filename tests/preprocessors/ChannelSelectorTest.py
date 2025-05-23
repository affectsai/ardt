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

import unittest

import numpy as np

from ardt.preprocessors.ChannelSelector import ChannelSelector

# Test parameters
SIGNAL_DURATION = 10
SAMPLE_RATE = 256


class ChannelSelectorTest(unittest.TestCase):
    def test_channel_selector_default_removal(self):
        """
        Tests that when called with defaults, that the ChannelSelector removes the 0th row from the input data, which
        usually corresponds to the timestamp data.
        """
        signal = np.random.random(size=(5, SAMPLE_RATE * SIGNAL_DURATION))
        preprocessor = ChannelSelector()
        processed = preprocessor(signal)

        # Assert that the processed signal has the expected number of samples
        self.assertEqual(signal.shape[0] - 1, processed.shape[0])

        # Assert that the data at row N+1 of the signal is the same as the data in row N of the processed output.
        for row in range(processed.shape[0]):
            self.assertFalse((signal[row + 1, :] - processed[row, :]).all())


if __name__ == '__main__':
    unittest.main()

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

from ardt.preprocessors.transformers.MinMaxScaler import MinMaxScaler


class MinMaxScalarTest(unittest.TestCase):
    def test_minmax_scalar(self):
        for min_val in range(10):
            for max_val in range(50, 60):
                signal = np.random.random(size=(3, 2560)) * 100
                scaler = MinMaxScaler(feature_range=(min_val, max_val))
                processed = scaler(signal)

                self.assertLessEqual(np.max(processed) - 1e-8, max_val)
                self.assertGreaterEqual(np.min(processed) + 1e-8, min_val)


if __name__ == '__main__':
    unittest.main()

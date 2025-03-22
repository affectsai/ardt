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
from sklearn import preprocessing as p

from ardt.preprocessors import SignalPreprocessor


class MinMaxScaler(SignalPreprocessor):
    """
    Applies a sklearn.preprocessing.MinMaxScaler to the signal data.
    """

    def __init__(self, feature_range=(0, 1), parent_preprocessor=None, child_preprocessor=None):
        """

        :param feature_range: the desired feature range for the sklearn.preprocessing.MinMaxScaler
        :param parent_preprocessor:
        """
        super().__init__(parent_preprocessor, child_preprocessor)
        self._feature_range = feature_range

    def process_signal(self, signal):
        min_max_scaler = p.MinMaxScaler(feature_range=self._feature_range)
        return min_max_scaler.fit_transform(signal)

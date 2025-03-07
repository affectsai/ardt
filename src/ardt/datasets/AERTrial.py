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


import abc
import numpy as np

from ardt.datasets import AERDataset
from enum import Enum, auto

class TruthType(Enum):
    QUADRANT = auto()
    AROUSAL = auto()
    VALENCE = auto()
    NEUTRAL = auto()

class AERTrial(abc.ABC):
    def __init__(self, dataset: AERDataset, participant_id: int, movie_id: int):
        """
        This class provides functionality to manage a dataset, associated participant
        information, and movie data. It initializes various container attributes to
        manage signal types, preprocessors, and data files correspondingly. These
        attributes will facilitate efficient organization and retrieval of
        participant-specific and movie-specific data for processing or analysis.

        :param dataset: The dataset object that contains the data to be handled.
        :param participant_id: The unique identifier for a participant within the dataset.
        :param movie_id: The unique identifier for a movie associated with the data.
        :type dataset: AERDataset
        :type participant_id: int
        :type movie_id: int
        """
        self._dataset = dataset
        self._participant_id = participant_id
        self._signal_types = set()
        self._signal_preprocessors = {}
        self._signal_data_files = {}
        self._media_id = movie_id

    @staticmethod
    def quadrant_to_arousal(q):
        return 2 if q == 1 or q == 2 else 1

    @staticmethod
    def quadrant_to_valence(q):
        return 2 if q == 1 or q == 4 else 1

    def load_preprocessed_signal_data(self,signal_type: str):
        '''
        deprecated ... don't use.
        :param signal_type:
        :return:
        '''
        return self.load_signal_data(signal_type)

    def load_signal_data(self, signal_type: str):
        signal_data = self.load_raw_signal_data(signal_type)
        if signal_type in self.signal_preprocessors.keys():
            signal_data = self.signal_preprocessors[signal_type](signal_data)
        else:
            pass
            #print("NO SIGNAL PREPROCESSOR FOUND")
        return signal_data

    @abc.abstractmethod
    def load_raw_signal_data(self, signal_type: str):
        """
        Loads and returns the requested signal as an (N+1)xM numpy array, where N is the number of channels, and M is
        the number of samples in the signal. The row at N=0 represents the timestamp of each sample. The value is
        given in epoch time if a real start time is available, otherwise it is in elapsed milliseconds with 0
        representing the start of the sample.

        :param signal_type:
        :return:
        """
        if signal_type not in self._signal_types:
            raise ValueError('Signal type {} is not known in this AERTrial'.format(signal_type))

        return np.empty(0)

    @abc.abstractmethod
    def load_ground_truth(self, truth=TruthType.QUADRANT):
        """
        Returns the ground truth label for this trial. For AER trials, this is the quadrant within the A/V space,
        numbered 0 through 3 as follows:
        - 0: High Arousal, High Valence
        - 1: High Arousal, Low Valence
        - 2: Low Arousal, Low Valence
        - 3: Low Arousal, High Valence

        :return: The ground truth label for this trial
        """
        pass

    @abc.abstractmethod
    def get_signal_metadata(self, signal_type):
        """
        Returns a dict containing the requested signal's metadata. Mandatory keys include:
        - 'signal_type' (the signal type)
        - 'sample_rate' (in samples per second)
        - 'n_channels' (the number of channels in the signal)

        See subclasses for implementation-specific keys that may also be present.

        :param signal_type: the type of signal for which to retrieve the metadata.
        :return: a dict containing the requested signal's metadata
        """
        if signal_type not in self._signal_types:
            raise ValueError('Signal type {} is not known in this AERTrial'.format(signal_type))

        return {}

    @property
    def dataset(self):
        return self._dataset

    @property
    def signal_data_files(self):
        return self._signal_data_files

    @signal_data_files.setter
    def signal_data_files(self, signal_data_files):
        for signal_type in signal_data_files.keys():
            self._signal_types.add(signal_type)

        self._signal_data_files = signal_data_files

    @property
    def signal_types(self):
        return self._signal_types

    @signal_types.setter
    def signal_types(self, signal_types):
        self._signal_types = signal_types

    @property
    def signal_preprocessors(self):
        return self._signal_preprocessors

    @signal_preprocessors.setter
    def signal_preprocessors(self, signal_preprocessors):
        self._signal_preprocessors = signal_preprocessors

    @property
    def native_media_id(self):
        """
        The media_id used in the underlying dataset. This is not adjusted by media_offset if one is used.

        :return:
        """
        return self._media_id

    @property
    def media_id(self):
        """
        The unique identifier for the media file, adjusted by the media file offset specified in the associated dataset.

        :property:

        :return: Returns the adjusted media file ID as an integer.
        :rtype: int
        """
        return self._media_id + self.dataset.media_offset

    @property
    def media_name(self):
        '''
        Some datasets may use media names e.g. "funny_video.mp4" instead of media ID numbers, and generate the media
        ID numbers during load_dataset. In this situations it may be useful to be able to access the original media name.

        This method will do that. If the dataset does not provide a media_name map in media_names_by_movie_id, then
        this method will just return self.media_id
        :return:
        '''
        name = self.dataset.get_media_name_by_movie_id(self.media_id-self.dataset.media_offset)

        if name is not None:
            return name

        return self.media_id - self.dataset.media_offset

    @property
    def native_participant_id(self):
        """
        The native participant ID used in the underlying dataset, this is not offset by the participant_offset if one is used.

        :return:
        """
        return self._participant_id

    @property
    def participant_id(self):
        """
        The unique identifier for the participant, adjusted by the participant offset specified in the associated dataset.

        :property:

        :return: Returns the adjusted participant ID as an integer.
        :rtype: int
        """
        return self._participant_id + self.dataset.participant_offset

    @property
    @abc.abstractmethod
    def participant_response(self):
        pass

    @property
    def expected_response(self):
        expected_responses = self.dataset.expected_media_responses
        return expected_responses[self.media_name]


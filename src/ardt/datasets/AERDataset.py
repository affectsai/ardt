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
#  is distributed on an "AS IS" BASIS, WITHOUT WARcANTIES OR CONDITIONS OF ANY KIND, either
#  express or implied. See the License for the specific language governing permissions and limitations
#  under the License.


import abc
from pathlib import Path
from typing import List, Optional

import numpy as np
import os

from ardt import config, ardt_deprecated
from .AERTrialFilter import AERTrialFilter

from pandas.io.sas.sas_constants import os_maker_length
import random
from itertools import zip_longest, cycle
import warnings



class AERDataset(metaclass=abc.ABCMeta):
    """
    AERDataset is the base class for all dataset implementations in AARDT. All AERDatasets expose the following
    properties:
    - trials: a list of all AERTrials associated with this dataset
    - signals: a list of signals loaded in this dataset instance (a proper subset of the available signals within this
    dataset)
    - participant_ids: a list of participant identifiers in the dataset. Participant IDs are offset by the value of
     participant_offset
    - media_ids: a list of identifiers for the media file used as emotional stimulus in the dataset. Media IDs are
    offset by the value of media_file_offset
    - participant_offset: a constant value that is added to all participants identifiers in the dataset. This is useful
    when you will be using trials from several different AERDatasets.
    - media_file_offset: a constant value that is added to all media file identifiers in the dataset. This is useful
    when you will be using trials from several different AERDatasets
    - signal_preprocessors: a mapping of signal_type to SignalPreprocessor chain, used to automate signal preprocessing
    when signals are loaded from the AERTrial instances under this dataset.

    AERDataset also encapsulates business logic that is reusable by its subclasses, including:
    - preload signal checking: calls to `preload` are first checked to see if all signals have already been preloaded by
     a previous invocation. If a preload is necessary, then the subclass' _preload_dataset method will be called,
     otherwise no action is taken.
    - get_trial_splits: used to generate training, validation and test splits based on participant identifiers.

    The general usage pattern looks something like this:
    >>> from ardt.datasets.ascertain import AscertainDataset
    >>> my_dataset = AscertainDataset(signals=['ECG'])
    >>> my_dataset.signal_preprocessors['ECG'] = aardt.preprocessors.NK2ECGPreprocessor()
    >>> my_dataset.preload()
    >>> my_dataset.load_trials()
    >>>
    >>> training_trials, validation_trials, test_trials =
    >>>   my_dataset.get_trial_splits([.5, .3, .2])
    >>>
    >>> for training_trial in training_trials:
    >>>     preprocessed_ecg = training_trial.load_signal_data('ECG')
    >>>     # do something with the preprocessed ecg signal.
    """
    def __init__(self, signals=None, participant_offset=0, mediafile_offset=None, media_offset=None, signal_metadata=None, expected_responses=None):
        """
        Represents a class that manages multiple signals and related data, such
        as participant and media file offsets. This class is initialized
        with optional signal data and offsets and provides internal storage
        for signal processors, participant identifiers, media identifiers,
        and trial information.

        :param signals: A list of signal types used within the instance, e.g.: ['ECG', 'EEG'].
        :param participant_offset: Offset applied to identifiers for participants.
        :param mediafile_offset: Offset applied to identifiers for media files.

        :ivar _signals: Internal storage for the list of signals.
        :ivar _signal_preprocessors: Dictionary for mapping signal processors.
        :ivar _participant_offset: Offset for participant identifiers.
        :ivar _media_offset: Offset for media file identifiers.
        :ivar _participant_ids: Set that tracks unique participant IDs.
        :ivar _media_ids: Set that tracks unique media file IDs.
        :ivar _all_trials: List that contains information about all trials.
        """
        if signals is None:
            signals = []
        if signal_metadata is None:
            signal_metadata = {}
        if expected_responses is None:
            expected_responses = {}

        if media_offset is None and mediafile_offset is not None:
            media_offset = mediafile_offset
            warnings.warn("AERDataset initialization parameter mediafile_offset is deprecated, use media_offset instead. mediafile_offset may be removed in a future release.")

        if media_offset is None:
            media_offset = 0

        self._signals = signals
        self._signal_preprocessors = {}
        self._participant_offset = participant_offset
        self._media_offset = media_offset
        self._participant_ids = set()
        self._media_ids = set()
        self._all_trials = []
        self._signal_metadata = signal_metadata
        self._expected_responses = expected_responses

    def preload(self):
        """
        Checks to see if a preload is necessary, and calls the subclass' _preload_dataset method as needed. AERDataset
        pre-loading is used to perform data transformations to optimize loading and processing when iterating over
        trials in the dataset. This is subclass-specific, and the details of how the preload works are encapsulated in
        the abstract _preload_dataset method.

        The status of the preload is saved in this dataset's working directory, specified by `self.get_working_Dir()`,
        in a file named `.preload.npy`. The file contains the list of all signals that have been preloaded for this
        AERDataset already.

        If this file does not exist, or if this AERDataset instance includes a signal type that is not already listed
        in the preload status file, then `self._preload_dataset()` is called. When this method returns, the preload
        status file is created or updated to include the new set of preloaded signal types.

        If this file exists and all signal types in this AERDataset are also listed in the preload status file, then no
        action is taken.

        :return:
        """
        preload_file = self.get_working_dir() / Path('.preload.npy')
        if preload_file.exists():
            preloaded_signals = set(np.load(preload_file))

            # If self.signals is a subset of the signals that have already been preloaded
            # then we don't have to preload anything.
            if set(self.signals).issubset(preloaded_signals):
                return

        self._preload_dataset()
        np.save(preload_file, self.signals)

    @abc.abstractmethod
    def _preload_dataset(self):
        """
        Abstract method invoked by self.preload() to perform the implementation-specific optimizations. See subclasses
        for more information about each AERDataset type's preload.

        Some datasets may need extensive processing to make them more efficient to work with. You can use this method
        to do that. For example, the DREAMER dataset is provided as a single, very large JSON data file. It would be
        very inefficient to have to hold that in memory, and query through it for every signal in each trial. Instead,
        DreamerDataset parses the JSON into a structured set of numpy files which it uses in load_trials instead.

        Store your intermediates in the dataset's working folder defined by self.get_working_dir().

        :return:
        """
        pass

    @abc.abstractmethod
    def _load_trials(self, trial_filters: Optional[List[AERTrialFilter]] = None):
        """
        Loads the AERTrials from the preloaded dataset into memory. This method should load all relevant trials from
        the dataset. To avoid memory utilization issues, it is strongly recommended to defer loading signal data into
        the AERTrial until that AERTrial's load_signal_data method is called.

        See subclasses for dataset-specific details.
        :return:
        """
        pass

    @abc.abstractmethod
    def _post_load_trials(self):
        pass

    def load_trials(self, trial_filters: Optional[List[AERTrialFilter]] = None):
        """
        This method calls self._load_trials to load all trials in this dataset. After all trials are loaded, the
        trials will be filtered according to the trial_filters list. If a trial does not pass all filters, it will
        be removed from the AERDataset.

        Once the filters are complete, the participant_ids and media_ids will be normalized so that they are numbered
        sequentially from 1 to N, where N is the number of participants or media files remaining in the dataset after
        it was filtered.

        The participant_ids and media_ids sets will be inferred from the trials loaded by this method.

        :param trial_filters:
        :return:
        """
        self._load_trials()

        # If we have no filters, we're done.
        if trial_filters is None or len(trial_filters) == 0:
            return

        all_loaded_trials = list(self.trials)

        # Filter the trials
        self.trials.clear()
        self.trials.extend(
            [trial for trial in all_loaded_trials if all(trial_filter.filter(trial) for trial_filter in trial_filters)]
        )

        # Some datasets may not have sequentially numbered participants or media files.. Even if they do, some may start
        # from 0 where others may start from 1. Additionally, the filters may have removed entire participants of
        # media ids from the dataset.
        #
        # Normalize the remaining participant_ids and media_ids so they are numbered sequentially starting from 1
        # normalized_participant_ids = {}
        # normalized_media_ids = {}
        # for trial in self.trials:
        #     if trial.participant_id not in normalized_participant_ids:
        #         normalized_participant_ids[trial.participant_id] = len(normalized_participant_ids) + 1
        # 
        #     if trial.media_id not in normalized_media_ids:
        #         normalized_media_ids[trial.media_id] = len(normalized_media_ids) + 1
        # 
        #     trial._participant_id = normalized_participant_ids[trial.participant_id]
        #     trial._media_id = normalized_media_ids[trial.media_id]

    @abc.abstractmethod
    def get_media_name_by_movie_id(self, movie_id):
        pass

    @property
    @abc.abstractmethod
    def expected_media_responses(self):
        pass

    def get_working_dir(self):
        """
        Returns the working path for this AERDataset instance, given by:
           ardt.config['working_dir'] / self.__class__.__name__ /

        For example, consider an AERDataset subclass named MyTestDataset:
            class MyTestDataset(AERDataset):
               pass

        The working directory is a subfolder of ardt.config['working_dir'] named "MyTestDataset/"

        This AERDataset working directory is where the preload status file is saved, and is also where any output
        generated by the _preload_dataset method should be stored.

        :return:
        """
        path = Path(config['working_dir']) / Path(self.__class__.__name__)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_working_path(self, trial_participant_id=None, trial_media_id=None, signal_type=None, stimuli=True, dataset_participant_id=None, dataset_media_id=None, dataset_media_name=None):
        if trial_media_id is not None and (trial_participant_id is None and dataset_participant_id is None):
            raise ValueError('Either trial_participant_id or dataset_participant_id must be given if media_id is specified.')

        if signal_type is not None and (trial_media_id is None and dataset_media_name is None and dataset_media_id is None):
            raise ValueError('One of trial_media_id, dataset_media_name or dataset_media_id must be given if signal_type is specified.')

        if signal_type is not None and signal_type not in self.signals:
            raise ValueError('Invalid signal type: {}'.format(signal_type))

        participant_id = None
        if trial_participant_id is not None:
            participant_id = f"{(trial_participant_id - self.participant_offset):02d}"
        elif dataset_participant_id is not None:
            participant_id = f"{dataset_participant_id:02d}"

        media_id = None
        if trial_media_id is not None:
            media_id = f"{(trial_media_id - self.media_file_offset):02}"
        elif dataset_media_id is not None:
            media_id = f"{dataset_media_id:02}"
        elif dataset_media_name is not None:
            media_id = dataset_media_name


        result = self.get_working_dir()
        if participant_id is not None:
            result /= f'Participant_{participant_id}'

        if media_id is not None:
            result /= f'Media_{media_id}'

        if not os.path.exists(result):
            os.makedirs(result)

        if signal_type is not None:
            result /= f'{signal_type}_{"stimuli" if stimuli else "baseline"}.npy'

        return result

    @property
    def signals(self):
        """
        Returns the set of signal types that are loaded by this AERDataset instance. This is a proper subset of the
        signal types available within this AERDataset. For example, DREAMER includes both 'EEG' and 'ECG' signal data,
        but this instance may only use 'ECG', 'EEG', or both.
        :return:
        """
        return self._signals

    @property
    def trials(self):
        """
        Returns a collection of all AERTrial instances loaded by this AERDataset. Order is not defined nor guaranteed.

        :return:
        """
        return self._all_trials

    def get_trial_splits(self, splits=None):
        """
        Returns the trials associated with this dataset, grouped into len(splits) splits. Splits are generated by
        participant-id. `splits` must be a list of relative sizes of each split, and np.sum(splits) must be 1.0. If
        `splits` is None, then [1.0] is assumed returning all trials.

        If splits=[0.7, 0.3] then the return value is a list with two elements, where the first element is a list
        containing trials from 70% of the participants in this dataset, and the second is a list containing trials from
        the remaining 30%. You may specify as many splits as needed, so for example, use `splits=[.70,.15,.15] to
        generate 70% training, 15% validation and 15% test splits.

        :param splits:
        :return: a list of trials if splits=None or [1], otherwise a list of N lists of trials, where N is the number
        of splits requested, and each list contains trials from the percent of participants specified by the split
        """
        if splits is None:
            splits = [1]

        if abs(1.0 - np.sum(splits)) > 1e-4:
            raise ValueError("Splits must sum to be 1.0")

        # If we only have 1 split then just return the list of all_trials, not a list of lists.
        if len(splits) == 1:
            return self._all_trials

        # Convert the percentages into participant counts
        splits = (np.array(splits) * len(self.participant_ids)).astype(dtype=np.int32)
        if sum(splits) != len(self.participant_ids):
            splits[0] += len(self.participant_ids) - sum(splits)

        # Split the participant ids randomly into len(splits) groups
        all_ids = set(self.participant_ids)
        participant_splits = []
        for i in range(len(splits)):
            participant_splits.append(
                list(np.random.choice(list(all_ids), splits[i], False))
            )
            all_ids = all_ids - set([x for xs in participant_splits for x in xs])

        # Obtain the groups of trials corresponding to each group of participant ids
        trial_splits = []
        for participant_split in participant_splits:
            trial_splits.append([trial for trial in self.trials if trial.participant_id in participant_split])

        return trial_splits

    def get_dataset_splits(self, splits=None):
        split_trials = self.get_trial_splits(splits)
        return [TrialWrapperDataset(t,
                                    self.participant_offset,
                                    self.media_file_offset,
                                    self._signal_metadata,
                                    self._expected_responses) for t in split_trials]

    @property
    def media_ids(self):
        """
        Returns the collection of all media identifiers associated with this AERDataset instance. The values returned
        have already been offset by self.media_file_offset. So for example, a media identifier from this AERDataset
        instance:
          N = self.media_ids[0]

        corresponds to the media id (N - self.media_file_offset) in the underlying dataset.

        :return:
        """
        return set([trial.media_id for trial in self.trials])

    @property
    def native_media_ids(self):
        return [trial.media_id - self.media_offset for trial in self.trials]

    @property
    def participant_ids(self):
        """
        Returns the collection of all participant identifiers associated with this AERDataset instance. The values
        returned have already been offset by self.participant_offset. So for example, a media identifier from this
        AERDataset instance:
          N = self.participant_ids[0]

        corresponds to the participant id (N - self.participant_offset) in the underlying dataset.

        :return:
        """
        return set([trial.participant_id for trial in self.trials])

    @property
    def native_participant_ids(self):
        return [trial.participant_id - self.participant_offset for trial in self.trials]

    @property
    def expected_media_responses(self):
        return self._expected_responses

    @property
    def media_offset(self):
        return self._media_offset

    @media_offset.setter
    def media_offset(self, offset):
        self._media_offset = offset

    @property
    @ardt_deprecated("Use media_offset instead.")
    def media_file_offset(self):
        """
        The constant value added to all media identifiers within the underlying dataset. This is useful for when you
        want to mix AERTrials from multiple AERDataset instances.

        For example, if aerDataset1 uses media_file_offset=0, and has media identifiers 1 through 50, then you
        might instantiate aerDataset2 using participant_offset=50. Then, media identifier 1 within the second
        dataset will be loaded as media_id=51 instead, avoiding any conflict at runtime.

        :return:
        """
        return self._media_offset

    @media_file_offset.setter
    @ardt_deprecated("Use media_offset instead.")
    def media_file_offset(self, media_file_offset):
        self._media_offset = media_file_offset

    @property
    def participant_offset(self):
        """
        The constant value added to all participant identifiers within the underlying dataset. This is useful for when
        you want to mix AERTrials from multiple AERDataset instances.

        For example, if aerDataset1 uses participant_offset=0, and has participant identifiers 1 through 50, then you
        might instantiate aerDataset2 using participant_offset=50. Then, participant identifier 1 within the second
        dataset will be loaded as participant_id=51 instead, avoiding any conflict at runtime.

        :return:
        """
        return self._participant_offset

    @participant_offset.setter
    def participant_offset(self, participant_offset):
        self._participant_offset = participant_offset

    @property
    def signal_preprocessors(self):
        """
        A map of signal_type to SignalPreprocessor instance, e.g.:
            'ECG' -> ardt.preprocessors.NK2ECGPreprocess

        These are available to all AERTrial instances loaded under this dataset, and are used to process the signals
        as they are loaded from each AERTrial.

        :return:
        """
        return self._signal_preprocessors

    def get_signal_metadata(self, signal_type):
        """
        Returns a dict containing the requested signal's metadata. Mandatory keys include:
        - 'sample_rate' (in samples per second)
        - 'n_channels' (the number of channels in the signal)

        See subclasses for implementation-specific keys that may also be present.

        :param signal_type: the type of signal for which to retrieve the metadata.
        :return: a dict containing the requested signal's metadata
        """
        if signal_type not in self._signal_metadata:
            raise ValueError('get_signal_metadata not implemented for signal type {}'.format(signal_type))
        return self._signal_metadata[signal_type]

    def set_signal_metadata(self, signal_type, metadata):
        if signal_type not in self._signal_metadata:
            self._signal_metadata[signal_type] = metadata
        else:
            self._signal_metadata[signal_type].update(metadata)

    def get_balanced_dataset(self, oversample=True, use_expected_response=False):
        '''
        Returns a balanced wrapper around this dataset that ensures the number of trials represented in each quadrant
        is the same. If oversample=True, every quadrant will be oversampled to increase its size to the maximum number
        of trials per quadrant in the original dataset. If oversample=False then each quadrant will be undersampled to
        reduce its size to the smallest number of trials per quadrant in the original dataset.

        The balanced datasets trials will be randomized.

        NOTE: this method does not consider the underlying signal data... so if you are working with a particular signal
        type, e.g., ECG, in a MultiDataset that combines CUADS and ASCERTAIN, then the number of signals per quadrant
        may still be unbalanced because CUADS has 3 ECG channels per trial, where ASCERTAIN has 2. The effect of this
        should be minimal due to random sampling of trials across the MultiSet when creating the Balanced subset.

        :param oversample:
        :return:
        '''
        return BalancedWrapperDataset(self,
                                      participant_offset=self.participant_offset,
                                      mediafile_offset=self.media_file_offset,
                                      signal_metadata=self._signal_metadata,
                                      expected_responses=self._expected_responses,
                                      oversample=oversample,
                                      use_expected_response=use_expected_response)

    def get_interleaved_trial_dataset(self, use_expected_responses=False):
        trials_by_quad = {1: [], 2: [], 3: [], 4: []}
        for trial in self._all_trials:
            quad = trial.expected_response if use_expected_responses else trial.load_ground_truth()
            if quad == 0 or quad > 4:
                continue
            trials_by_quad[quad].append(trial)

        for l in trials_by_quad.values():
            random.shuffle(l)
        trial_lists = list(trials_by_quad.values())

        # Find the max length of any list
        max_length = max(len(lst) for lst in trial_lists)
        oversampled_lists = [ np.random.choice( lst, max_length, replace=True ) for lst in trial_lists ]

        def merge(lists):
            merged_list = [item for group in zip_longest(*lists, fillvalue=None) for item in group if item is not None]
            return merged_list

        return TrialWrapperDataset(
            merge(oversampled_lists),
            participant_offset=self.participant_offset,
            mediafile_offset=self.media_file_offset,
            signal_metadata=self._signal_metadata,
            expected_responses=self._expected_responses)


class TrialWrapperDataset(AERDataset):
    """
    This is a wrapper class used to create a meta-dataset around a set of trials for a split...
    """
    def __init__(self, trials, participant_offset=0, mediafile_offset=0, signal_metadata=None, expected_responses=None):
        super().__init__(participant_offset=participant_offset,
                       mediafile_offset=mediafile_offset,
                       signal_metadata=signal_metadata,
                       expected_responses=expected_responses)

        self._all_trials = trials
        self._media_names_by_id = {}

        for trial in self._all_trials:
            self._media_names_by_id[trial.media_id-mediafile_offset] = trial.media_name

    def _preload_dataset(self):
        pass

    def _load_trials(self):
        pass

    def _post_load_trials(self):
        pass

    def get_media_name_by_movie_id(self, movie_id):
        return self._media_names_by_id[movie_id]

class BalancedWrapperDataset(AERDataset):
    """
    This is a wrapper class used to create a meta-dataset around a set of trials for a split... it either over or
    undersamples trials from different quadrants to create a dataset that has an equal number of trials per quadrant.
    """
    def __init__(self, dataset, participant_offset=0, mediafile_offset=0, signal_metadata=None, expected_responses=None, oversample = True, use_expected_response=False):
        super().__init__(participant_offset=participant_offset,
                         mediafile_offset=mediafile_offset,
                         signal_metadata=signal_metadata,
                         expected_responses=expected_responses )

        trial_by_quad = {
            1: [],
            2: [],
            3: [],
            4: []
        }
        counts = {
            1: 0,
            2: 0,
            3: 0,
            4: 0
        }

        for trial in dataset.trials:
            q = trial.expected_response if use_expected_response else trial.load_ground_truth()
            if q == 0 or q > 4:
                continue

            counts[q] += 1
            trial_by_quad[q].append(trial)

        quad_size = np.max(np.array(list(counts.values()))) if oversample is True else np.min(np.array(list(counts.values())))

        self._all_trials = []
        for i in np.arange(1,5):
            self._all_trials.extend(
                np.random.choice(trial_by_quad[i],      # quadrant to select from
                                 size=quad_size,        # target size per quadrant
                                 replace=oversample     # if oversample is true, we need replace=True.
                 ))
        random.shuffle(self._all_trials)

        self._media_names_by_id = {}

        for trial in self._all_trials:
            self._media_names_by_id[trial.media_id-mediafile_offset] = trial.media_name

    def _preload_dataset(self):
        pass

    def _load_trials(self):
        pass

    def _post_load_trials(self):
        pass

    def get_media_name_by_movie_id(self, movie_id):
        return self._media_names_by_id[movie_id]


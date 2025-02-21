#  Copyright (c) 2024-2025. Affects AI LLC
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
import os.path
from os import PathLike


import tensorflow as tf
from hatch.cli import self
from tensorflow.data import AUTOTUNE
import numpy as np
import queue
import traceback

from ardt.datasets.AERTrial import TruthType, AERTrial
from ardt.datasets.AERDataset import AERDataset
from tqdm import tqdm
import multiprocessing
import threading
from typing import Iterable,Union
import time

class TFRecordDatasetGenerator:
    """
    A utility class that generates a TFRecord dataset for use in tensorflow based ML training.
    """

    def __init__(self, tfrecord_filename: PathLike|str, dataset: AERDataset, signal_types: Union[Iterable,str], signal_len: int = 0):
        self._aer_dataset = dataset
        self._tfrecord_filename = tfrecord_filename

        self._signal_len = signal_len
        self._trial_count = 0
        self._signal_counts = {}
        self._channel_counts = {}

        if isinstance(signal_types, str):
            signal_types = [signal_types]
        self._signal_types = signal_types

        for trial in self._aer_dataset.trials:
            self._trial_count += 1
            for type in self._signal_types:
                signal_metadata = trial.get_signal_metadata(type)
                self._signal_counts[type] = \
                    0 if type not in self._signal_counts else self._signal_counts[type]+1
                self._channel_counts[type] = \
                    0 if type not in self._channel_counts else self._channel_counts[type] + signal_metadata['n_channels']

        self._signal_count = sum(self._signal_counts.values())
        self._channel_count = sum(self._channel_counts.values())

    @property
    def feature_description(self):
        return {
            "signal": tf.io.FixedLenFeature([self._signal_len], tf.float32),  # Use VarLenFeature for variable-length signals
            "label": tf.io.FixedLenFeature([], tf.int64),  # Scalar label
        }

    def get_tf_record_dataset(self, generate_if_needed=False, num_parallel_calls=tf.data.AUTOTUNE, options=None, cache: Union[bool,PathLike] = True, shuffle_buffer_size=None, repeat: int = 1, batch_size=None, batch_drop_remainder=True, prefetch=False, **kwargs) -> tf.data.Dataset:
        '''
        Creates a TFRecordDataset from the tfrecord file located at `self._tfrecord_filename`. If it does not exist, and
        if `generate_if_needed` is `True`, then `self.generate(...)` will be called to generate the tfrecord file. Options to
        `self.generate(...)` are passed to this function through `**kwargs`

        :param generate_if_needed: If True, then `self.generate(...)` will be called if the tfrecord file does not exist.
        :param num_parallel_calls: The number of parallel calls to use for mapping records
        :param options: If not None, this is passed to `.with_options()` on the dataset returned.
        :param cache:   If this is True, then we enable caching on the returned dataset. If it is PathLike, then the cache is persistent at this path location. If False, or None, no caching is enabled.
        :param shuffle_buffer_size: If
        :param repeat:
        :param batch_size:
        :param batch_drop_remainder:
        :param kwargs:
        :return:
        '''
        if not os.path.exists(self._tfrecord_filename):
            if not generate_if_needed:
                raise FileNotFoundError(f"{self._tfrecord_filename} does not exist. Check the path, or call generate() first to created it.")
            else:
                self.generate(**kwargs)

        tfds = tf.data.TFRecordDataset(self._tfrecord_filename) \
            .map(self._parse_record, num_parallel_calls=num_parallel_calls)

        if options is not None:
            tfds = tfds.with_options(options)

        if cache is not None:
            if isinstance(cache, bool):
                tfds = tfds.cache()
            else:
                tfds = tfds.cache(cache)

        if shuffle_buffer_size is not None:
            tfds = tfds.shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True)

        tfds = tfds.repeat(repeat)

        if batch_size is not None:
            tfds = tfds.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE, drop_remainder=batch_drop_remainder)

        if prefetch:
            tfds = tfds.prefetch(buffer_size=tf.data.AUTOTUNE)

        return tfds

    def _parse_record(self, record_proto):
        record = tf.io.parse_single_example(record_proto, self.feature_description)
        return record["signal"], record["label"]


    def generate(self, use_expected_response=False, truth=TruthType.QUADRANT, one_channel_per_record=True, queue_depth=10, num_workers=None):
        '''
        Generates a TFRecord file containing each trial in the AERDataset

        :param use_expected_response:
        :param one_channel_per_record:
        :param queue_depth:
        :param num_workers:
        :return:
        '''
        if num_workers is None:
            num_workers = max(int(multiprocessing.cpu_count()*.75),1)      # At least 1, but not more than (cpu_count-2)
            print(f"Using {num_workers} workers to load signals")

        if queue_depth is None or queue_depth <= 0:
            queue_depth = 10

        manager = multiprocessing.Manager()
        record_queue = manager.Queue(maxsize=queue_depth)  # Buffer up to 100 records

        # Start writer thread
        writer_thread_stop_event = multiprocessing.Event()
        writer_thread = threading.Thread(target=self._writer_thread_func, args=(record_queue, writer_thread_stop_event, one_channel_per_record, queue_depth))
        writer_thread.start()

        try:
            trial_batches = np.array_split(self._aer_dataset.trials, num_workers)
            # Start worker processes
            with multiprocessing.Pool(processes=num_workers) as pool:
                pool.starmap(self._process_trial_batch, [(trial_batch, record_queue, use_expected_response, truth, one_channel_per_record) for trial_batch in trial_batches])
        except Exception as e:
            print(f"Exception occurred while processing trials: {e.__class__.__name__}: {e}")
            traceback.print_exc()  # Prints full traceback
        finally:
            writer_thread_stop_event.set()
            writer_thread.join()  # Wait for writer to finish

        print("All data has been written successfully.")

    def _writer_thread_func(self, record_queue, stop_event, one_channel_per_record, queue_depth):
        queue_unit="signal channels" if one_channel_per_record else "signals"

        with tf.io.TFRecordWriter(self._tfrecord_filename) as writer:
            with tqdm(
                total = self._channel_count if one_channel_per_record else self._signal_count,
                desc="Writing TFRecords",
                unit=queue_unit,
                position=0, leave=True
            ) as write_pbar:
                while not stop_event.is_set():
                    try:
                        record = record_queue.get(timeout=1)  # Wait for records
                        writer.write(record)
                        write_pbar.update(1)

                        write_pbar.set_description(f"Queue Depth: {max(1,record_queue.qsize()):2d}/{queue_depth:2d} | Writing TFRecords")
                    except queue.Empty:
                        continue  # Avoid blocking if queue is temporarily empty

        print(f"TFRecord writing complete: {self._tfrecord_filename}")

    def _process_trial_batch(self, trial_batch, queue, use_expected_response, truth, one_channel_per_record):
        for trial in trial_batch:
            self._process_trial(trial, queue, use_expected_response, truth, one_channel_per_record)

    def _process_trial(self, trial, queue, use_expected_response, truth, one_channel_per_record):
        """Processes a single trial and adds serialized TFRecord entries to the queue."""
        label = trial.expected_response if use_expected_response else trial.load_ground_truth()
        label = 0 if label is None else label  # Convert None to 0

        newlabel = label if truth == TruthType.QUADRANT else AERTrial.quadrant_to_valence(label) if truth == TruthType.VALENCE else AERTrial.quadrant_to_arousal(label)

        label = newlabel
        for signal_type in self._signal_types:
            signal_data = trial.load_signal_data(signal_type)
            if self._signal_len != 0 and signal_data.shape[1] != self._signal_len:
                raise ValueError(f"Actual signal length {signal_data.shape[1]} does not match expected signal length {self._signal_len}")

            if self._signal_len == 0:
                self._signal_len = signal_data.shape[1]

            if one_channel_per_record:
                for channel_num in range(signal_data.shape[0]):
                    signal = signal_data[channel_num].transpose().reshape(-1, 1)  # Shape as (N, 1)

                    # Convert to TFRecord format
                    feature = {
                        "signal": tf.train.Feature(float_list=tf.train.FloatList(value=signal.flatten())),
                        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
                    }
                    example = tf.train.Example(features=tf.train.Features(feature=feature))

                    # Put serialized record into the queue
                    queue.put(example.SerializeToString())
            else:
                raise NotImplementedError("We currently only support one channel per trial.")
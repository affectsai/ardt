import traceback

import torch

from ardt.datasets.AERTrial import TruthType
from torch.utils.data import Dataset

from ardt.datasets import AERDataset

import lmdb
import math
import numpy as np
import multiprocessing
import threading
import torch
import traceback
import os
import queue
from tqdm import tqdm


class DistributedWeightedRandomSampler(torch.utils.data.Sampler):
    def __init__(self, aerds, truth=TruthType.QUADRANT, signal_type='ECG', world_size=None, rank=None, replacement=True):
        if world_size is None:
            if not dist.is_available():
                world_size = 1
            world_size = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                rank = 0
            rank = dist.get_rank()

        self.num_replicas = world_size
        self.rank = rank
        self.replacement = replacement

        class_counts={}
        sample_labels=[]
        # Create an index mapping of (trial_index, channel_index)
        for trial_idx, trial in enumerate(aerds.trials):
            label = trial.load_ground_truth(truth)

            num_channels = trial.get_signal_metadata(signal_type)['n_channels']  # Number of channels
            sample_labels.extend([label]*num_channels)

            # Count how many samples in each class
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1

        class_counts = np.array([class_counts[i] for i in sorted(class_counts.keys())])
        class_weights = 1. / class_counts

        self.num_samples = math.ceil(len(sample_labels) / world_size)
        self.total_size = self.num_samples * world_size
        self.weights = torch.as_tensor(np.array([ class_weights[l] for l in sample_labels]))
        self.indices = None

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.rank)
        indices_global = torch.multinomial(self.weights, self.total_size, generator=generator, replacement=self.replacement)
        self.indices = indices_global[0:self.total_size:self.num_replicas]
        return iter(self.indices)

    def __len__(self):
        return self.num_samples


class TorchDatasetWrapper(Dataset):
    def __init__(self, dataset: AERDataset, signal_type: str = 'ECG', signal_len=256*30, truth=TruthType.QUADRANT, cache_path="aer_cache.lmdb"):
        """
        Args:
            aer_trials (list): List of AERTrial objects.
        """
        self._aer_dataset = dataset
        self.aer_trials = dataset.trials
        self.signal_type = signal_type
        self.signal_len = signal_len
        self.truth = truth
        self.all_samples = []
        self.cache_path = cache_path
        self.num_cache_workers = 28
        self.cache = None
        self.trials_by_key = {}

        # Create an index mapping of (trial_index, channel_index)
        for trial_idx, trial in enumerate(self.aer_trials):
            trial_key = f"trial_{trial.participant_id}_{trial.media_id}"
            self.trials_by_key[trial_key] = trial
            num_channels = trial.get_signal_metadata(self.signal_type)['n_channels']  # Number of channels
            for channel_idx in range(num_channels):
                self.all_samples.append((trial_key, channel_idx))

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):
        trial_key, channel_idx = self.all_samples[idx]
        trial = self.trials_by_key[trial_key]

        if trial is None:
            raise ValueError(f"Missing trial for key {trial_key}!?")

        cache = lmdb.open(self.cache_path, readonly=True, lock=False)

        if cache is None:
            raise ValueError("LMDB cache not initialized! Did you forget to call generate()?")

        with cache.begin() as txn:
            np_bytes = txn.get(trial_key.encode())

        if np_bytes is None:
            raise ValueError(
                f"Missing preprocessed signal for trial {idx}, key {trial_key}! Did you forget to run generate()?")

        # Convert back to NumPy
        signal_data = np.frombuffer(np_bytes, dtype=np.float32)
        signal_data = signal_data.reshape(-1,self.signal_len)

        label = trial.load_ground_truth(truth=self.truth)  # Integer label

        # Extract the specific channel
        channel_data = signal_data[channel_idx]  # Shape: (M,)

        # Convert to PyTorch tensors
        channel_data = torch.tensor(channel_data, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.int32)

        return channel_data, label

    def generate(self, queue_depth=10, num_workers=None):
        """Generates an LMDB cache for Torch dataset loading."""

        if os.path.exists(self.cache_path):
            print("✅ LMDB cache already exists.")
            return

        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)

        if num_workers is None:
            num_workers = max(int(multiprocessing.cpu_count() * 0.75), 1)
            print(f"Using {num_workers} workers to load signals")

        if queue_depth is None or queue_depth <= 0:
            queue_depth = 10

        # Shared queue for preprocessing results
        manager = multiprocessing.Manager()
        record_queue = manager.Queue(maxsize=queue_depth)

        # Start writer thread
        writer_thread_stop_event = multiprocessing.Event()
        writer_thread = threading.Thread(target=self._writer_thread_func,
                                         args=(record_queue, writer_thread_stop_event, queue_depth))
        writer_thread.start()

        try:
            trial_batches = np.array_split(self._aer_dataset.trials, num_workers)

            # Start worker processes
            with multiprocessing.Pool(processes=num_workers) as pool:
                pool.starmap(self._process_trial_batch,
                             [(trial_batch, record_queue) for trial_batch in trial_batches])
        except Exception as e:
            print(f"Exception occurred while processing trials: {e.__class__.__name__}: {e}")
            traceback.print_exc()
        finally:
            writer_thread_stop_event.set()
            writer_thread.join()  # Ensure the writer finishes before exiting

        print("✅ LMDB cache successfully created.")

    def _writer_thread_func(self, record_queue, stop_event, queue_depth):
        """Single-threaded LMDB writer that stores preprocessed signals in cache."""

        env = lmdb.open(self.cache_path, map_size=1*1024*1024*1024, writemap=True)  # Open LMDB for writing

        with env.begin(write=True) as txn:
            with tqdm(total=len(self._aer_dataset.trials), desc="Writing LMDB", unit="trials") as write_pbar:
                while not stop_event.is_set():
                    try:
                        record = record_queue.get(timeout=1)  # Wait for records
                        if record is None:
                            break  # Stop processing when queue is empty

                        trial_key, signal_bytes = record  # Unpack data
                        txn.put(trial_key.encode(), signal_bytes)  # Write to LMDB

                        # 🔥 **Verify that the key exists immediately after writing**
                        if txn.get(trial_key.encode()) is None:
                            raise ValueError(f"Write failed: {trial_key} was not stored in LMDB!")

                        write_pbar.update(1)
                        write_pbar.set_description(
                            f"Queue Depth: {max(1, record_queue.qsize()):2d}/{queue_depth:2d} | Writing LMDB")
                    except queue.Empty:
                        continue  # Avoid blocking if queue is temporarily empty

        env.close()
        print(f"✅ LMDB writing complete: {self.cache_path}")

    def _process_trial_batch(self, trial_batch, queue, batch_offset=9):
        """Processes a batch of trials and adds serialized LMDB entries to the queue."""
        for batch_idx, trial in enumerate(trial_batch):
            trial_key = f"trial_{trial.participant_id}_{trial.media_id}"
            signal_data = trial.load_signal_data(self.signal_type)
            signal_data = np.ascontiguousarray(signal_data)
            queue.put((trial_key, signal_data.tobytes()))

from .SignalPreprocessor import SignalPreprocessor
import numpy as np

class MultiChannelECGZScoreNormalization(SignalPreprocessor):
    """
    Applies z-score normalization to the input signal
    """

    def __init__(self, child_preprocessor=None, parent_preprocessor=None, channels_first=True):
        super().__init__(child_preprocessor=child_preprocessor, parent_preprocessor=parent_preprocessor)
        self.channels_first = channels_first

    def process_signal(self, signal):
        num_channels = signal.shape[0 if self.channels_first else 1]

        # Preallocate output array
        normalized_signal = np.empty_like(signal, dtype=np.float32)

        for c in range(num_channels):
            # Extract channel data
            channel_data = signal[c, :] if self.channels_first else signal[:, c]

            # Compute mean and standard deviation
            channel_mean = np.mean(channel_data)
            channel_stdev = np.std(channel_data)

            # Avoid division by zero
            channel_stdev = max(channel_stdev, 1e-8)

            # Normalize
            normalized_channel_data = (channel_data - channel_mean) / channel_stdev

            # Store normalized data
            if self.channels_first:
                normalized_signal[c, :] = normalized_channel_data
            else:
                normalized_signal[:, c] = normalized_channel_data

        return normalized_signal
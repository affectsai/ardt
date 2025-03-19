# Affective Research Dataset Toolkit (ARDT)
AARDT, pronouced "_art_," is a utility library for working with AER Datasets available to the academic community for research in 
automated emotion recognition. While it may likely be applied to datasets in other research areas, the author(s)' 
are primarily focused on AER.

Quick Index of this README:
- Want to know if you can use it? Jump to [Intended Use and License](#license)
- Want to know how to use it? Jump to [Quick Start](#quickstart)
- Want to help out? Jump to [Contributing](#contributing)

## Quick Start
__Step 1: Installation__

```bash
pip install ardt
```

__Step 2: Configuration__

Configure the paths to your AER datasets in the `ardt_config.yaml` file. 
```yaml
{
    # Some ARDT dataset implementation may need to preprocess the raw data. When this happens, it'll store
    # the intermediate outputs in the working_dir
    'working_dir': '/mnt/datasets/ardt/working_storage',
    
    # Configure any datasets you want to use... key is defined by the AERDataset implementation itself.
    # We show templates for the three dataset implementations provided out of the box, but you can add more or remove 
    #  any of these as needed.
    'working_dir': '/mnt/datasets/ardt_work_folder/',
    'datasets': {
        'ascertain': {
            # Path to the expanded ASCERTAIN dataset:
            'path': '/mnt/datasets/ascertain',
            
            # Names of the subfolders under ASCERTAIN where you expanded ASCERTAIN_Raw.zip and ASCERTAIN_Features.zip respectively:
            'raw_data_path': 'ASCERTAIN_Raw',
            'features_data_path': 'ASCERTAIN_Features'
        },
        'dreamer': {
            'path': '/mnt/datasets/dreamer',
            'dreamer_data_filename': "DREAMER_Data.json"
        },
        'cuads': {
            'path': '/mnt/datasets/cuads',
        }
    },
}
```

The configuration specifies the working path for ARDT to store dataset caches, and the location for each AER dataset 
available through the ARDT API. ARDT looks for this file in the following location, in this order:

1. The path specified by the `ARDT_CONFIG_PATH` environment variable
2. `ardt_config.yaml` in the current working directory

Be sure that your application has read access to each of the datasets' `path` locations, and read-write access to the `working_dir`.
ASCERTAIN uses approximately 1.1G of disk space, CUADS approximately 1.8G, and DREAMER approximately 5G of disk space in 
`working_dir`. Future datasets may increase the storage requirements. 

__Step 3: Using a Dataset__

In the simplest possible case, you just want to load a single dataset and iterate over its trials. Most likely you 
also want to process one of the trial's recorded signals. The following example prints trial data and does something
with that trial's ECG signal data.

```python
from ardt.datasets.cuads import CuadsDataset
from ardt.datasets import TruthType

# Loads cuads from the datasets.cuads.path in ardt_config.yaml
dataset = CuadsDataset()
dataset.preload()           # always call preload prior to load_trials
dataset.load_trials()       # loads the dataset trial data...

for trial in dataset.trials:
    print(f'Participant {trial.participant_id} viewed media file {trial.media_name} '
          f'and evaluated it into quadrant {trial.load_ground_truth(TruthType.QUADRANT)}. '
          f'Expected response was {trial.expected_response}')
    
    process_ecg_signal(trial.load_signal_data('ECG'))    
```

The pattern is the same for each supported dataset. Simply instantiate the dataset object, call the preload() method to 
ensure its working cache is initialized and populated, and then load the trials. 

Individual datasets have different data schemas and file formats. For example, ASCERTAIN is provided as a series of Matlab .mat
files; CUADS as a set of CSV files, and DREAMER as a single, large, JSON formatted datafile. Parsing these data schemas can often 
be time consuming, and memory and cpu intensive tasks. The dataset's `preload()` method is meant to mitigate this by parsing 
the dataset into an easily usable intermediate format. The implementations provided parse the datasets into individual Numpy 
files per trial, which can quickly retrieved in O(1) time at runtime.

Additionally, datasets are often very large in size and loading the full dataset into memory may be problematic in some environments. 
The `load_trials()` method only loads trial metadata, including participant and media IDs, truth values, etc. The underlying signal 
data is not retrieved until it is needed when handling an individual trial. 

The dataset trials are instances of AERTrial. An AERTrial represents a single participant viewing a single stimulus. The 
"ground truth" represents how the participant rated their own emotional response and is encoded as the quadrant number within 
the arousal/valence plane. The raw arousal and valence scores are not currently exposed, but may be added in a future update. 
The ground truth is provided as a quadrant number by default, but may also be requested using `TruthType.AROUSAL` and 
`TruthType.VALENCE` instead. 

__Step 4: Learn About What Else You Can Do:__

ARDT is a versatile framework that allows you to work with multiple datasets simultaneously. It provides APIs to wrap 
the datasets in TensorFlow Datasets for machine learning, and a comprehensive preprocessing pipeline for signal filtering
and manipulation. 

Much of this is covered in this README. For additional assistance you can open an issue on our GitHub, or reach out to
the authors directly.

You will also find comprehensive examples in this the [CUADS Data Quality Notebook](https://github.com/affectsai/cuads_dataset/blob/ardt/CUADS_Data_Quality.ipynb). 

## Intended Use and License
<a name="license"></a>
This library is intended for use by only by academic researchers to facilitate advancements in emotion research. It is 
__not for commercial use__ under any circumstances.

This library is licensed under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en) 
International License. 

__You are free to__:
* __Share__ — copy and redistribute the material in any medium or format 
* __Adapt__ — remix, transform, and build upon the material

__Under the followiung terms__:
* __Attribution__ — You must [give appropriate credit](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en#ref-appropriate-credit), provide a link to the license, and [indicate if changes were made](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en#ref-indicate-changes). You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
* __NonCommercial__ — You may not use the material for [commercial purposes](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en#ref-commercial-purposes)
* __ShareALike__ — If you remix, transform, or build upon the material, you must distribute your contributions under the [same license](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en#ref-same-license) as the original.
* __No additional restrictions__ — You may not apply legal terms or [technological measures](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en#ref-technological-measures) that legally restrict others from doing anything the license permits.

## Quick Start 
<a name="quickstart"></a>
### Concepts
AARDT is designed with a few simple concepts:
1. A _trial_ is a single session in which a participant is exposed to an emotional stimulus, and includes data from one or more sensors captured during the session. This may include ECG, EEG, video or audio recordings of the participant, or whatever else you can think of.
2. A _dataset_ is a collection of trials from multiple participants
3. Sensor data from a trial may need to be processed before being used, and you can do so using the _preprocessor_ pipeline

Most importantly, AER datasets __are not distributed__ with this library. You need to request access to the datasets from the 
dataset authors and download them before following this guide.

### Loading signals from dataset trials
In this example we assume that you have downloaded DREAMER, which is provided in a single JSON file, and that it is stored at `${DREAMER_HOME}/DREAMER_Data.json`.

__Step 1 - Instantiate an AERDataset:__
The AERDataset is the baseclass for all AER datasets, and the details of interacting with each one is in its subclasses, which currently includes `ardt.datasets.ascertain.AscertainDataset` and `ardt.datasets.dreamer.DreamerDataset`. Instantiate a `DreamerDataset` like so:

```python
import os
from ardt.datasets.dreamer import DreamerDataset

# Typically you'd load this from a configuration file... we'll get to that later.
dreamer_home = os.environ['DREAMER_HOME']
ecg_dataset = DreamerDataset(dreamer_home, signals=['ECG'])
```

The `signals` argument takes a list of signals to load into the AERDataset, and can be any proper subset of the signals available within 
the dataset in question. DREAMER provides ECG and EEG recordings, so you can specify any of `['EEG']`, `['ECG']`, or `['EEG','ECG']`. The
order specified does not matter.

__Step 2 - preload and load the dataset__:
Now that you have the DreamerDataset, there are two steps to get it ready for use: _preload_, and _load_. 

The _preload_ step performs any preprocessing of the raw dataset provided by the dataset authors necessary to get it ready to use in AARDT. DREAMER, for example, is provided 
as a single JSON file that is several gigabytes in size. AARDT's preload breaks the JSON into individual Numpy files for each trial, without
ever loading the entire JSON file into memory. This allows it to be used on memory constrained systems, and enables efficient prefetching from 
NAS storage. The preload mechanism is cached, and therefore only runs the first time it is invoked on a given dataset. The preload mechanism only
preloads the signals listed when the dataset was constructed, and will automatically re-run if a new signal is requested that was not 
included in the previous preload.

The _load_ step populates the datasets list of trials, with metadata only. Signal data is lazy-loaded later.

```python
# preload only runs once, regardless of how many times you call it
# so there is no need to check. 
ecg_dataset.preload()

# after preloading, you can load the trials
ecg_dataset.load_trials()
```

__Step 3 - obtain signal data from the trials__: 
With the trials loaded, you can now obtain the signal data and do your analysis on it.

```python
for trial in ecg_signal.trials:
    ecg_signal = trial.load_signal_data('ECG')
    process_ecg(ecg_signal)
```

That's it! And its the same regardless of which AER dataset you are using. If you want to use ASCERTAIN instead of DREAMER, 
just replace `ardt.datasets.dreamer.DreamerDataset` with `ardt.datasets.ascertain.AscertainDataset`, everything else remains
the same.

### Preprocessing signals
The first step in virtually all workloads is to preprocess the signal data, and you can use AARDTs _preprocessors_ to build 
an automated pipeline to do that when signals are loaded from a trial.

For example, lets assume you want to trim each ECG signal in the DREAMER dataset to the final 30 seconds of the sample. You can use 
the `FixedDurationPreprocessor` to do this automatically, like so:

```python
import os
from ardt.datasets.dreamer import DreamerDataset
from ardt.preprocessors import FixedDurationPreprocessor

# Typically you'd load this from a configuration file... we'll get to that later.
dreamer_home = os.environ['DREAMER_HOME']
ecg_dataset = DreamerDataset(dreamer_home, signals=['ECG'])

# Add the preprocessor pipeline to the dataset, for the signal it should be applied to.
# Each signal type can have its own preprocessor pipeline.
ecg_dataset.signal_preprocessors['ECG'] = FixedDurationPreprocessor(signal_duration=30, sample_rate=256,
                                                                    padding_value=0)

# Preload and load the dataset...
ecg_dataset.preload()
ecg_dataset.load_trials()

for trial in ecg_dataset.trials:
    # When you request the signal data from the trial, if the dataset
    # has a preprocessor for that signal type, it will be applied to the
    # signal before it is returned. You are guaranteed to have a 30s 
    # sample here.
    #
    # If the signal was less than 30s originally, it was padded on the left 
    # with 0 values. 
    ecg_signal_30s = trial.load_signal_data('ECG')

    # Do something with ecg_signal_30s
```

### Creating your own preprocessor, and preprocessor chaining
You can subclass `SignalPreprocessor` to create your own, and preprocessors can be chained together. For example,
let's say we want to normalize the signal to values between 0 and 1, and also trim them to 30 seconds fixed duration.

```python
import os
import numpy as np
from sklearn import preprocessing as p

from ardt.datasets.dreamer import DreamerDataset
from ardt.preprocessors import FixedDurationPreprocessor


class MyNormalizer(ardt.preprocessors.SignalPreprocessor):
    def __init__(self, parent_preprocessor=None):
        super().__init__(parent_preprocessor)

    def process_signal(self, signal):
        min_max_scaler = p.MinMaxScaler()
        return min_max_scaler.fit_transform(signal)


dreamer_home = os.environ['DREAMER_HOME']
ecg_dataset = DreamerDataset(dreamer_home, signals=['ECG'])

# Create a pipeline by instantiating MyNormalizer, and passing in a 
# FixedDurationPreprocessor as its parent. You can chain as many 
# preprocessors together as you need. The parent will always be called
# first - so the outermost preprocessor is the last one to execute.
pipeline = MyNormalizer(
    FixedDurationPreprocessor(signal_duration=30, sample_rate=256, padding_value=0)
)

ecg_dataset.signal_preprocessors['ECG'] = pipeline

# Preload and load the dataset...
ecg_dataset.preload()
ecg_dataset.load_trials()

for trial in ecg_dataset.trials:
    # Here, the signal data is already trimmed or padded to be 30s long, 
    # and has been normalized using the MinMaxScaler to values between 
    # 0 and 1.
    ecg_signal = trial.load_signal_data('ECG')

```

Note that the order of your pipeline is critically important. Here, we apply `FixedDurationPreprocessor` first, _before_ 
we normalize the values. This may be problematic, since ECG signals are prone to baseline wander. Padding zero values in 
before normalization will artificially skew the normalization results. It would be better to normalize the signal first, 
then apply the `FixedDurationPreprocessor`:

```python
pipeline = FixedDurationPreprocessor(
    signal_duration=30, 
    sample_rate=256, 
    padding_value=0, 
    parent_preprocessor=MyNormalizer()    
)
```

Alternatively you can use the `child_preprocessor` to chain the other way:
```python
pipeline = MyNormalizer(
    child_preprocessor=FixedDurationPreprocessor(signal_duration=30, sample_rate=256, padding_value=0)
)
```

A `child_preprocessor` will be invoked _after_ the preprocessor completes, so this achieves the same effect of 
normalizing the signal first, then truncating or padding it to 30 seconds.

### Using with TensorFlow
To facilitate use with TensorFlow, use the `TFDatasetWrapper` to decorate your `AERDataset` as a `tf.data.Dataset` suitable 
for use with `tf.model.fit()`

```python
import ardt.datasets

# Don't forget to setup your preprocessor pipelines, then preload and 
# load the dataset first!
tfdsw = ardt.datasets.TFDatasetWrapper(ecg_dataset)

# Create the tf.data.Dataset 
tfdataset = tfdsw('ECG', batch_size=64, buffer_size=500, repeat=1)

# Setup your tensorflow model, then use the tfdataset:
myModel = get_tensorflow_model()

# Train your model using preprocessed signals from the AERDataset
myModel.fit(tfdataset)
```

To separate training, validation and test splits, you can specify the splits to the `TFDatasetWrapper` and then indicate
which split you intend when you call it.

```python
import ardt.datasets

# Don't forget to setup your preprocessor pipelines, then preload and 
# load the dataset first!

# Specify 60% of participants for the training split, 30% for validation and 10% for testing.
tfdsw = ardt.datasets.TFDatasetWrapper(ecg_dataset, splits=[.6, .3, .1])

# Setup your tensorflow model, then use the tfdataset:
myModel = get_tensorflow_model()

# Train your model using preprocessed signals from the AERDataset, using trials from the split at index 0 (60%)
myModel.fit(
    x=tfdsw('ECG', n_split=0),
    validation_data=tfdsw('ECG', n_split=1)
    ...
)

# Later, evaluate against the test set
reults = myModel.evaluate(
    x=tfdsw('ECG', n_split=2)
)
```

TFDatasetWrapper provides a `tf.data.dataset` which will prefetch up to `buffer_size` trials at random, creating batches of 
size `batch_size`, and will iterate the dataset `repeat` times. The prefetch queue uses `tf.data.AUTOTUNE` to self-optimize.

## Adding New Datasets
Whether you are creating your own dataset, or just want to use one that isn't already included, AARDT is designed to be 
extensible allowing you to integrate additional datasets as needed. This section serves as a guide to help you do this. 

### Step 1: Dataset Paths
Dataset paths are configured in the `config.yml` file. Each dataset has its own section, and you can add new ones as needed. For example, to add the CUADS dataset, we did this:
```config.yml
config = {
    'working_dir': '/mnt/affectsai/aerds/',
    'datasets': {
        ...,
        'cuads': {
            'path': '/mnt/affectsai/datasets/cuads',
        }
    },
}
```

Any additional properties you need can be added under the `cuads` element. 

### Step 2: Implement AERDataset and AERTrial Subclasses
The `AERDataset` is the base class for all dataset implementations in AARDT. It is primarily responsible for loading 
instances of `AERTrial`. 

All the implementation details, including dataset layout and access details, are encapsulated in your implementation of 
this base class. See any of the existing implementations for examples. We provide implementations for ASCERTAIN, CUADS, 
and DREAMER each of which is thoroughly commented. See 
* `src/ardt/datasets/ascertain/Ascertaindataset.py`, 
* `src/ardt/datasets/dreamer/DreamerDataset.py`, 
* `src/ardt/datasets/cuads/CuadsDataset.py`.

To extend `AERDataset` do the following:
1. Create a new class as a subclass of AERDataset like so:
    ```python
    from ardt.datasets import AERDataset
    
    class MyAwesomeDataset(AERDataset):
        def __init__(self, signals):
            super().__init__(signals)
    ```
    You should minimally provide a list of signal types to `super.__init__`. This is a list of signal types provided by this
   dataset, e.g.: `['ECG','EEG']`. Feel free to add whatever additional arguments you might need to support your implementation.


2. Override `load_trials(self)` and `get_signal_metadata` methods from AERDataset. `load_trial(self)` is where all the
   hard work of implementing a dataset is done... here, you will parse the dataset to produce individual `AERTrial` 
instances. `get_signal_metadata(self,signal)` returns a map of metadata about the requested signal. Minimally this should include: 
   * `n_channels`: the number of channels for this signal, and
   * `sample_rate`: the sample rate in Hz for this signal
    ```python
    from ardt.datasets import AERDataset
        
    class MyAwesomeDataset(AERDataset):
        def __init__(self, signals):
            if signals is None:
                signals = ['ECG']       # If not specified, let's load ECG signals from MyAwesomeDataset...
                    
            super().__init__(signals)
        
        @abc.abstractmethod
        def load_trials(self):
            """
            Loads the AERTrials from the preloaded dataset into memory. This method should load all relevant trials from
            the dataset. To avoid memory utilization issues, it is strongly recommended to defer loading signal data into
            the AERTrial until that AERTrial's load_signal_data method is called.
        
            During load_trials, implementations should populate `self.trials`. Trial participant and media identifiers must
            be numbered sequentially from 1 to N where N is the number of participants or media files in the dataset
        
            See subclasses for dataset-specific details.
            :return:
            """
            mytrials = []  # actually load your trial data...
            self.trials.extend( mytrials )
        
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
        
            if signal_type == 'ECG':
                return {
                    'n_channels': 2,
                    'sample_rate': 256
                }
                
            return {}
    ```
   

   3. Create a new class as a subclass of AERTrial like so:
       ```python
       from ardt.datasets import AERTrial
    
       class MyAwesomeDatasetTrial(AERTrial):
           def __init__(self, dataset, participant_id, movie_id)):
               super().__init__(dataset, participant_id, movie_id))
    
           @abc.abstractmethod
           def load_signal_data(self, signal_type):
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
           def load_ground_truth(self):
               """
               Returns the ground truth label for this trial. For AER trials, this is the quadrant within the A/V space,
               numbered 0 through 3 as follows:
               - 0: High Arousal, High Valence
               - 1: High Arousal, Low Valence
               - 2: Low Arousal, Low Valence
               - 3: Low Arousal, High Valence
    
               :return: The ground truth label for this trial
               """
               return 0
    
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
    
               response = self.dataset.get_signal_metadata(signal_type)
               response['duration'] = 60 # get the length of the signal data
      
               return response    
       ```
   
       The AERTrial takes a reference to the dataset that created it, and the participant_id and media_id that this
      trial represents. It must implement `load_signal_data` and `load_ground_truth` as documented. It may optionally
      override `get_signal_metadata` to augment the response from the dataset, for example, to include signal duration.

There is more to it than this but this should be enough to get you started. See the `AERDataset` and `AERTrial` classes
for method documentation, and then CUADS, ASCERTAIN and DREAMER examples for guidance.

## Contributing <a name="contributing"></a>
We are happy to support you by accepting pull requests that make this library more broadly applicable, or by accepting
issues to do the same. If you have an AER dataset you would like us to integrate, please open an issue for that as well, 
but we will be unable to process issues requesting integration with non-AER datasets at this time.

If you would like to get involved by maintaining dataset integrations in other areas of research, please get in touch 
and we'd be happy to have the help!
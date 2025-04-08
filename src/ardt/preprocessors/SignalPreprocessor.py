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

from abc import ABCMeta, abstractmethod


class SignalPreprocessor(metaclass=ABCMeta):
    """
    SignalPreprocessor is an abstract class that defines the interface for signal preprocessing pipelines

    Example usage::

        input_signal = np.random.rand(1, 100)
        processor = MySignalProcessor(parent_preprocessor=None, child_preprocessor=None)
        processed_signal = processor(input_signal)

    Each signal processor has an optional parent and child preprocessor. If a parent preprocessor is given, then it will
    be executed before this preprocessor is applied. If a child preprocessor is given, then it will be executed after
    this preprocessor is applied. This allows for building complex preprocessor chains to be applied to the data.

    Parameters
    ----------
    parent_preprocessor : SignalProcessor, optional
        The parent preprocessor chain
    child_preprocessor : SignalPreprocessor, optional
        The child preprocessor chain
    """
    def __init__(self, parent_preprocessor=None, child_preprocessor=None):
        self._parent_preprocessor = parent_preprocessor
        self._child_preprocessor = child_preprocessor
        self._context = {}

    def resolve_processor_chain(self, chain=None):
        """
        Deprecated - use resolve() instead
        """
        return self.resolve(chain)

    @abstractmethod
    def process_signal(self, signal):
        """
        Executes this processor chain on the given signal

        Parameters
        ----------
        signal : np.ndarray
            Time-series input signal with shape NxM where N is the number of channels and M is the length of the signal.

        Return
        ------
        np.ndarray
            The processed signal. SignalPreprocessors may modify the shape of the signal so be sure to check the number
            of channels and length of the result.
        """
        pass

    @property
    def context(self):
        """
        The context dictionary is used to store any data that needs to be shared across processors in the chain.

        :return:
        """
        return self._context

    def resolve(self, chain=None):
        """
        Resolves the processor chain for debugging.

        Return
        ------
        list<str>
            Returns a list of preprocessor type names, in the order in which they will be executed when this preprocessor is run.
        """
        if chain is None:
            chain = []

        chain = chain if self._parent_preprocessor is None else self._parent_preprocessor.resolve(chain)
        chain.append(self.__class__.__name__)

        if self._child_preprocessor is not None:
            self._child_preprocessor.resolve(chain)

        return chain

    def __call__(self, signal, context=None, *args, **kwargs):
        if context is None:
            context = {}

        self.context.update(context)

        result = signal
        result = self._parent_preprocessor(result, self.context) if self._parent_preprocessor is not None else result
        result = self.process_signal(result)
        result = self._child_preprocessor(result, self.context) if self._child_preprocessor is not None else result

        context.update(self.context)
        return result

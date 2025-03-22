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
    def __init__(self, parent_preprocessor=None, child_preprocessor=None):
        """
        Constructs a signal preprocessor with an optional parent and child preprocessor chain.
        If parent_preprocessor is given, then it will be called before this preprocessor is applied. If
        child_preprocessor is given, then it will be called after this preprocessor is applied. This allows for
        building complex preprocessor chains to be applied to the data.

        :param parent_preprocessor:
        """
        self._parent_preprocessor = parent_preprocessor
        self._child_preprocessor = child_preprocessor
        self._context = {}

    def resolve_processor_chain(self, chain=None):
        master = chain == None
        if master:
            chain = []

        if self._parent_preprocessor is not None:
            self._parent_preprocessor.resolve_processor_chain(chain)

        chain.append(type(self).__name__)

        if self._child_preprocessor is not None:
            self._child_preprocessor.resolve_processor_chain(chain)

        return chain

    @abstractmethod
    def process_signal(self, signal):
        """
        :param signal: The signal to trim, with size NxM where N is the number of channels, and M is the number of samples.
        :param context: the context dictionary for this chain
        :param args:
        :param kwargs:
        :return:
        """
        pass

    @property
    def context(self):
        """
        Gets this preprocessor's context dict.

        :return:
        """
        return self._context

    def resolve(self, chain=None):
        """
        Resolves the preprocessor chain.

        :param chain: a list of preprocessor type names that will already be called prior to this preprocessor, or None.
        :return: a list of preprocessor type names, in the order in which they will be executed when this
        preprocessor is called.
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
        # print(f"Executing preprocessor: {type(self)}. Before: shape={result.shape}, min: {result.min()}, max: {result.max()}")
        result = self.process_signal(result)
        # print(f"Done executing preprocessor: {type(self)}. After: hape={result.shape}, min: {result.min()}, max: {result.max()}")
        result = self._child_preprocessor(result, self.context) if self._child_preprocessor is not None else result

        context.update(self.context)
        return result

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

from ardt.preprocessors import SignalPreprocessor


class PreprocessorA(SignalPreprocessor):
    def process_signal(self, signal):
        self.context[self.__class__.__name__] = self.context['counter']
        self.context['counter'] += 1

    def __init__(self, parent_preprocessor=None, child_preprocessor=None):
        SignalPreprocessor.__init__(self, parent_preprocessor, child_preprocessor)


class PreprocessorB(SignalPreprocessor):
    def process_signal(self, signal):
        self.context[self.__class__.__name__] = self.context['counter']
        self.context['counter'] += 1

    def __init__(self, parent_preprocessor=None, child_preprocessor=None):
        SignalPreprocessor.__init__(self, parent_preprocessor, child_preprocessor)


class PreprocessorC(SignalPreprocessor):
    def process_signal(self, signal):
        self.context[self.__class__.__name__] = self.context['counter']
        self.context['counter'] += 1

    def __init__(self, parent_preprocessor=None, child_preprocessor=None):
        SignalPreprocessor.__init__(self, parent_preprocessor, child_preprocessor)


class PreprocessorD(SignalPreprocessor):
    def process_signal(self, signal):
        self.context[self.__class__.__name__] = self.context['counter']
        self.context['counter'] += 1

    def __init__(self, parent_preprocessor=None, child_preprocessor=None):
        SignalPreprocessor.__init__(self, parent_preprocessor, child_preprocessor)


class PreprocessorChaining(unittest.TestCase):
    def test_parent_call(self):
        chain = PreprocessorA(parent_preprocessor=PreprocessorB(parent_preprocessor=PreprocessorC()))
        stack = chain.resolve()
        self.assertEqual(len(stack), 3)
        self.assertEqual(stack[0], PreprocessorC.__name__)
        self.assertEqual(stack[1], PreprocessorB.__name__)
        self.assertEqual(stack[2], PreprocessorA.__name__)

    def test_child_call(self):
        chain = PreprocessorA(child_preprocessor=PreprocessorB(child_preprocessor=PreprocessorC()))
        stack = chain.resolve()
        self.assertEqual(len(stack), 3)
        self.assertEqual(stack[0], PreprocessorA.__name__)
        self.assertEqual(stack[1], PreprocessorB.__name__)
        self.assertEqual(stack[2], PreprocessorC.__name__)

    def test_mixed_call(self):
        chain = PreprocessorA(
            child_preprocessor=PreprocessorB(
                parent_preprocessor=PreprocessorC(
                    child_preprocessor=PreprocessorD())))

        stack = chain.resolve()
        self.assertEqual(len(stack), 4)
        self.assertEqual(stack[0], PreprocessorA.__name__)
        self.assertEqual(stack[1], PreprocessorC.__name__)
        self.assertEqual(stack[2], PreprocessorD.__name__)
        self.assertEqual(stack[3], PreprocessorB.__name__)

    def test_context_call(self):
        # An initial context to pass into the call chain ...
        context = {'counter': 1}

        # The call chain, expected execution order is A->C->D->B
        chain = PreprocessorA(
            child_preprocessor=PreprocessorB(
                parent_preprocessor=PreprocessorC(
                    child_preprocessor=PreprocessorD())))

        # Execute the chain with the initial context
        chain([], context)

        # Assert expected values in the context
        self.assertEqual(context[PreprocessorA.__name__], 1)
        self.assertEqual(context[PreprocessorC.__name__], 2)
        self.assertEqual(context[PreprocessorD.__name__], 3)
        self.assertEqual(context[PreprocessorB.__name__], 4)
        self.assertEqual(context['counter'], 5)


if __name__ == '__main__':
    unittest.main()

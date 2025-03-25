import abc
from typing import Callable

from ardt.datasets import AERTrial


class AERDerivedSignal(Callable):
    def __init__(self, name: str="IBI", description: str="Interbeat interval derived from ECG", source_signal: str="ECG"):
        self.name = name
        self.description = description
        self.source_signal = source_signal

    def __call__(self, trial: AERTrial):
        if self.source_signal not in trial.signal_types:
            raise ValueError('Derived signal {} requires origin signal type {} from trial'.format(self.name, self.source_signal))

        return self.get_derived_signal(trial)

    @abc.abstractmethod
    def get_derived_signal(self, trial: AERTrial):
        pass
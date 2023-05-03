from abc import ABC, abstractmethod

class FilterWheel(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @property
    @abstractmethod
    def FocusOffsets(self):
        pass

    @property
    @abstractmethod
    def Names(self):
        pass

    @property
    @abstractmethod
    def Position(self):
        pass
    @Position.setter
    @abstractmethod
    def Position(self, value):
        pass
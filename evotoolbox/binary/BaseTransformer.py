from abc import ABC, abstractmethod


class BaseTransformer(ABC):
    @abstractmethod
    def transform(self, solution):
        pass


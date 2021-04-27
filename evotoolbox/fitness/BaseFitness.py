from abc import ABC, abstractmethod


class BaseFitness(ABC):
    def setdata (self, features, labels):
        self.features = features
        self.labels = labels
        
    
    @abstractmethod
    def evaluate(self, solution):
        pass
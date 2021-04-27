from abc import ABC, abstractmethod


class BaseInitializer(ABC):
    def __init__ (self, n_features, n_agents):
        self.n_features = n_features
        self.n_agents = n_agents

    @abstractmethod
    def init(self, solution):
        pass

from abc import ABC, abstractmethod


class BaseOptimizer(ABC):
    def __init__(self, binary_transformer, max_iter, lb, ub):
        self.transformer_name = type(binary_transformer).__name__
        self.binary_transformer = binary_transformer.transform
        self.max_iter = max_iter
        self.lb = lb
        self.ub = ub
        
        
    @abstractmethod
    def optimize(self, fitness_func, initial_positions, n_features, n_agents):
        pass



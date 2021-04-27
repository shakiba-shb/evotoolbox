import numpy as np
from evotoolbox.initializers import BaseInitializer

class RandomInitializer(BaseInitializer):
    def init(self, fitness_func):
        positions = np.zeros((self.n_agents, self.n_features))
        for i in range(self.n_features):
            positions[:, i] = np.random.uniform(0, 1, self.n_agents)
        positions[positions > 0.5] = 1
        positions[positions <= 0.5] = 0
        
        return positions
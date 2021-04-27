import numpy as np
from evotoolbox.binary import BaseTransformer

class SigmoidTransformer(BaseTransformer):
    def __init__(self, alpha = 1):
        self.alpha = alpha
        
    def transform(self, solution):
        sol_bin = np.zeros_like(solution, dtype='int')
        r = 1 / (1 + np.exp(-1 * self.alpha * solution))
        for i in range(sol_bin.shape[0]):
            if r[i] < np.random.random():
                sol_bin[i] = 1
        return sol_bin

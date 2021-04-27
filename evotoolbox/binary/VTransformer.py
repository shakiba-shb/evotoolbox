import numpy as np
from evotoolbox.binary import BaseTransformer

class VTransformer(BaseTransformer):      
    def transform(self, solution):
        sol_bin = np.zeros_like(solution, dtype='int')
        r = np.abs(np.tanh(solution))
        for i in range(sol_bin.shape[0]):
            if r[i] < np.random.random():
                sol_bin[i] = 1
        return sol_bin


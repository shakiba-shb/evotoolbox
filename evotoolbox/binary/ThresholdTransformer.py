import numpy as np
from evotoolbox.binary import BaseTransformer

class ThresholdTransformer(BaseTransformer):
    def __init__(self, threshold):
        self.threshold = threshold
        
    def transform(self, solution):
        sol_bin = np.zeros_like(solution, dtype='int')
        for i in range(sol_bin.shape[0]):
            if solution[i] > self.threshold:
                sol_bin[i] = 1
        return sol_bin
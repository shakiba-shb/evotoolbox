import numpy as np
from evotoolbox.binary import BaseTransformer

class QTransformer(BaseTransformer):
    def __init__(self, xmax, power):
        self.xmax = xmax
        self.power = power
        
    def transform(self, solution):
        sol_bin = np.zeros_like(solution, dtype='int')
        for i in range(sol_bin.shape[0]):
            if solution[i] < self.xmax / 2:
                r = np.power(np.abs((solution[i])/(0.5 * self.xmax)), self.power)
            else:
                r = 1
            if np.random.random() < r:
                sol_bin[i] = 1
            else:
                sol_bin[i] = 0
        return sol_bin

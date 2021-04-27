import numpy as np
from evotoolbox.initializers import BaseInitializer

class GreedyInitializer(BaseInitializer):
    def init(self, fitness_func):
        pos = np.eye(self.n_features)
        fit = []
        
        positions = np.zeros((self.n_agents, self.n_features))
        for i in range(self.n_features):
            positions[:, i] = np.random.uniform(0, 1, self.n_agents)
        positions[positions > 0.5] = 1
        positions[positions <= 0.5] = 0   
        
        for i in range(self.n_features):
            fit.append(fitness_func(pos[i]))
        feature_ranking = np.argsort(fit)
        for i in range(self.n_agents//2):
            print(i)
            f = fitness_func(positions[i])
            for j in reversed(feature_ranking):
                if positions[i,j]== 0:
                    positions[i,j]=1
                    f_new = fitness_func(positions[i])
                    if f_new < f:
                        f = f_new
                    else:
                        positions[i,j]=0
            for j in (feature_ranking):
                if positions[i,j]== 1 and np.sum(positions[i]) > 1:
                    positions[i,j]=0
                    f_new = fitness_func(positions[i])
                    if f_new <= f:
                        f = f_new
                    else:
                        positions[i,j]=1
         
        return positions 
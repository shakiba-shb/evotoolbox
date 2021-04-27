import numpy as np
import math
import random
from numpy.random import rand
from evotoolbox.optimizers import BaseOptimizer

class SSA(BaseOptimizer):
    
    def __init__(self, binary_transformer, max_iter, lb, ub):
        super().__init__(binary_transformer, max_iter, lb, ub)
    
    
    def optimize(self, fitness_func, initial_positions, n_features, n_agents):
        lb = [self.lb] * n_features
        ub = [self.ub] * n_features
        
        convergence_curve = np.zeros(self.max_iter)
        
        BinaryPositions = np.zeros((n_agents, n_features), dtype='int')
        SalpPositions = initial_positions
        SalpFitness = np.full(n_agents, float("inf"))

        FoodPosition = np.zeros(n_features)
        FoodFitness = float("inf")
        
        for i in range(n_agents):
            SalpFitness[i] = fitness_func(SalpPositions[i, :])
        
        I = np.argsort(SalpFitness)
        Sorted_salps_fitness = np.copy(SalpFitness[I])
        Sorted_salps = np.copy(SalpPositions[I, :])
    
        FoodPosition = np.copy(Sorted_salps[0, :])
        BinaryFoodPosition = np.copy(Sorted_salps[0, :])
        FoodFitness = Sorted_salps_fitness[0]
        
        for t in range(self.max_iter):
            c1 = 2 * math.exp(-((4 * t / self.max_iter) ** 2))
            
            for i in range(n_agents):
                if i < n_agents / 2:
                    for j in range(0, n_features):
                        c2 = random.random()
                        c3 = random.random()
                        # Eq. (3.1) in the paper
                        if c3 < 0.5:
                            SalpPositions[i, j] = FoodPosition[j] + c1 * (
                                (ub[j] - lb[j]) * c2 + lb[j]
                            )
                        else:
                            SalpPositions[i, j] = FoodPosition[j] - c1 * (
                                (ub[j] - lb[j]) * c2 + lb[j]
                            )
                else:
                    point1 = SalpPositions[i - 1]
                    point2 = SalpPositions[i]

                    SalpPositions[i] = (point2 + point1) / 2
                    # Eq. (3.4) in the paper
            
            for i in range(n_agents):
                # Check if salps go out of the search spaceand bring it back
                for j in range(n_features):
                    SalpPositions[i, j] = np.clip(SalpPositions[i, j], lb[j], ub[j])
                    
                BinaryPositions[i] = self.binary_transformer(SalpPositions[i])
                
                SalpFitness[i] = fitness_func(BinaryPositions[i])
                
                if SalpFitness[i] < FoodFitness:
                    FoodPosition = np.copy(SalpPositions[i, :])
                    BinaryFoodPosition = np.copy(BinaryPositions[i, :])
                    FoodFitness = SalpFitness[i]
            
            print([
                    "At iteration "
                    + str(t)
                    + " the best fitness is "
                    + str(FoodFitness)
            ])
            convergence_curve[t] = FoodFitness

        return {
            'solution': BinaryFoodPosition,
            'c': convergence_curve,
            'nf': np.sum(BinaryFoodPosition)
        }
            
            
            
            
            
            
            
            
            
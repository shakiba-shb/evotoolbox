import numpy as np
import math
import random
from numpy.random import rand
from evotoolbox.optimizers import BaseOptimizer

class SSAGWO(BaseOptimizer):
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
        
        Alpha_pos = np.zeros(n_features)
        Alpha_fitness = float("inf")
    
        Beta_pos = np.zeros(n_features)
        Beta_fitness = float("inf")
    
        Delta_pos = np.zeros(n_features)
        Delta_fitness = float("inf")
        
        for t in range(self.max_iter):
            c1 = 2 * math.exp(-((4 * t / self.max_iter) ** 2))
            a = 2 - t * ((2) / self.max_iter)
            
            for i in range(n_agents//2):
                for j in range(n_features):
    
                    r1 = random.random()  # r1 is a random number in [0,1]
                    r2 = random.random()  # r2 is a random number in [0,1]
    
                    A1 = 2 * a * r1 - a
                    # Equation (3.3)
                    C1 = 2 * r2
                    # Equation (3.4)
    
                    D_alpha = abs(C1 * Alpha_pos[j] - SalpPositions[i, j])
                    # Equation (3.5)-part 1
                    X1 = Alpha_pos[j] - A1 * D_alpha
                    # Equation (3.6)-part 1
    
                    r1 = random.random()
                    r2 = random.random()
    
                    A2 = 2 * a * r1 - a
                    # Equation (3.3)
                    C2 = 2 * r2
                    # Equation (3.4)
    
                    D_beta = abs(C2 * Beta_pos[j] - SalpPositions[i, j])
                    # Equation (3.5)-part 2
                    X2 = Beta_pos[j] - A2 * D_beta
                    # Equation (3.6)-part 2
    
                    r1 = random.random()
                    r2 = random.random()
    
                    A3 = 2 * a * r1 - a
                    # Equation (3.3)
                    C3 = 2 * r2
                    # Equation (3.4)
    
                    D_delta = abs(C3 * Delta_pos[j] - SalpPositions[i, j])
                    # Equation (3.5)-part 3
                    X3 = Delta_pos[j] - A3 * D_delta
                    # Equation (3.5)-part 3
    
                    SalpPositions[i, j] = (X1 + X2 + X3) / 3  # Equation (3.7)
            
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

                    if(random.random() < 0.5):
                        SalpPositions[i] = (point2 + point1 + FoodPosition) / 3
                    else:
                        SalpPositions[i] = (point2 + point1) / 2

                    # Eq. (3.4) in the paper
            
            for i in range(n_agents):
                # Check if salps go out of the search spaceand bring it back
                for j in range(n_features):
                    SalpPositions[i, j] = np.clip(SalpPositions[i, j], lb[j], ub[j])
                    
                BinaryPositions[i] = self.binary_transformer(SalpPositions[i])
                
                SalpFitness[i] = fitness_func(BinaryPositions[i])
                
                if SalpFitness[i] < Alpha_fitness:
                    Delta_fitness = Beta_fitness  # Update delte
                    Delta_pos = Beta_pos.copy()
                    Beta_fitness = Alpha_fitness  # Update beta
                    Beta_pos = Alpha_pos.copy()
                    Alpha_fitness = SalpFitness[i]
                    # Update alpha
                    Alpha_pos = SalpPositions[i, :].copy()
    
                if SalpFitness[i] > Alpha_fitness and SalpFitness[i] < Beta_fitness:
                    Delta_fitness = Beta_fitness  # Update delte
                    Delta_pos = Beta_pos.copy()
                    Beta_fitness = SalpFitness[i]  # Update beta
                    Beta_pos = SalpPositions[i, :].copy()
    
                if SalpFitness[i] > Alpha_fitness and SalpFitness[i] > Beta_fitness and SalpFitness[i] < Delta_fitness:
                    Delta_fitness = SalpFitness[i]  # Update delta
                    Delta_pos = SalpPositions[i, :].copy()
                
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
            
            
            
            
import numpy as np
import random
from numpy.random import rand
from evotoolbox.optimizers import BaseOptimizer

class GWO(BaseOptimizer):
    
    def __init__(self, binary_transformer, max_iter, lb, ub):
        super().__init__(binary_transformer, max_iter, lb, ub)
    
    
    def optimize(self, fitness_func, initial_positions, n_features, n_agents):
        lb = [self.lb] * n_features
        ub = [self.ub] * n_features
        
        convergence_curve = np.zeros(self.max_iter)
            
        # initialize alpha, beta, and delta_pos
        Binary_Alpha_pos = np.zeros(n_features, dtype='int')
        Alpha_pos = np.zeros(n_features)
        Alpha_fitness = float("inf")
    
        Beta_pos = np.zeros(n_features)
        Beta_fitness = float("inf")
    
        Delta_pos = np.zeros(n_features)
        Delta_fitness = float("inf")
        
        Positions = initial_positions
        BinaryPositions = np.zeros_like(Positions, dtype='int')

        for t in range(self.max_iter):
            for i in range(n_agents):
    
                # Return back the search agents that go beyond the boundaries of the search space
                for j in range(n_features):
                    Positions[i, j] = np.clip(Positions[i, j], lb[j], ub[j])
                    
                BinaryPositions[i] = self.binary_transformer(Positions[i])
    
                # Calculate objective function for each search agent
                fitness = fitness_func(BinaryPositions[i])
    
                # Update Alpha, Beta, and Delta
                if fitness < Alpha_fitness:
                    Delta_fitness = Beta_fitness  # Update delte
                    Delta_pos = Beta_pos.copy()
                    Beta_fitness = Alpha_fitness  # Update beta
                    Beta_pos = Alpha_pos.copy()
                    Alpha_fitness = fitness
                    # Update alpha
                    Alpha_pos = Positions[i, :].copy()
                    Binary_Alpha_pos = BinaryPositions[i, :].copy()
    
                if fitness > Alpha_fitness and fitness < Beta_fitness:
                    Delta_fitness = Beta_fitness  # Update delte
                    Delta_pos = Beta_pos.copy()
                    Beta_fitness = fitness  # Update beta
                    Beta_pos = Positions[i, :].copy()
    
                if fitness > Alpha_fitness and fitness > Beta_fitness and fitness < Delta_fitness:
                    Delta_fitness = fitness  # Update delta
                    Delta_pos = Positions[i, :].copy()
    
            a = 2 - t * ((2) / self.max_iter)
            # a decreases linearly fron 2 to 0
    
            # Update the Position of search agents including omegas
            for i in range(n_agents):
                for j in range(n_features):
    
                    r1 = random.random()  # r1 is a random number in [0,1]
                    r2 = random.random()  # r2 is a random number in [0,1]
    
                    A1 = 2 * a * r1 - a
                    # Equation (3.3)
                    C1 = 2 * r2
                    # Equation (3.4)
    
                    D_alpha = abs(C1 * Alpha_pos[j] - Positions[i, j])
                    # Equation (3.5)-part 1
                    X1 = Alpha_pos[j] - A1 * D_alpha
                    # Equation (3.6)-part 1
    
                    r1 = random.random()
                    r2 = random.random()
    
                    A2 = 2 * a * r1 - a
                    # Equation (3.3)
                    C2 = 2 * r2
                    # Equation (3.4)
    
                    D_beta = abs(C2 * Beta_pos[j] - Positions[i, j])
                    # Equation (3.5)-part 2
                    X2 = Beta_pos[j] - A2 * D_beta
                    # Equation (3.6)-part 2
    
                    r1 = random.random()
                    r2 = random.random()
    
                    A3 = 2 * a * r1 - a
                    # Equation (3.3)
                    C3 = 2 * r2
                    # Equation (3.4)
    
                    D_delta = abs(C3 * Delta_pos[j] - Positions[i, j])
                    # Equation (3.5)-part 3
                    X3 = Delta_pos[j] - A3 * D_delta
                    # Equation (3.5)-part 3
    
                    Positions[i, j] = (X1 + X2 + X3) / 3  # Equation (3.7)
    
            convergence_curve[t] = Alpha_fitness
    
            if t % 1 == 0:
                print(
                    ["At iteration " + str(t) + " the best fitness is " + str(Alpha_fitness)]
                )
        
        return {
            'solution': Binary_Alpha_pos,
            'c': convergence_curve,
            'nf': np.sum(Binary_Alpha_pos)
        }
            
            
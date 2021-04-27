import numpy as np
import math
import random
from numpy.random import rand
from evotoolbox.optimizers import BaseOptimizer

class SSADE(BaseOptimizer):
    
    def __init__(self, binary_transformer, max_iter, lb, ub):
        super().__init__(binary_transformer, max_iter, lb, ub)
    
    
    def optimize(self, fitness_func, initial_positions, n_features, n_agents):
        lb = [self.lb] * n_features
        ub = [self.ub] * n_features
        F = 0.5
        CR = 0.7
        Pr=0.7
        Er=0.2
        
        convergence_curve = np.zeros(self.max_iter)
        
        BinaryPositions = np.zeros((n_agents, n_features), dtype='int')
        SalpPositions = initial_positions
        SalpFitness = np.full(n_agents, float("inf"))
        v = np.zeros((n_agents, n_features))
        u = np.zeros((n_agents, n_features))
        temp = np.zeros((n_agents, n_features))

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
            
        SecondPosition = np.copy(Sorted_salps[1, :])
        BinarySecondPosition = np.copy(Sorted_salps[1, :])
        SecondFitness = Sorted_salps_fitness[1]
        
        for t in range(self.max_iter):
            c1 = 2 * math.exp(-((4 * t / self.max_iter) ** 2))
            
            for i in range(n_agents):
                for j in range(0, n_features):
                    if i < n_agents / 2:
                        c2 = random.random()
                        c3 = random.random()
                        # Eq. (3.1) in the paper
                        if c3 < 0.5:
                            temp[i,j] = FoodPosition[j] + c1 * (
                                (ub[j] - lb[j]) * c2 + lb[j]
                            )
                        else:
                            temp[i,j] = FoodPosition[j] - c1 * (
                                (ub[j] - lb[j]) * c2 + lb[j]
                            )
                        
                        v[i,j] = SalpPositions[i,j] + F*(FoodPosition[j] - SecondPosition[j])
                    
                    else:
                        point1 = SalpPositions[i - 1]
                        point2 = SalpPositions[i]

                        SalpPositions[i] = (point2 + point1) / 2
                        # Eq. (3.4) in the paper
                        
                        jk = np.random.permutation(np.arange(n_agents))
                        v[i,j] = SalpPositions[i,j] + F*(SalpPositions[jk[0],j] - SalpPositions[jk[1],j])

        
                    if v[i,j] < 0.5:
                        v[i,j] = 0
                    else:
                        v[i,j] = 1
                        
                    if random.random() < CR:
                        u[i,j] = v[i,j]
                    else:
                        u[i,j] = BinaryPositions[i,j]
                    
                temp_fitness = fitness_func(self.binary_transformer(temp[i]))
                cur_fitness = SalpFitness[i]
                u_fitness = fitness_func(u[i])
                    
                if temp_fitness < u_fitness and temp_fitness < cur_fitness:
                    SalpPositions[i, :] = temp[i].copy()
                if u_fitness < cur_fitness and u_fitness < temp_fitness:
                    SalpPositions[i, :] = u[i].copy() 
                if cur_fitness < u_fitness and cur_fitness < temp_fitness:
                    SalpPositions[i, :] = SalpPositions[i]
                           
            
            for i in range(n_agents):
                # Check if salps go out of the search spaceand bring it back
                for j in range(n_features):
                    SalpPositions[i, j] = np.clip(SalpPositions[i, j], lb[j], ub[j])
                    
                BinaryPositions[i] = self.binary_transformer(SalpPositions[i])
                
                SalpFitness[i] = fitness_func(BinaryPositions[i])
                
                if SalpFitness[i] < FoodFitness:
                    SecondPosition = FoodPosition.copy()
                    BinarySecondPosition = BinaryFoodPosition.copy()
                    SecondFitness = FoodFitness
                    
                    FoodPosition = np.copy(SalpPositions[i, :])
                    BinaryFoodPosition = np.copy(BinaryPositions[i, :])
                    FoodFitness = SalpFitness[i]
                elif SalpFitness[i] < SecondFitness:
                    SecondPosition = np.copy(SalpPositions[i, :])
                    BinarySecondPosition = np.copy(BinaryPositions[i, :])
                    SecondFitness = SalpFitness[i]
                    
            if random.random() < Er:
                I = np.argsort(SalpFitness)
                Sorted_salps_fitness = np.copy(SalpFitness[I])
                Sorted_salps = np.copy(SalpPositions[I, :])
                for i in range(n_agents//4*3, n_agents):
                    for j in range(n_features):
                        remaining_agents = n_agents//4*3 - 3
                        p = [0.25, 0.25, 0.25] + remaining_agents * [0.25/remaining_agents]
                        r = np.random.choice(np.arange(n_agents//4*3), p=p)
                        if random.random() < Pr:
                            SalpPositions[i, j] = Sorted_salps[r, j]
                        else:
                            SalpPositions[i, j] = lb[j] + random.random() * (ub[j]-lb[j])
            
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
            
            
            
            
            
            
            
            
            
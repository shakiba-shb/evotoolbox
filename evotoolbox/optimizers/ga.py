import numpy as np
from numpy.random import rand
from evotoolbox.optimizers import BaseOptimizer

class GA(BaseOptimizer):
    
    def __init__(self, binary_transformer, max_iter, lb, ub, MR = 0.01, CR = 0.8):
        super().__init__(binary_transformer, max_iter, lb, ub)
        self.MR = MR
        self.CR = CR
        
        
    def binary_conversion(self, X, thres, N, dim):
        Xbin = np.zeros([N, dim], dtype='int')
        for i in range(N):
            Xbin[i] = self.binary_transformer(X[i])
        
        return Xbin
    
    
    def roulette_wheel(self,prob):
        num = len(prob)
        C   = np.cumsum(prob)
        P   = rand()
        for i in range(num):
            if C[i] > P:
                index = i;
                break
        
        return index


    def optimize(self, fitness_func, initial_positions, n_features, n_agents):
        N        = n_agents
        max_iter = self.max_iter
        MR = self.MR
        CR = self.CR
        #thres = self.thres
        dim = n_features
        lb = self.lb
        ub = self.ub
        
        
        if np.size(lb) == 1:
            ub = ub * np.ones([1, dim], dtype='float')
            lb = lb * np.ones([1, dim], dtype='float')
        
        # Initialize position 
        X     = initial_positions

        # Fitness at first iteration
        fit   = np.zeros([N, 1], dtype='float')
        Xgb   = np.zeros([1, dim], dtype='int')
        fitG  = float('inf')
        
        for i in range(N):
            fit[i,0] = fitness_func(X[i,:])
            if fit[i,0] < fitG:
                Xgb[0,:] = X[i,:]
                fitG     = fit[i,0]     
                
        # Pre
        curve = np.zeros([1, max_iter], dtype='float')
        t     = 0
        
        curve[0,t] = fitG.copy()
        #print("Generation:", t + 1)
        #print("Best (GA):", curve[0,t])
        t += 1
        
        
        while t < max_iter:
            
            # Probability
            inv_fit = 1 / (1 + fit)
            prob    = inv_fit / np.sum(inv_fit) 
     
            # Number of crossovers
            Nc = 0
            for i in range(N):
                if rand() < CR:
                  Nc += 1
                  
            x1 = np.zeros([Nc, dim], dtype='int')
            x2 = np.zeros([Nc, dim], dtype='int')
            for i in range(Nc):
                # Parent selection
                k1      = self.roulette_wheel(prob)
                k2      = self.roulette_wheel(prob)
                P1      = X[k1,:].copy()
                P2      = X[k2,:].copy()
                # Random one dimension from 1 to dim
                index   = np.random.randint(low = 1, high = dim-1)
                # Crossover
                x1[i,:] = np.concatenate((P1[0:index] , P2[index:]))
                x2[i,:] = np.concatenate((P2[0:index] , P1[index:]))
                # Mutation
                for d in range(dim):
                    if rand() < MR:
                        x1[i,d] = 1 - x1[i,d]
                        
                    if rand() < MR:
                        x2[i,d] = 1 - x2[i,d]
    
            
            # Merge two group into one
            Xnew = np.concatenate((x1 , x2), axis=0)
            
            # Fitness
            Fnew = np.zeros([2 * Nc, 1], dtype='float')
            for i in range(2 * Nc):
                Fnew[i,0] = fitness_func(Xnew[i,:])
                if Fnew[i,0] < fitG:
                    Xgb[0,:] = Xnew[i,:]
                    fitG     = Fnew[i,0]
                       
            # Store result
            curve[0,t] = fitG.copy()
            print("Generation:", t + 1)
            print("Best (GA):", curve[0,t])
            t += 1
            
            # Elitism 
            XX  = np.concatenate((X , Xnew), axis=0)
            FF  = np.concatenate((fit , Fnew), axis=0)
            # Sort in ascending order
            ind = np.argsort(FF, axis=0)
            for i in range(N):
                X[i,:]   = XX[ind[i,0],:]
                fit[i,0] = FF[ind[i,0]]
           
                
        # Best feature subset
        Gbin       = Xgb[0,:]
        Gbin       = Gbin.reshape(dim)
        pos        = np.asarray(range(0, dim))    
        sel_index  = pos[Gbin == 1]
        num_feat   = len(sel_index)
        # Create dictionary
        ga_data = {'solution': Gbin, 'c': curve, 'nf': num_feat}
        
        return ga_data 
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

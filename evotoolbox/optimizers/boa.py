import numpy as np
from evotoolbox.optimizers import BaseOptimizer

class BOA(BaseOptimizer):

    def __init__(self, binary_transformer, max_iter, lb, ub, p=0.8, a=0.1,
                 c_min=0.01, c_max=0.25):
        super().__init__(binary_transformer, max_iter, lb, ub)
        self.p = p
        self.a = a
        self.c_min = c_min
        self.c_max = c_max
        
    
    def optimize(self, fitness_func, initial_positions, n_features, n_agents):
    
        curve = np.zeros([1, self.max_iter], dtype='float')
        c = self.c_min
        positions = initial_positions
        bfitness = np.zeros(positions.shape[0])
        for i in range(n_agents):
            bfitness[i] = fitness_func(positions[i])
        
        min_index = np.argmin(bfitness)
        fmin = bfitness[min_index]
        best_pos = positions[min_index]
        
        S = np.zeros(n_features)
        for t in range(self.max_iter):  
            for i in range(n_agents):
                Fnew=fitness_func(positions[i])
                fp=(c*(np.power(Fnew, self.a)))
      
                if np.random.random() > self.p:
                    dis = np.random.random() * np.random.random() * best_pos - positions[i]
                    S=positions[i]+dis*fp
                else:
                    epsilon=np.random.random()
                    jk=np.random.permutation(n_agents)
                    dis=epsilon*epsilon*positions[jk[0]]-positions[jk[1]]
                    S=positions[i]+dis*fp
    
                S = self.binary_transformer(S)
                
                # for j in range(N_features):
                #     S[j] = np.clip(S[j], 0, 1)
                Fnew = fitness_func(S)
    
                if (Fnew<=bfitness[i]):
                    positions[i]=S.copy()
                    bfitness[i]=Fnew
               
                if Fnew<=fmin:
                     best_pos=S.copy()
                     fmin=Fnew
                     
            c+=((self.c_max - self.c_min)/self.max_iter)
            best_i = np.argmin(bfitness)
            ss = np.sum(positions, axis=1)
            #print('it', t, 'sf', np.sum(positions[best_i]), 'fit', bfitness[best_i])
            curve[0,t] = bfitness[best_i].copy()
        # Create dictionary
        boa_data = {'solution': positions[best_i], 'c': curve, 'nf': ss}
        
        return boa_data
    
    
    
    
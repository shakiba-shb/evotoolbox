import numpy as np
from evotoolbox.initializers import BaseInitializer

class OblInitializer(BaseInitializer):
    def init(self, fitness_func):
        solution = np.zeros((self.n_agents//2, self.n_features))
        for i in range(self.n_features):
            solution[:, i] = np.random.uniform(0, 1, self.n_agents//2)
        solution[solution > 0.5] = 1
        solution[solution <= 0.5] = 0
        
        opp_solution = 1 - solution
        fit = np.zeros(self.n_agents)
        '''for i in range(N_agents):
            fit_solution = fitness(solution[i], train_x, test_x, train_y, test_y)
            fit_opp_solution = fitness(opp_solution[i], train_x, test_x, train_y, test_y)
            print("run: %i f=%f fo=%f" % (i, fit_solution, fit_opp_solution))
            if fit_opp_solution > fit_solution:
                solution[i] = opp_solution[i]'''
        sol = np.concatenate((solution, opp_solution), axis=0)
        for i in range(self.n_agents):
            fit[i] = fitness_func(sol[i])  
            #print("run: %i f=%f fo=%f" % (i, fit_solution, fit_opp_solution))
        feature_ranking = np.argsort(fit)
        n = feature_ranking.shape[0]
        solution = sol[feature_ranking]      
        return solution

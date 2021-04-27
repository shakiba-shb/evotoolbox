import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from evotoolbox.fitness import BaseFitness


class MultiobjectiveFitness(BaseFitness):
     def __init__(self, alpha, k = 5):
         self.alpha = alpha
         self.k = k
         
    
     def evaluate(self, solution):
        n_selected_features = np.sum(solution)
        n_features = solution.shape[0]
        
        if n_selected_features < 1:
            return np.inf
    
        X_SelectedFeatures = self.features[:, solution.astype(bool)]
    
        knn_classifier = KNeighborsClassifier(n_neighbors=self.k)
        cv = KFold(n_splits=5, random_state=42, shuffle=True)
            
        scores = np.zeros(self.k)
        for i, (train_index, test_index) in enumerate(cv.split(X_SelectedFeatures)):
            train_x, test_x, train_y, test_y = X_SelectedFeatures[train_index], X_SelectedFeatures[test_index], self.labels[train_index], self.labels[test_index]
            knn_classifier.fit(train_x, train_y)
            scores[i] = knn_classifier.score(test_x, test_y)
        
        accuracy = np.average(scores)
        error = 1 - accuracy
        
        fitness = self.alpha * error + (1 - self.alpha) * (n_selected_features/n_features)    
        return fitness
    
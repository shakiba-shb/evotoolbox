import pandas as pd
import evotoolbox
from evotoolbox.initializers import RandomInitializer
from evotoolbox.fitness import MultiobjectiveFitness
from evotoolbox.binary import ThresholdTransformer
from evotoolbox.optimizers import GWO

# Load your data
data = pd.read_csv('C:\\Users\\Amir\\Desktop\\.Toolbox\\Datasets\\colon.csv').to_numpy()
X = data[:,:-1]
Y = data[:, -1]
n_features = X.shape[1]

# Define the algorithm options
initializer = RandomInitializer(n_features=n_features, n_agents=10)
optimizer = GWO(ThresholdTransformer(0.5), max_iter=100, lb=0, ub=1)
fitness = MultiobjectiveFitness(alpha=0.99)

# Fit the data using the provided options
result = evotoolbox.fit(X, Y, initializer, optimizer, fitness)

# The result is a dictionary with three keys:
# solution: the binary solution with shape (n_features,) where the selected features are 1 and others are 0
# c: the convergence curve of the fitness
# nf: number of selected features in the final solution
print(result['solution'])
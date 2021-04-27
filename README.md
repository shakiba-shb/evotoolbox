# Evolutionary feature selection toolbox
This package contains a set of tool to easily apply evolutionary feature selection techniques to your datasets.
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Initializers](#initializers)
- [Fitness Functions](#fitness-functions)
- [Binary Transformers](#binary-transformers)
- [Optimizers](#optimizers)
- [Credits](#credits)


## Installation
You can install the package using the following command.
```
pip install evotoolbox
```


## Quick start
This package is based on 4 essential building blocks: _initializer_, _optimizer_, _fitness function_, and _binary transformer_. To find the best features using an evolutionary algorithm you must choose one of the provided classes within this package or you can easily define your own custom classes. 

The following sample code shows how to use the default _Random Initializer_, _GWO Optimizer_, _MultiObjective Fitness_, and _Threshold Transformer_ to do a simple feature selection task.
``` python
import pandas as pd
import evotoolbox
from evotoolbox.initializers import RandomInitializer
from evotoolbox.fitness import MultiobjectiveFitness
from evotoolbox.binary import ThresholdTransformer
from evotoolbox.optimizers import GWO

# Load your data
data = pd.read_csv('colon.csv').to_numpy()
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
```


## Initializers

### Provided Initializers
This package provides three default initializer classes. These classes can be imported from `evotoolbox.initializers`. All of the initializers must receive the number of features and the number of agents as arguments.

#### RandomInitializer
The RandomInitializer creates a random initial population.

#### GreedyInitializer
The GreedyInitializer creates a random population and the tries to enhance the solution based on a greedy algorithm. This is a two step process. In the first pass the greedy algorithm checks if the fitness increases by setting zeros to ones in the solution one by one. If the solution fitness gets better that feature will be selected. In the second pass the algorithm tries to drop the features using the same logic.
This initializer usually generates high quality solutions with small number of features. The downside is that the initialization time increases linearly with the number of features and it can take hours to complete on very large datasets.

#### OblInitializer
The RandomInitializer creates a random initial population and then also generates the complement of these solutions and compare each of these pairs and keeps the better solution in each pair and discards the other.

### Defining your own initializer
You can easily define your own initializer by extending the `evotoolbox.initializers.BaseInitializer` class.
You can access the number of features and agents in this class. You also receive a fitness function to evaluate your solutions in the initialization process.
``` python
import numpy as np
from evotoolbox.initializers import BaseInitializer

class MyCustomInitializer(BaseInitializer):
    def init(self, fitness_func):
        # create a numpy array of the initial population
        positions = np.zeros((self.n_agents, self.n_features))
        
        # do your magic here!

        # you can evaluate and compare the solutions using fitness_func
        sample_fitness = fitness_func(positions[0])
        # return the generated initial positions
        return positions
```


## Fitness Functions
Fitness functions play an important role in meta-heuristic optimization. Generally in a feature selection task, a multi objective fitness function is used to achieve a high accuracy while keeping the number of features at minimum.

### Provided fitness function

#### MultiobjectiveFitness
One of the most popular fitness functions used in feature selection is defined by this equation.
```
Fitness(selected_features) = alpha * KNN_ACCURACY(selected_features) + (1-alpha) * count(selected_features)
```
This package provides a `MultiobjectiveFitness` class which applies K-fold cross validation and calculates the accuracy of a KNN classifier and then uses the above equation to find the fitness of the given features. To use this fitness function simply insatiate this class with `alpha` and `k` (number of neighbors) parameters.
``` python
fitness = MultiobjectiveFitness(alpha=0.99, k=5)
```

### Defining your own fitness function
Of course you can implement your own fitness function. To do so, you must extend the `BaseFitness` class and implement the evaluate function. This function receives the solution to evaluate as an argument. You can access the data with `self.features` and `self.labels` variables. Here's an example.
``` python
from evotoolbox.fitness import BaseFitness

class MyFitness(BaseFitness):
     def evaluate(self, solution):
        # use self.features and self.labels to evaluate the given solution
        # return a number, corresponding to the current solution fitness
        return fitness
```


## Binary Transformers
Most of the evolutionary algorithms work in continuos space. However, for a feature selection task, we must convert these continuos values to binary values so that we can use them to choose the best features.

### Provided binary transformers
This package provides a variety of binary transformers. The transformer is given to the optimizer so that it can use it on every iteration to convert the continuos solutions into binary values. There is one threshold based and three transfer function based methods which you can use. A transfer function gives the probability of a feature being set to 1 and it is used as follows. `Z(x)` is the binary value of `x` and `T(x)` is the transfer function.

![transfer function](https://render.githubusercontent.com/render/math?math=Z%28x%29%20%3D%5Cbegin%7Bcases%7D1%20%26%20rand%28%29%20%3C%3D%20T%28x%29%5C%5C0%20%26%20otherwise%5Cend%7Bcases%7D%20)

#### ThresholdTransformer
The simplest is the `ThresholdTransformer`. This transformer uses a threshold to simply set anything above the threshold to one, and anything below the threshold to zero.
``` python
from evotoolbox.binary import ThresholdTransformer
transformer = ThresholdTransformer(0.5) # provide the threshold value here
```

#### SigmoidTransformer
_Sigmoid_ or _S_ transfer function is a popular transfer function which is defined as below.

![S transformer equation](https://render.githubusercontent.com/render/math?math=S%28x%29%20%3D%20%5Cfrac%7B1%7D%7B%281%2Be%5E%7B-%5Calpha%2Ax%7D%29%7D)

``` python
from evotoolbox.binary import SigmoidTransformer
transformer = SigmoidTransformer(1) # set alpha value here
```

#### VTransformer
There are different types of _V_ transformers. This package uses the following equation.

![V transformer equation](https://render.githubusercontent.com/render/math?math=V%28x%29%20%3D%20%7Ctanh%28x%29%7C)

``` python
from evotoolbox.binary import VTransformer
transformer = VTransformer(1) # set alpha value here
```

#### QTransformer
Quadratic transfer functions are another group of transfer functions and are defined with the following formula.

![Q transformer equation](https://render.githubusercontent.com/render/math?math=Q%28x%29%20%3D%20%5CBigl%7C%5Cfrac%7Bx%7D%7B0.5%2A%20X_%7Bmax%7D%7D%5CBigr%7C%5Ep)

``` python
from evotoolbox.binary import QTransformer
transformer = QTransformer(6, 1) # set Xmax and p
```


### Defining your own binary transformer
You might need to implement a custom binary transformer. Like other classes, you can easily extend the `BaseTransformer` class in `evotoolbox.binary` and implement the `transform` function to create your own transformer. Here's an example.
``` python
import numpy as np
from evotoolbox.binary import BaseTransformer

class VTransformer(BaseTransformer):
    def __init__(self, custom_parameter):
        # define your custom parameters here
        self.custom_parameter = custom_parameter
        
    def transform(self, solution):
        binary_solution = np.zeros_like(solution, dtype='int')
        # put your logic here 
        return binary_solution
```


## Optimizers
This package comes with a variety of optimizers to use conveniently. You are also free to define your own optimizers.

### Provided Optimizers
These optimizers are currently available in this package. Please note that this is an ongoing project and this list will be updated regularly with new algorithms.

Each of the optimizers can have their own parameters, but the first four arguments when instantiating an optimizer class are shared and required. These arguments are: _binary\_transformer_, _max\_iter_, _lb_, and _ub_. these arguments control the binary transformer used to binarize the continuos values, max number of iterations, lower bound, and upper bound, respectively.

#### Grey Wolf Optimizer (GWO)
GWO is introduced by ... for more info refer to the [relevant paper](url).
``` python
import GWO from evotoolbox.optimizers
initializer = GWO(binary_transformer, max_iter, lb, ub)
```

#### Butterfly Optimization Algorithm (BOA)
BOA is introduced by ... for more info refer to the [relevant paper](url).
``` python
import BOA from evotoolbox.optimizers
initializer = BOA(binary_transformer, max_iter, lb, ub, p=0.8, a=0.1, c_min=0.01, c_max=0.25)
```

#### Genetic Algorithm (GA)
GA is introduced by ... for more info refer to the [relevant paper](url).
``` python
import GA from evotoolbox.optimizers
initializer = GA(binary_transformer, max_iter, lb, ub, MR = 0.01, CR = 0.8)
```

#### Harris Hawk Optimizer (HHO)
HHO is introduced by ... for more info refer to the [relevant paper](url).
``` python
import HHO from evotoolbox.optimizers
initializer = HHO(binary_transformer, max_iter, lb, ub, beta = 1.5)
```

#### Salp Swarm Algorithm (SSA)
SSA is introduced by ... for more info refer to the [relevant paper](url).
``` python
import SSA from evotoolbox.optimizers
initializer = SSA(binary_transformer, max_iter, lb, ub)
```

### Defining your own optimizer
You probably want to implement your own optimizer to try out a new algorithm. Defining a new optimizer is simple, you should extend the `BaseOptimizer` class provided in `evotoolbox.optimizers`. You can define additional parameters required for your algorithm in the class constructor. All optimizers must implement the abstract `optimize(self, fitness_func, initial_positions, n_features, n_agents)` method defined in `BaseOptimizer` Take a look at this example.
``` python
import numpy as np
from evotoolbox.optimizers import BaseOptimizer

class BOA(BaseOptimizer):
    def __init__(self, binary_transformer, max_iter, lb, ub, my_parameter):
        super().__init__(binary_transformer, max_iter, lb, ub)
        self.my_parameter = my_parameter
        

    def optimize(self, fitness_func, initial_positions, n_features, n_agents):
        # Optimize the problem using the given arguments
        # You may use your custom parameter with self.my_parameter
        # initial_positions will be the population initialized before
        # Your function must return a dictionary as defined below:
        # solution: the binary solution with shape (n_features,) where the selected features are 1 and others are 0
        # c: the convergence curve of the fitness
        # nf: number of selected features in the final solution
        return {
            'solution': None,
            'c': None,
            'nf': None,
        }
```


## Credits
Authors:
- [Shakiba Shahbandegan](https://github.com/shakiba-shb)
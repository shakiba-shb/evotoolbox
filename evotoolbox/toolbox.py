def fit(features, labels, initializer, optimizer, fitness):
    fitness.setdata(features, labels)
    initial_positions = initializer.init(fitness.evaluate)
    n_features = initializer.n_features
    n_agents = initializer.n_agents
    solution = optimizer.optimize(fitness.evaluate, initial_positions , n_features, n_agents)
    return solution
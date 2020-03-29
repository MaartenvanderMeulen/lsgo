import copy
from deap import base # conda install -c conda-forge deap
from deap import cma
from deap import creator
from deap import tools
import math
import numpy as np
import random


def cmaes01(dim, f, y_target=0.0):
    '''Return x for which f(x) is minimal, with x[i] only 0 or 1. Stop early when y-target is reached.'''
    if dim < 2 or 10000 < dim:
        print("nparams value is invalid, must be in [2, 10000]")
        return None
    population_size = max(round(4 + 3 * math.log(dim) + 0.5), dim // 2) # dim/2 is result of experiments by Maarten on dim > 100
    ngen = 25 + int(0.2*dim) # result of experiments by Maarten
    print("dimension of problem space ", dim)
    print("population size            ", population_size)
    print("generations                ", ngen)    
    random.seed(42)
    np.random.seed(42)    
    strategy = cma.Strategy(centroid=[1]*dim, sigma=0.5, lambda_=population_size)
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)
    evaluation_count, best_x, best_y = 0, None, None
    for gen in range(ngen):
        population = toolbox.generate()
        for i in range(len(population)):
            for j in range(dim):
                population[i][j] = 1 if population[i][j] >= 0.5 else 0
        for x in population:
            y = f(x)
            x.fitness.values = (y,)
            evaluation_count += 1
            if best_y is None or best_y > y:
                best_x, best_y = copy.deepcopy(x), copy.deepcopy(y)
                print(f"generation {gen+1}, evaluations {evaluation_count}, best {best_y}, sigma {strategy.sigma}")
        if best_y <= y_target:
            break
        toolbox.update(population)
    return best_x, best_y


if __name__ == "__main__":    
    for dim in [20, 50, 100, 200, 500, 1000]:
        def f(x):
            error = 0.0
            for i in range(len(x)):
                if x[i] != target_x[i]:
                    error += 1.0
            return error
        target_x = [random.randint(0, 1) for i in range(dim)]
        best_x, best_y = cmaes01(dim, f)
    

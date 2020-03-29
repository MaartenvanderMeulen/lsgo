import array
import benchmarks
from deap import algorithms
from deap import base
from deap import cma
from deap import creator
from deap import tools
import numpy as np
import random


def evaluate(x):
    return (benchmarks.f(x),)


dim = benchmarks.common_dim
print("deap on f", benchmarks.common_f_id, "dim", benchmarks.common_dim)
if False: # standard ea for optimalisation
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    info = benchmarks.get_info(benchmarks.common_f_id, dim)
    toolbox.register("attr_float", random.uniform, info['lower'], info['upper'])
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, dim)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=1.0/dim)
    toolbox.register("select", tools.selTournament, tournsize=3)
    nparents, nchilds = 4000, 40
    pop = toolbox.population(n=nparents)
    algorithms.eaMuPlusLambda(pop, toolbox, mu=nparents, lambda_=nchilds, 
                              cxpb=0.5, mutpb=0.2, ngen=100000000, verbose=False) 
                              # ngen=heel veel : stopconditie wordt geregeld door benchmarks
                              # TODO : maak hier weer een loop van, die kijkt wanneer benchmarks.best_y < 0.01
if True:
    FITCLSNAME = "FIT_TYPE"
    INDCLSNAME = "IND_TYPE"

    NDIM = dim

    if False:
        strategy = cma.Strategy(centroid=[0.0]*NDIM, sigma=0.5, lambda_=24*8)
        # dim 1000, ackley : #evals = 63749; 63950; 62980; 65312
        # ngen = 338
    if False:
        strategy = cma.Strategy(centroid=[0.0]*NDIM, sigma=0.5, lambda_=24*4)
        # dim 1000, ackley : #evals = 42987; 42771; 
        # ngen = 447; 445; 
    if True:
        strategy = cma.Strategy(centroid=[0.0]*NDIM, sigma=0.5, lambda_=24*2)
        # dim 1000, ackley : #evals = 36961
        # ngen = 770,
    if False:
        strategy = cma.Strategy(centroid=[0.0]*NDIM, sigma=0.5, lambda_=24*1)
        # dim 1000, ackley : #evals = - 
        # ngen = -
    if False:
        strategy = cma.Strategy(centroid=[0.0]*NDIM, sigma=1.0)
        # lambda_ = populatie = 4 + 3 * log(dim) = 4 + 3 * log(1000) = 24
        # dim 10 : #evals =average(534;629;493) = 552
        # dim 100 : #evals =average(4076;3953;3979) = 4003
        # dim 1000 : #evals = 63467; -; 
        # ngen = 2600

    creator.create(FITCLSNAME, base.Fitness, weights=(-1.0,))
    creator.create(INDCLSNAME, list, fitness=creator.__dict__[FITCLSNAME])

    toolbox = base.Toolbox()
    toolbox.register("evaluate", evaluate)
    toolbox.register("generate", strategy.generate, creator.__dict__[INDCLSNAME])
    toolbox.register("update", strategy.update)

    if False:
        pop, _ = algorithms.eaGenerateUpdate(toolbox, ngen=100000, verbose=False)
    else:
        ngen = 100000
        try:
            for gen in range(ngen):
                population = toolbox.generate()
                if True:
                    fitnesses = toolbox.map(toolbox.evaluate, population)
                else:
                    raise RuntimeError("TODO : parallel evaluation of poppulatioin")
                for ind, fit in zip(population, fitnesses):
                    ind.fitness.values = fit
                toolbox.update(population)
        finally:
            print("final gen", gen)

print("dim", benchmarks.common_dim, ", after", benchmarks.count, "evaluations f(x) =", f"{benchmarks.best_y:.3f}")

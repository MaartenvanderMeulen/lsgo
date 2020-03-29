# https://esa.github.io/pagmo2/docs/cpp/algorithms/mbh.html#_CPPv2N5pagmo3mbhE

from pygmo import *
algo = algorithm(cmaes(gen = 500, ftol=0.1, ))
#algo = algorithm(mbh())
algo.set_verbosity(1)
prob = problem(ackley(1000))
pop = population(prob, 48)
pop = algo.evolve(pop) 

import benchmarks
# https://github.com/CMA-ES/pycma
import cma
import numpy as np


x0 = np.ones((benchmarks.common_dim)) * -1
# dim 1000 kan opgelost worden met popsize 500, en sigma 0.5
options = cma.evolution_strategy.CMAOptions()
#print(options)
opt, es = cma.fmin2(benchmarks.f, x0, 0.02, options={'maxfevals': 1e6, 'popsize':48}, bipop=False, restarts=0)

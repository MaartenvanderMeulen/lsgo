import benchmarks
import numpy as np
import random
from scipy.optimize import basinhopping, differential_evolution
# https://docs.scipy.org/doc/scipy-1.2.0/reference/optimize.html#module-scipy.optimize


dim = benchmarks.common_dim
info = benchmarks.get_info(benchmarks.common_f_id, dim)
bounds = [(info['lower'], info['upper']) for i in range(dim)]
if False:
    x0 = np.ones((dim)) * -1
    res = basinhopping(benchmarks.f, x0, minimizer_kwargs={"method": "BFGS"}, niter=26000*dim)
    print(res)
if False:
    maxiter = 1000
    popsize=500
    print("max func eval = ", maxiter*popsize*dim)
    workers=1
    res = differential_evolution(benchmarks.f, bounds, maxiter=maxiter, popsize=popsize)
    print(res)
if False:
    from scipy.optimize import shgo
    res = shgo(benchmarks.f, bounds, minimizer_kwargs={"method": "BFGS"})
if True:
    from scipy.optimize import dual_annealing
    res = dual_annealing(benchmarks.f, bounds)

import numpy as np
from pySOT import *
from poap.controller import ThreadController, BasicWorkerThread
import benchmarks


class Worker:
    def __init__(self, dim):
        func_id = benchmarks.common_f_id
        self.info = benchmarks.common_f_name
        info = benchmarks.get_info(func_id, dim)
        self.xlow = np.ones((dim)) * info['lower']
        self.xup = np.ones((dim)) * info['upper']
        self.dim = dim
        self.integer = np.arange(0, dim)
        self.continuous = []
        print(self.integer)

    def objfunction(self, x):        
        return benchmarks.f(x)


dim = benchmarks.common_dim
n_start_pts = 2*dim + 1
maxeval = 2000*dim
worker = Worker(dim=dim)
check_opt_prob(worker)
nthreads = 1 # set to higher number for parallel evaluation
nsamples = 40 
strategy = SyncStrategyNoConstraints(
    worker_id=0, data=worker, maxeval=maxeval, nsamples=nsamples,
    exp_design=SymmetricLatinHypercube(dim=dim, npts=n_start_pts),
    response_surface=RBFInterpolant(kernel=CubicKernel, tail=LinearTail, maxp=min(maxeval, 50000)),
    sampling_method=CandidateDYCORS(data=worker, numcand=min(100*dim, 5000)))
controller = ThreadController()
controller.strategy = strategy
for i in range(nthreads):
    worker_thread = BasicWorkerThread(controller, worker.objfunction)
    controller.launch_worker(worker_thread)
result = controller.run()
print("dim", benchmarks.common_dim, ", after", benchmarks.count, "evaluations f(x) =", f"{benchmarks.best_y:.3f}")

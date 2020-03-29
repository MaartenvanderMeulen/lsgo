global testing, internal_only, sleep_after_launch
testing = True
internal_only = True
sleep_after_launch = 0.5 # launch delay in seconds to prevent problems on torque
if testing:
    import benchmarks
import copy
from deap import algorithms
from deap import base
from deap import cma
from deap import creator
from deap import tools
import math
import numpy as np
import random
import subprocess
import sys, os
import time


class Worker:
    call_count = 0
    best_y = 1e15
    rescale = True # if True, scale [lo, hi] --> [0, 1]
    replace_out_of_bounds_individuals = True # if True, all x are in [lo, hi] (rescale=False) or [0, 1] (rescale=True)
    hi = 30
    lo = -30

    def set_params(dim, script, folder):
        Worker.dim = dim
        Worker.script = script
        Worker.folder = folder
        Worker.call_count = 0
        Worker.best_y = 1e15
        print("rescale      ", Worker.rescale)
        print("resample oob ", Worker.replace_out_of_bounds_individuals)
        
    def make_name(prefix):
        return f"{Worker.folder}/{prefix}{Worker.call_count:05d}.txt"

    def write_in_file(file_name, id, x):
        with open(file_name, 'w') as f:
            f.write(f"{id}")
            for xi in x:
                f.write("\t" + str(xi))
            f.write("\n*\n")

    def is_file_complete(file_name):
        if os.path.exists(file_name):
            with open(file_name, 'r') as f:
                return '*' in f.read()
        return False
    
    def wait_till_file_complete(file_name):
        t0 = time.time()
        delay = 1 # seconds
        while True:
            if Worker.is_file_complete(file_name):
                return
            if delay >= 4:
                print("waiting", delay, "seconds for", file_name, "to be complete")            
            time.sleep(delay)
            delay *= 2
            if time.time() > t0 + 7200:
                raise RuntimeError("No response for two hours from worker : stop")
            if time.time() > t0 + 3600:
                print("No response for one hour yet from worker...")

    def read_out_file(file_name, id):
        with open(file_name, 'r') as f:
            txt = f.readline().split("\t")
            if id != int(txt[0]):
                raise RuntimeError(f"file {file_name} : expected id {id} instead of {txt[0]}")
            return float(txt[1])
          
    def convert_value(out_file_value):
        # to turn minimizing 'y' into maximizing 'out'
        # worker has done : out = 1.0 / (1.0 + y)
        assert 0 < out_file_value and out_file_value <= 1 
        y = (1 - out_file_value) / out_file_value # turn it back into minimizing 'y'
        if Worker.best_y > y:
            Worker.best_y = y
        return y
        
    def objfunctionn_file(xs):
        """Evaluate len(xs) points"""
        id, out_name, in_name, y = [], [], [], []
        for i in range(len(xs)):
            Worker.call_count += 1
            id.append(Worker.call_count)
            out_name.append(Worker.make_name("out"))
            if os.path.exists(out_name[i]):
                if True: # not Worker.is_file_complete(out_name[i]):
                    os.remove(out_name[i])
            in_name.append(Worker.make_name("in"))
            Worker.write_in_file(in_name[i], Worker.call_count, xs[i])
        for i in range(len(xs)):
            if True: # not Worker.is_file_complete(out_name[i]):
                subprocess.call([Worker.script, in_name[i], out_name[i]])
                if sleep_after_launch > 0:
                    time.sleep(sleep_after_launch) # delay in seconds to prevent problems with the torque cluster 
        for i in range(len(xs)):
            Worker.wait_till_file_complete(out_name[i])
            y.append(Worker.convert_value(Worker.read_out_file(out_name[i], id[i])))
        return y
            
    if testing:
        def objfunctionn_internal(xs):
            """Evaluate len(xs) points"""
            if Worker.rescale:
                #print("xs_internal", (np.array(xs[0])-0.5)*(Worker.hi - Worker.lo))
                y = [benchmarks.f((np.array(xs[ind])-0.5)*(Worker.hi - Worker.lo)) for ind in range(len(xs))]
            else:
                y = [benchmarks.f(np.array(xs[ind])) for ind in range(len(xs))]
            Worker.best_y = min(Worker.best_y, np.min(y))
            Worker.call_count += len(xs)
            return y
            
    def objfunctionn(xs):
        xs_copy = copy.deepcopy(xs)
        if Worker.rescale:
            for ind in range(len(xs_copy)):
                xs_copy[ind] = np.array(xs_copy[ind])/(Worker.hi - Worker.lo)+0.5
        if internal_only:
            return Worker.objfunctionn_internal(xs_copy)
        else:
            y_file = Worker.objfunctionn_file(xs_copy)
            if testing:
                y_internal = Worker.objfunctionn_internal(xs_copy)
                for ind in range(len(xs_copy)):
                    if abs(y_file[ind] - y_internal[ind]) > 0.01:
                        print("ERROR, difference internal and file : ind", ind, "y_file", y_file[ind], "y_internal", y_internal[ind])
                        print(xs_copy[ind])
                        exit(0)
            return y_file
            
    def generate_resamples(toolbox, resamples):
        """used by Worker.generate(), to get the resamples for out of bound individuals"""
        tmp = toolbox.generate()
        for row in range(len(tmp)):
            for dim in range(Worker.dim):
                if Worker.lo <= tmp[row][dim] and tmp[row][dim] <= Worker.hi:
                    resamples[dim].append(tmp[row][dim])
        for dim in range(Worker.dim):
            if len(resamples[dim]) <= 1:
                return generate_resamples(toolbox, resamples)
        return resamples
        
    def generate(toolbox):
        """as toolbox.generate(), but all is within [Worker.lo,Worker.hi] bounds"""
        population = toolbox.generate()
        if Worker.replace_out_of_bounds_individuals:
            resamples = Worker.generate_resamples(toolbox, [[] for d in range(Worker.dim)])
            count_out_of_bound = 0
            for i in range(len(population)):
                for dim in range(Worker.dim):
                    if population[i][dim] < Worker.lo or Worker.hi < population[i][dim]:
                        population[i][dim] = random.choice(resamples[dim])
                        count_out_of_bound += 1
            if count_out_of_bound:
                print(count_out_of_bound, "out of bounds individuals replaced")
        return population

            
def main(dim, nworkers, script, folder, ngen):
    print("nparams      ", dim)
    print("nworkers     ", nworkers)
    print("script       ", script)
    print("folder       ", folder)
    print("ngenerations ", ngen)
    if dim < 2 or 10000 < dim:
        print("nparams value is invalid, must be in [2, 10000]")
        return
    if nworkers < 1 or 200 < nworkers :
        print("nworkers value is invalid, must be in [1, 200]")
        return
    if not os.path.exists(script):
        print("script does not exist")
        return
    if not os.path.exists(folder):        
        print("folder does not exist")
        return
    if ngen <= 1:
        print("ngen value is invalid, must be >= ")
        print("Hint : the old pySOT 'incl-centre' setting is replaced by ngen setting.")
        return
    Worker.set_params(dim, script, folder)
    random.seed(42)
    np.random.seed(42)
        
    lambda_ = int(4 + 3 * math.log(dim))
    if dim > 100:
        lambda_ *= 2 # for dim == 1000, 4 + 3 * log(dim) is not enough
    lambda_ = max(lambda_, nworkers)
    print("population   ", lambda_, "(is evaluated in parallel)")
    strategy = cma.Strategy(centroid=[0.5]*dim, sigma=0.5, lambda_=lambda_)

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)
    for gen in range(ngen):
        population = Worker.generate(toolbox)
        fitnesses = Worker.objfunctionn(population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = (fit,)
        print(f"generations {gen+1} evaluations {Worker.call_count} best {Worker.best_y} sigma {strategy.sigma}")
        if Worker.best_y < 0.1:
            break
        toolbox.update(population)
    

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print('Usage:')
        print('  %s <nparams> <ncores> <script> <outfolder> <ngen>' % (sys.argv[0]))
        print('Comments')
        print('  Uses CMA-ES optimizer (see wikipedia)')
        print('  <nparams> The number of parameters to calibrate.')
        print('  <ncores> The number of cores to use = parallel evaluation.')
        print('  <script> The script that launches a gdxworker in the background.')
        print('  <outfolder> were to put the intermediate in/out files.')
        print('  <ngen> number of generations.')
        sys.exit(2)
        
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5]))

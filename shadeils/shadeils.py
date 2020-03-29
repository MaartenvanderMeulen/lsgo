#!/mnt/home/daniel/working/shadeils/venv/bin/python
## !/usr/bin/python
#     pylint: disable=E1101
import argparse
import sys

from os import path, remove, system

from . import DEcrossover
from .DE_shadeils import EAresult

from scipy.optimize import fmin_l_bfgs_b

from numpy.random import seed, permutation, uniform, randint

import numpy as np

from . import SHADE

from os.path import isfile

import random
from .mts import mtsls


"""
This class allow us to have a pool of operation. When we ask the Pool one of
them, the selected operator is decided following the last element whose improvement
ratio was better. The idea is to apply more times the operator with a better
improvement.
"""
class PoolLast:
    def __init__(self, options):
        """
        Constructor
        :param options:to store (initially the probability is equals)
        :return:
        """
        size = len(options)
        assert size > 0

        self.options = np.copy(options)
        self.improvements = []
        self.count_calls = 0
        self.first = permutation(self.options).tolist()

        self.new = None
        self.improvements = dict(zip(options, [0] * size))

    def reset(self):
        self.first = permutation(self.options).tolist()
        self.new = None
        options = self.options
        size = len(options)
        self.improvements = dict(zip(options, [0] * size))

    def has_no_improvement(self):
        return np.all([value == 0 for value in self.improvements.values()])

    def get_new(self):
        """
        Get one of the options, following the probabilities
        :return: one of the stored object
        """
        # First time it returns all
        if self.first:
            return self.first.pop()

        if self.new is None:
            self.new = self.update_prob()

        return self.new

    def is_empty(self):
        counts = self.improvements.values()
        return np.all(counts == 0)

    def improvement(self, obj, account, freq_update, minimum=0.15):
        """
        Received how much improvement this object has obtained (higher is better), it only update
        the method improvements

        :param object:
        :param account: improvement obtained (higher is better), must be >= 0
        :param freq_update: Frequency of improvements used to update the ranking
        :return: None
        """
        if account < 0:
            return

        if obj not in self.improvements:
            raise Exception("Error, object not found in PoolProb")

        previous = self.improvements[obj]
        self.improvements[obj] = account
        self.count_calls += 1

        if self.first:
            return

        if not self.new:
            self.new = self.update_prob()
        elif account == 0 or account < previous:
            self.new = self.update_prob()

    def update_prob(self):
        """
        update the probabilities considering improvements value, following the equation
        prob[i] = Improvements[i]/TotalImprovements

        :return: None
        """

        if np.all([value == 0 for value in self.improvements.values()]):
            import ipdb; ipdb.set_trace()
            new_method = np.random.choice(self.improvements.keys())
            # print("new_method: {}".format(new_method))
            return new_method

        # Complete the ranking
        indexes = np.argsort(self.improvements.values())
        posbest = indexes[-1]
        best = list(self.improvements.keys())[posbest]
        return best

def get_improvement(alg_name, before, after):
    """
    Print the improvement with an algorithm
    """
    if before == 0:
        ratio = 0
    else:
        ratio = (before-after)/before

    return "{0}: {1:.3e} -> {2:.3e} [{3:.2e}, {4:.2f}]\n".format(alg_name, before, after, before-after, ratio)

SR_global_MTS = []
SR_MTS = []

def apply_localsearch(name, method, fitness_fun1, bounds, current_best, current_best_fitness, maxevals, fid):
    global SR_MTS
    global SR_global_MTS

    lower = bounds[0][0]
    upper = bounds[0][1]

    if method == 'grad':
        # print("Voor fmin_l_bfgs_b")
        sol, fit, info = fmin_l_bfgs_b(fitness_fun1, x0=current_best, approx_grad=1, bounds=bounds, maxfun=maxevals, disp=False)
        # print("Na fmin_l_bfgs_b")
        funcalls = info['funcalls']
    elif method == 'mts':
#        import ipdb
#        ipdb.set_trace()
        if name.lower() == "global":
            SR = SR_global_MTS
        else:
            SR = SR_MTS

        # print("Voor mtsls")
        res, SR_MTS = mtsls(fitness_fun1, current_best, current_best_fitness, lower, upper, maxevals, SR)
        # print("Na mtsls")
        sol = res.solution
        fit = res.fitness
        funcalls = maxevals
    else:
        raise NotImplementedError(method)

    if fit <= current_best_fitness:
        if fid: fid.write(get_improvement("{0} {1}".format(method.upper(), name), current_best_fitness, fit))
        return EAresult(solution=np.array(sol), fitness=fit, evaluations=funcalls)
    else:
        return EAresult(solution=current_best, fitness=current_best_fitness, evaluations=funcalls)

def random_population(lower, upper, dimension, size):
    return uniform(lower, upper, dimension*size).reshape((size, dimension))

def applySHADE(crossover, fitness, funinfo, dimension, evals, population, populationFitness, bestId, current_best, fid, H=None):
#    import ipdb; ipdb.set_trace()
    if current_best.fitness < populationFitness[bestId]:
        population[bestId,:] = current_best.solution
        populationFitness[bestId] = current_best.fitness

    if H is None:
       H = population.shape[0]

    result, bestId = SHADE.improve(run_info=funinfo, replace=False, dimension=dimension, name_output=None,
            population=population, H=H, population_fitness=populationFitness, fun=fitness, check_evals=evals, initial_solution=current_best.solution, MemF=applySHADE.MemF, MemCR=applySHADE.MemCR)
    if fid: fid.write(get_improvement("SHADE partial", current_best.fitness, result.fitness))
    return result, bestId


optimo = True

def check_evals(totalevals, evals, bestFitness, globalBestFitness, fid):
    if not evals:
        return evals
    elif totalevals >= evals[0]:
        best = min(bestFitness, globalBestFitness)
        if fid: fid.write("[%.1e]: %e,%d\n" %(evals[0], best, totalevals))
        if fid: fid.flush()
        evals.pop(0)

    return evals

def reset_ls(dim, lower, upper, method='all'):
    global SR_global_MTS
    global SR_MTS

    if method == 'all' or method == 'mts':
        SR_global_MTS = np.ones(dim)*(upper-lower)*0.2
        SR_MTS = SR_global_MTS

def reset_de(popsize, dimension, lower, upper, H, current_best_solution=None):
    population = random_population(lower, upper, dimension, popsize)

    if current_best_solution is not None:
        posrand = randint(popsize)
        population[posrand] = current_best_solution

    applySHADE.MemF = 0.5*np.ones(H)
    applySHADE.MemCR = 0.5*np.ones(H)
    return population


def set_region_ls():
    global SR_global_MTS
    global SR_MTS

    SR_MTS = np.copy(SR_global_MTS)

def get_ratio_improvement(previous_fitness, new_fitness):
    if previous_fitness == 0:
        improvement = 0
    else:
        improvement = (previous_fitness-new_fitness)/previous_fitness

    return improvement


def ihshadels(fitness_fun1, fitness_funn, call_count, funinfo, dim, evals, fid, info_de, popsize=100, debug=False, threshold=0.05):
    """
    Implementation of the proposal for CEC2015
    """
    global SR_MTS, SR_global_MTS
    SR_MTS = []
    SR_global_MTS = []

    lower = funinfo['lower']
    upper = funinfo['upper']
    evals = evals[:]

    initial_sol = np.ones(dim)*((lower+upper)/2.0)
    if debug:
        print("Voor fitness_fun1(initial_sol), line 254")
    current_best_fitness = fitness_fun1(initial_sol)
    if debug:
        print("Na fitness_fun1(initial_sol), line 254")

    maxevals = evals[-1]
    totalevals = 1

    bounds = list(zip(np.ones(dim)*lower, np.ones(dim)*upper))
    bounds_partial = list(zip(np.ones(dim)*lower, np.ones(dim)*upper))

    popsize = min(popsize, 100)
    population = reset_de(popsize, dim, lower, upper, info_de)
    if debug:
        print("Voor fitness_funn line 260, call_count", call_count)
    populationFitness = fitness_funn(population)
    if debug:
        print("Na fitness_funn line 260, call_count", call_count)
    bestId = np.argmin(populationFitness)

    initial_sol = np.ones(dim)*(lower+upper)/2.0
    if debug:
        print("Voor fitness_fun1(initial_sol), line 272")
    initial_fitness = fitness_fun1(initial_sol)
    if debug:
        print("Na fitness_fun1(initial_sol), line 274")

    if initial_fitness < populationFitness[bestId]:
        if fid: fid.write("Best initial_sol\n")
        population[bestId] = initial_sol
        populationFitness[bestId] = initial_fitness

    current_best = EAresult(solution=population[bestId,:], fitness=populationFitness[bestId], evaluations=totalevals)

    crossover = DEcrossover.SADECrossover(2)
    best_global_solution = current_best.solution
    best_global_fitness = current_best.fitness
    current_best_solution = best_global_solution

    apply_de = apply_ls = True
    applyDE = applySHADE

    reset_ls(dim, lower, upper)
    methods = ['mts', 'grad']

    pool_global = PoolLast(methods)
    pool = PoolLast(methods)

    num_worse = 0

    evals_gs = min(50*dim, 25000)
    evals_de = min(50*dim, 25000)
    evals_ls = min(10*dim, 5000)
    num_restarts = 0

    while totalevals < maxevals:
        method = ""

        if not pool_global.is_empty():
            previous_fitness = current_best.fitness
            method_global = pool_global.get_new()
            if debug:
                print("voor apply_localsearch, line 310")
            current_best = apply_localsearch("Global", method_global, fitness_fun1, bounds, current_best_solution, current_best.fitness, evals_gs, fid)
            totalevals += current_best.evaluations
            if debug:
                print("na apply_localsearch #evals", current_best.evaluations, "total now", totalevals)
            improvement = get_ratio_improvement(previous_fitness, current_best.fitness)

            pool_global.improvement(method_global, improvement, 2)
            evals = check_evals(totalevals, evals, current_best.fitness, best_global_fitness, fid)
            current_best_solution = current_best.solution
            current_best_fitness = current_best.fitness

            if current_best_fitness < best_global_fitness:
                best_global_solution = np.copy(current_best_solution)
                if debug:
                    print("Voor 323")
                best_global_fitness = fitness_fun1(best_global_solution)
                if debug:
                    print("Na 323")

        for i in range(1):
            current_best = EAresult(solution=current_best_solution, fitness=current_best_fitness, evaluations=0)
            set_region_ls()

            method = pool.get_new()

            if apply_de:
                if debug:
                    print("voor applyDE")
                result, bestInd = applyDE(crossover, fitness_fun1, funinfo, dim, evals_de, population, populationFitness, bestId, current_best, fid, info_de)
                if debug:
                    print("na applyDE")
                improvement = current_best.fitness - result.fitness
                totalevals += result.evaluations
                evals = check_evals(totalevals, evals, result.fitness, best_global_fitness, fid)
                current_best = result

            if apply_ls:
                if debug:
                    print("voor apply_localsearch, line 343")
                result = apply_localsearch("Local", method, fitness_fun1, bounds_partial, current_best.solution, current_best.fitness, evals_ls, fid)
                if debug:
                    print("na apply_localsearch")
                improvement = get_ratio_improvement(current_best.fitness, result.fitness)
                totalevals += result.evaluations
                evals = check_evals(totalevals, evals, result.fitness, best_global_fitness, fid)
                current_best = result

                pool.improvement(method, improvement, 10, .25)

            current_best_solution = current_best.solution
            current_best_fitness = current_best.fitness

            if current_best_fitness < best_global_fitness:
                 best_global_fitness = current_best_fitness
                 best_global_solution = np.copy(current_best_solution)

            # Restart if it is not improved
            if (previous_fitness == 0):
                ratio_improvement = 1
            else:
                ratio_improvement = (previous_fitness-result.fitness)/previous_fitness

            if fid: fid.write("TotalImprovement[{:d}%] {:.3e} => {:.3e} ({})\tRestart: {}\n".format(
                int(100*ratio_improvement), previous_fitness, result.fitness,
                num_worse, num_restarts))

            if ratio_improvement >= threshold:
                num_worse = 0
            else:
                num_worse += 1
                imp_str = ",".join(["{}:{}".format(m, val) for m, val in pool.improvements.items()])
                if fid: fid.write("Pools Improvements: {}".format(imp_str))

                # Random the LS
                reset_ls(dim, lower, upper, method)

            if num_worse >= 3:
                num_worse = 0
#                import ipdb; ipdb.set_trace()
                if fid: fid.write("Restart:{0:.2e} for {1:.2f}: with {2:d} evaluations\n".format(current_best.fitness, ratio_improvement, totalevals))
                # Increase a 1% of values
                posi =  np.random.choice(popsize)
                new_solution = np.random.uniform(-0.01, 0.01, dim)*(upper-lower)+population[posi]
                new_solution = np.clip(new_solution, lower, upper)
                # print("voor EAresult")
                current_best = EAresult(solution=new_solution, fitness=fitness_fun1(new_solution), evaluations=0)
                # print("na EAresult")
                current_best_solution = current_best.solution
                current_best_fitness = current_best.fitness

                # Init DE
                population = reset_de(popsize, dim, lower, upper, info_de)
                # print("Voor fitness_funn line 378, call_count", call_count)
                populationFitness = fitness_funn(population)
                # print("Na fitness_funn line 378, call_count", call_count)
                totalevals += popsize

                totalevals += popsize
                # Random the LS
                pool_global.reset()
                pool.reset()
                reset_ls(dim, lower, upper)
                num_restarts += 1

            if fid: fid.write("{0:.2e}({1:.2e}): with {2:d} evaluations\n".format(current_best_fitness, best_global_fitness, totalevals))
#            fid.write("improvement_group[{}] : {:.2e}\n".format(i, (initial_fitness - result.fitness)))
            if fid: fid.flush()

            if totalevals >= maxevals:
                break

    if fid: fid.write("%e,%s,%d\n" %(abs(best_global_fitness), ' '.join(map(str, best_global_solution)), totalevals))
    if fid: fid.flush()
    return result

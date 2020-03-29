import copy
from deap import base # conda install -c conda-forge deap
from deap import cma
from deap import creator
from deap import tools
import math
import numpy as np
import random


def cmaes(dim, f, y_target=0.0):
    '''Return x for which f(x) is minimal. Stop early when y-target is reached.'''
    if dim < 2 or 10000 < dim:
        print("nparams value is invalid, must be in [2, 10000]")
        return None
    population_size = max(math.ceil(4 + 3 * math.log(dim) + 0.5), dim // 2) # dim/2 is result of experiments by Maarten on dim > 100
    population_size *= 2
    ngen = 600 # 25 + int(0.2*dim) # result of experiments by Maarten
    print("dimension of problem space ", dim)
    print("population size            ", population_size)
    print("generations                ", ngen)    
    #random.seed(42)
    #np.random.seed(42)    
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    nhops = 10
    for hop in range(nhops):
        centroid = [random.randint(-3, 3) for i in range(dim)]
        strategy = cma.Strategy(centroid=centroid, sigma=0.5, lambda_=population_size)
        toolbox = base.Toolbox()
        toolbox.register("generate", strategy.generate, creator.Individual)
        toolbox.register("update", strategy.update)
        evaluation_count, best_x, best_y = 0, [1]*dim, None
        try:
            for gen in range(ngen):
                population = toolbox.generate()
                if False:
                    for i in range(len(population)):
                        for j in range(dim):
                            population[i][j] = round(population[i][j]*1000) / 1000
                for x in population:
                    y = f(x)
                    x.fitness.values = (y,)
                    evaluation_count += 1
                    if best_y is None or best_y > y:
                        best_x, best_y = copy.deepcopy(x), copy.deepcopy(y)
                if best_y <= y_target:
                    break
                toolbox.update(population)
            x_str = ", ".join([f"{xi:.3f}" if xi != round(xi) else f"{int(round(xi))}" for xi in best_x])
            print(f"evaluations {evaluation_count} evaluations f({x_str}) = {best_y:.3f}")
        except:
            pass
    return best_x, best_y


if __name__ == "__main__":    
    f14_points = (
        (4, 98, 19, -7191),
        (51, 22, 8, -8857),
        (95, 80, 96, -729057),
        (84, 34, 57, -162469),
        (58, 61, 81, -286155),
        (2, 88, 18, -2936),
        (68, 31, 24, -50390),
        (9, 46, 78, -31957),
        (63, 48, 8, -24009),
        (72, 2, 55, -7679),
        (92, 48, 33, -145441),
        (15, 33, 87, -42723),
        (8, 55, 51, -22169),
        (41, 8, 25, -8068),
        (5, 98, 19, -9052),
        (65, 66, 85, -364198),
        (3, 18, 93, -4704),
        (21, 44, 31, -28442),
        (38, 12, 71, -32101),
    )
    def f14(x):
        '''The 'easy' puzzle of Victor x0*ABC+x1*AB+x2*AC+...+x7, waarbij x tussen -9 en 9'''
        global f14_points
        sum_se = 0.0        
        for A, B, C, y in f14_points:
            result = x[0]*A*B*C + x[1]*A*B + x[2]*A*C + x[3]*B*C + x[4]*A + x[5]*B + x[6]*C + x[7]
            # result = -1*A*B*C + x[0]*A*B + x[1]*A*C + x[2]*B*C + x[3]*A + x[4]*B + x[5]*C + x[6]
            if not math.isfinite(result):
                # print(result, x)
                return 1e9
            assert math.isfinite(result)
            sum_se += (y - result) ** 2
        rmse = math.sqrt(sum_se / len(f14_points))
        return rmse
    best_x, best_y = cmaes(8, f14, 0.1)
    # print(best_x, best_y)
    

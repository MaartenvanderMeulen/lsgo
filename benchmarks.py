from math import sqrt, exp, pi, sin, cos
import numpy as np
import random
import time
import math
import copy

# some constants shared by all optimalization experiments
global common_dim, common_f_id
common_dim = 8
common_f_id = 14
common_f_name = "VictorEasy" # "Ackley"

def get_info(fun, dim):
    """
    Return the bounds of the function
    """
    bounds = [100, 10, 100, 100, 30, 100, 1.28, 500, 5.12, 32, 600, 50, 50, 9]
    objective_value = 0

    if fun == 8:
        objective_value = -12569.5 * dim / 30.0

    return {
        'lower': -bounds[fun - 1],
        'upper': bounds[fun - 1],
        'threshold': objective_value,
        'best': 1e-8
    }


def get_function(fun):
    """
    Evaluate the solution
    @param fun function value (1-13)
    @param x solution to evaluate
    @return the obtained fitness
    """
    functions = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14]
    return functions[fun - 1]


def get_max_function():
    return 13

    
global count, best_y, x_offset, last_t
count = 0
best_y = None
best_x = None
last_t = time.time()


def init_call_f():
    """call this once before each run of call_f's """
    global count, best_y, x_offset, last_t
    count = 0
    best_y = None
    best_x = None
    last_t = time.time()
    

def progress():
    x_str = ", ".join([f"{xi:.3f}" if xi != round(xi) else f"{int(round(xi))}" for xi in best_x])
    print(f"{count} evaluations f({x_str}) = {best_y:.3f}")

def f(x):
    """Call f and do some extra's:
    -- move x away from 0 to make easy guesses by solvers fail
    -- count the number of call's
    -- show count when f(x) < 0.1    
    """
    global count, best_y, best_x, last_t, common_f_id
    count += 1
    x = np.array(x)
    y = get_function(common_f_id)(x)
    if best_y is None or best_y > y:
        best_y = y
        best_x = copy.deepcopy(x)
    if last_t < time.time() - 10:
        last_t = time.time()
        progress()                
    return y


def f1(x):
    """
    function 1: Sphere
    @param x solution
    """
    return (x**2).sum()


def f2(x):
    """
    function 2: Schwefel's Problem 2.22
    @param x solution
    """
    f = np.abs(x).sum()
    g = np.abs(x).prod()
    return f + g


def f3(x):
    """
    function 3: Schwefel's Problem 1.2
    @param x solution
    """
    #    f = 0
    #    size = np.size(x)
    #
    #    for i in xrange(size):
    #       f = f + np.sum(x[:i+1])**2
    f = np.power(np.add.accumulate(x), 2).sum()
    return f


def f4(x):
    """
    function 4: Schwefel problem 2.21
    @param x solution
    """
    return np.abs(x).max()

def f5(x):    
    """
    function 5: Generalized Rosenbrock's Function
    @param x solution
    """
    return sum(100 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)
    
    

def f6(x):
    """
    function 6: Step Function
    @param x solution
    """
    return np.power(np.floor(x + 0.5), 2).sum()


def f7(x):
    """
    function 7: Quartic Function with Noise
    @param x solution
    """
    dim = np.size(x)
    expr = np.arange(1, dim + 1) * np.power(x, 4)
    return expr.sum() + np.random.uniform(0, 1)


def f8(x):
    """
    function 8: Generalized Schwefel's Problem 2.26
    @param x solution
    """
    return -(x * np.sin(np.sqrt(np.abs(x)))).sum()


def f9(x):
    """
    function 9: Generalized Rastrigin's Function
    @param x solution
    """
    expr = x**2 - 10 * np.cos(2 * pi * x) + 10
    return expr.sum()


def f10(x):
    """
    function 10: Ackley's Function
    @param x solution
    """
    a = (x**2).sum()
    b = np.cos(2 * pi * x).sum()
    dim = np.size(x)
    f = -20 * exp(-0.2 * sqrt(1.0 / dim * a)) - exp(
        (1.0 / dim) * b) + 20 + exp(1)
    return f


def f11(x):
    """
    function 11: Generalized Griewank Function
    @param x solution
    """
    a = (x**2).sum()
    dim = np.size(x)
    b = np.cos(x / np.sqrt(np.arange(1, dim + 1))).prod()
    return 1.0 / 4000 * a - b + 1


def u(x, a, k, m):
    if x > a:
        result = k * pow(x - a, m)
    elif -a <= x <= a:
        result = 0
    elif x < -a:
        result = k * pow(-x - a, m)
    else:
        print("Error")

    return result


def f12(x):
    """
    function 12: Penalized Functions f12
    @param x solution
    """
    dim = np.size(x)
    y = 1 + (0.25 * (x + 1))
    f = 0
    g = 0

    for i in range(dim - 1):
        f = f + (pow(y[i] - 1, 2) * (1 + 10 * pow(sin(pi * y[i + 1]), 2)))

    f = f + (10 * pow(sin(pi * y[0]), 2)) + pow(y[dim - 1] - 1, 2)
    f = f * pi / 30

    for i in range(dim):
        g += u(x[i], 5., 100., 4)

    return f + g


def f13(x):
    """
    function 13: Penalized Functions f13
    @param x solution
    """
    dim = np.size(x)
    f = 0
    g = 0

    for i in range(dim - 1):
        f = f + (pow(x[i] - 1, 2) * (1 + pow(sin(3 * pi * x[i + 1]), 2)))

    f = f + pow(sin(3 * pi * x[0]),
                2) + (pow(x[dim - 1] - 1, 2) *
                      (1 + sin(pow(2 * pi * x[dim - 1], 2))))
    f = f * 0.1

    for i in xrange(dim):
        xi = x[i]

        if xi > 5:
            g = g + 100 * (xi - 5)**4
        elif xi <= 5 and xi >= -5:
            g = g + 0
        elif xi < -5:
            g = g + pow(100 * ((-1) * xi - 5), 4)

    return f + g


global f14_points
f14_points_oud = (
    (4, 98, 19, -7224),
    (51, 22, 8, -8593),
    (95, 80, 96, -729174),
    (84, 34, 57, -162397),
    (58, 61, 81, -286344),
    (2, 88, 18, -2918),
    (68, 31, 24, -50228),
    (9, 46, 78, -32167),
    (63, 48, 8, -23868),
    (72, 2, 55, -7745),
    (92, 48, 33, -145279),
    (15, 33, 87, -42831),
    (8, 55, 51, -22247),
    (41, 8, 25, -8086),
    (5, 98, 19, -8854),
    (65, 66, 85, -364174),
    (3, 18, 93, -4890),
    (21, 44, 31, -28322),
    (38, 12, 71, -32314),)
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
        # result = -1*A*B*C + 0*A*B + 0*A*C + 0*B*C + x[0]*A + x[1]*B + x[2]*C + x[3]
        # result = -1*A*B*C + x[0]*A*B + x[1]*A*C + x[2]*B*C + x[3]*A + x[4]*B + x[5]*C + x[6]
        sum_se += (y - result) ** 2
    rmse = math.sqrt(sum_se / len(f14_points))
    return rmse
    
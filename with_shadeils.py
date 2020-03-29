import benchmarks
from shadeils.shadeils import ihshadels


def fitness_funn(population):
    result = []
    for x in population:
        result.append(benchmarks.f(x))
    return result
    

dim = benchmarks.common_dim
funinfo = benchmarks.get_info(benchmarks.common_f_id, dim)
evals = [26000*dim]
print("shadeils on f", benchmarks.common_f_id, "dim", dim)
with open("c:/tmp/shadeils.txt", "w") as fd:
    nhops = 10
    for hop in range(nhops):
        call_count = 1
        benchmarks.init_call_f()
        print("hop", hop)
        ihshadels(benchmarks.f, fitness_funn, call_count, funinfo, dim, evals, fd, threshold=0.001, popsize=300, info_de=100)
        benchmarks.progress()

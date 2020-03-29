import benchmarks
import numpy as np
import sys


with open(sys.argv[1], 'r') as f:
    x = f.readline().split("\t")
id = int(x[0])
x = np.array(x[1:], dtype='float')
if True:
    #print("xs_external", (x-0.5)*60)
    y = benchmarks.f((x-0.5)*60)
else:
    y = benchmarks.f(x)
out = 1.0 / (1.0 + y) # mAXimize 'out' == mINimize 'y'
with open(sys.argv[2], 'w') as f:
    f.write(f"{id}\t{out}\t{y}\n*\n")

import numpy as np
import random
from shadeils.shadeils import ihshadels
import subprocess
import sys, os
import time


class Worker:
    call_count = 0
    best_y = 1e15

    def set_params(dim, nworkers, script, folder, maxeval):
        Worker.dim = dim
        Worker.nworkers = nworkers
        Worker.script = script
        Worker.folder = folder
        Worker.maxeval = maxeval
        Worker.call_count = 0
        Worker.best_y = 1e15
        
    def get_info():
        return {'lower': 0.0, 'upper': 1.0, 'threshold': 0.0, 'best': 1e-8}
        
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
        delay = 4 # seconds
        while True:
            if Worker.is_file_complete(file_name):
                return
            print("waiting for", file_name, "to be complete")            
            time.sleep(delay)
            delay *= 2
            if delay > 3600:
                print("No response for one hour yet from worker")
            if delay > 7200:
                raise RuntimeError("No response for two hours from worker : stop")


    def read_out_file(file_name):
        with open(file_name, 'r') as f:
            return float(f.readline().split("\t")[1])
          
    def convert_value(out_file_value):
        # to turn minimizing 'y' into maximizing 'out'
        # worker has done : out = 1.0 / (1.0 + y)
        assert 0 < out_file_value and out_file_value <= 1 
        y = (1 - out_file_value) / out_file_value # turn it back into minimizing 'y'
        print("shadeils_calibrator : the", Worker.call_count, "th evaluation f(x) =", f"{y:.3f}")
        if Worker.best_y > y:
            Worker.best_y = y
            print("shadeils_calibrator : after", Worker.call_count, "evaluations f(best_x) =", f"{y:.3f} (improvement)")
        return y
            
    def objfunction_nworkers(xs):
        """Evaluate len(xs) points, but at most Worker.nworkers, in parallel"""
        #if Worker.call_count > Worker.maxeval:
        #    print("max eval exceeded")
        #    exit(0) # we are finished
        if len(xs) > Worker.nworkers:
            raise RuntimeError(f"call_count {Worker.call_count} : xs parameter of objfunction_nworkers to large")
        out_name, in_name, y = [], [], []
        for i in range(len(xs)):
            #print("A, call_count = ", Worker.call_count, "i", i)
            Worker.call_count += 1
            #print("B, call_count = ", Worker.call_count)
            out_name.append(Worker.make_name("out"))
            if os.path.exists(out_name[i]):
                # always remove until random.seed(seed) is checked to be reproducable
                os.remove(out_name[i])
            in_name.append(Worker.make_name("in"))
            Worker.write_in_file(in_name[i], Worker.call_count, xs[i])
        for i in range(len(xs)):
            if not Worker.is_file_complete(out_name[i]):
                subprocess.call([Worker.script, in_name[i], out_name[i]])
        for i in range(len(xs)):
            #print("D, call_count = ", Worker.call_count, "i", i)
            Worker.wait_till_file_complete(out_name[i])
            y.append(Worker.convert_value(Worker.read_out_file(out_name[i])))
            #print("E, call_count = ", Worker.call_count, "i", i, "out_name", out_name[i], "y", y[-1])
        #print("F, call_count = ", Worker.call_count, "y", y)
        return y
            
    def objfunctionn(xs):
        """Evaluate len(xs) points, in batches of Worker.nworkers parallel evaluations"""
        print("shadeils_calibrator : call to objfunctionn", len(xs))
        err = []
        for i in range(0, len(xs), Worker.nworkers):
            err.extend(Worker.objfunction_nworkers(xs[i:i+Worker.nworkers]))
        return err

            
    def objfunction1(x):
        """Evaluate x"""
        err = Worker.objfunction_nworkers([x])[0]
        return err

            
def run_gdx_calibrator(dim, nworkers, script, folder, maxeval):
    print("nparams   ", dim)
    print("nworkers  ", nworkers)
    print("script    ", script)
    print("folder    ", folder)
    print("maxeval   ", maxeval)
    if dim < 2:
        print("nparams value is invalid")
        return
    if nworkers < 1 or 200 < nworkers :
        print("nworkers value is invalid")
        return
    if not os.path.exists(script):
        print("script does not exist")
        return
    if not os.path.exists(folder):        
        print("folder does not exist")
        return
    if maxeval <= 1:
        print("maxeval value makes no sense")
        print("Hint : the old pySOT 'incl-centre' setting is replaced by the SHADE-ILS maxeval setting.")
        return
    Worker.set_params(dim, nworkers, script, folder, maxeval)
    random.seed(42)
    np.random.seed(42)
    ihshadels(Worker.objfunction1, Worker.objfunctionn, Worker.call_count, Worker.get_info(), dim, [maxeval], None, \
        threshold=0.1, popsize=100, info_de=100)
    

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print('Usage:')
        print('  %s <nparams> <ncores> <script> <outfolder> <maxeval>' % (sys.argv[0]))
        print('Comments')
        print('  Uses SHADE-ILS optimizer instead of pySOT.')
        print('       github.com/dmolina/shadeils')
        print('  Runs are restartable, and continue were they were killed.')
        print('  <nparams> The number of parameters to calibrate.')
        print('  <ncores> The number of cores to use.')
        print('  <script> The script that launches a gdxworker in the background.')
        print('  <outfolder> were to put the intermediate in/out files.')
        print('  <maxeval> maximum number of evaluations.')
        sys.exit(2)
        
    run_gdx_calibrator(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5]))

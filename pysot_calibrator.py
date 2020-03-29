import numpy as np
from pySOT import *
from poap.controller import ThreadController, BasicWorkerThread
import sys, os

class gdxworker:
    def __init__(self, dim, worker_id, script, folder):
        self.xlow = np.zeros(dim)
        self.xup = np.ones(dim)
        self.dim = dim
        self.folder = folder
        self.info = "gdxworker /NI" + str(dim) + " /NO1"
        self.integer = []
        self.continuous = np.arange(0, dim)
        self.worker_id = worker_id
        self.script = script
        self.call_count = 0

    def make_name(self, prefix):
        try:
            return os.path.join(self.folder, ("%s%04d%03d.txt"
                % (prefix, self.call_count, self.worker_id))) 
        except:
            print("worker", self.worker_id, "call", self.call_count,"COULD NOT CREATE FILE NAME FROM PREFIX", prefix)
            raise


    def write_in_file(self, file_name, x):
        try:
            assert len(x) == self.dim
            with open(file_name, 'w') as f:
                f.write("1")
                for xi in x:
                    f.write("\t" + str(xi))
                f.write("\n*\n")
        except:
            print("worker", self.worker_id, "call", self.call_count,"COULD NOT WRITE",file_name)
            raise


    def launch(self, command):
        try:
            if os.system(command) != 0:
                raise Exception(command + " failed")
        except:
            print("worker", self.worker_id, "call", self.call_count,"COULD NOT EXECUTE",command)
            raise


    def wait_till_file_complete(self, file_name):
        delay = 4 # seconds
        while True:
            try:
                with open(file_name, 'r') as f:
                    if '*' in f.read():
                        return
            except:
                pass
            print("worker", self.worker_id, "call", self.call_count,"waiting for", file_name, "to be complete")            
            time.Sleep(delay)
            delay *= 2
            if delay > 3600:
                print("No response from worker " + str(self.worker_id) + " yet")
            if delay > 7200:
                raise Exception("No response from worker " + str(self.worker_id))


    def read_out_file(self, file_name):
        try:
            with open(file_name, 'r') as f:
                return float(f.readline().split("\t")[1])
        except:
            print("worker", self.worker_id, "call", self.call_count,"COULD NOT READ", file_name)
            raise

            
    def convert_value(self, out_file_value):
        try:
            assert 0 < out_file_value and out_file_value <= 1 
            err = (1 - out_file_value) / out_file_value
            return err
        except:
            print("worker", self.worker_id, "call", self.call_count,"COULD NOT CONVERT", out_file_value, "TO ERROR VALUE")
            raise


    def objfunction(self, x):
        try:
            self.call_count += 1
            out_name = self.make_name("out")
            if os.path.exists(out_name):
                os.remove(out_name)
            in_name = self.make_name("in")
            self.write_in_file(in_name, x)
            self.launch(self.script + " " + in_name + " " + out_name)
            self.wait_till_file_complete(out_name)
            err = self.convert_value(self.read_out_file(out_name))
            #print("worker", self.worker_id, "call", self.call_count,"error", err)
            return err
        except:
            print("worker", self.worker_id, "call", self.call_count,"SOMETHING WENT WRONG!")
            raise

def make_multiple_of_n(i, n):
    return int((i + n - 1) // n) * n


class GdxSymmetricLatinHypercube(SymmetricLatinHypercube): # includes centre (all 0.5)
    def contains_centre(self, xsample):
        for i in range(len(xsample)):
            count = 0
            for j in range(self.dim):
                if xsample[i,j] == 0.5:
                    count += 1
            if count == self.dim:
                return True
        return False


    def generate_points(self):
        xsample = SymmetricLatinHypercube.generate_points(self)
        if not self.contains_centre(xsample):
            xsample[0] = np.ones((self.dim))/2
        return xsample


def run_gdx_calibrator(dim, nworkers, script, folder, inclCentre):
    n_start_pts = make_multiple_of_n(2*dim+1, nworkers)
    maxeval = n_start_pts + make_multiple_of_n(min(max(300, 100 * dim),400), 2*nworkers)
    print("nparams   ", dim)
    print("nworkers  ", nworkers)
    print("script    ", script)
    print("folder    ", folder)
    print("inclCentr ", inclCentre)
    print(n_start_pts, "start point evaluations")
    print(maxeval, "total evaluations")
    worker = [None] * nworkers
    for i in range(0, nworkers):
        worker[i] = gdxworker(dim=dim,worker_id=(i+1),script=script,folder=folder)
    controller = ThreadController()
    if inclCentre == 1:
        exp_design=GdxSymmetricLatinHypercube(dim=dim, npts=n_start_pts)
    else:
        exp_design=SymmetricLatinHypercube(dim=dim, npts=n_start_pts)
    sampling_methods = [
        CandidateUniform(data=worker[0], numcand=min(100*dim,2000)),
        CandidateDYCORS(data=worker[0], numcand=min(100*dim,2000))]
    strategy = SyncStrategyNoConstraints(
        worker_id=0, data=worker[0], maxeval=maxeval, nsamples=nworkers,
        exp_design=exp_design,
        response_surface=RBFInterpolant(kernel=CubicKernel, tail=LinearTail, maxp=min(maxeval,2000)),
        sampling_method=MultiSampling(sampling_methods, cycle=[0,1]))
    controller.strategy = strategy
    for i in range(0, nworkers):
        worker_thread = BasicWorkerThread(controller, worker[i].objfunction)
        controller.launch_worker(worker_thread)
    return controller.run()


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print('Usage:')
        print('  %s <nparams> <ncores> <script> <outfolder> 0' % (sys.argv[0]))
        print('Comments')
        print('  <nparams> The number of parameters to calibrate.')
        print('  <ncores> The number of cores to use.')
        print('  <script> The script that launches the gdxworker.')
        print('  <outfolder> were to put the intermediate in/out files.')
        print('  <inclcentre> 1 : include centre in experiment design.')
        sys.exit(2)
    try:
        solution = run_gdx_calibrator(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5]))
        print("solution ", solution.params[0], solution.value)
    except:
        raise
    sys.exit(0)

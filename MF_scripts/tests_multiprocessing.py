from multiprocessing import Process
from MF_Numpy import MF_Numpy
from exemples import gen_matrix
import numpy as np
import time

# testing the multiprocessing module. 
# Currently does not work adequately (just duplicates the work)
# need to work on concurrency and shared memory 

def train_parallel(target):
    # Create four processes
    processes = []
    for _ in range(4):
        process = Process(target=target)
        processes.append(process)

    # Start the processes
    for process in processes:
        process.start()

    # Wait for the processes to finish
    for process in processes:
        process.join()

if __name__ == '__main__':

    R = gen_matrix(shape=(500,500), correlation=0.3, sparsity=0.8)
    K = int(np.round(np.log(max(R.shape))))
    mf = MF_Numpy(R, K=K, alpha=0.01, beta=0.05, iterations=50)

    print("\n\nTest parallel :\n\n")
    start = time.time()

    train_parallel(mf.train)

    stop = time.time()
    print("\nTemps de fonctionnement :", stop - start)
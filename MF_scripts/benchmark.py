import time
import numpy as np
import matplotlib.pyplot as plt
from MF_Numpy import MF_Numpy, MF_python 
from MF_Cython import MF_Cython
from exemples import gen_matrix


def benchmark_train(classes, size_range):
    results = {}

    print("starting computations...")

    for class_name in classes:
        class_results = {}
        for size in size_range:

            matrix = gen_matrix((size, size), correlation=0.3, sparsity=0.8)
            K = int(np.round(np.log(max(matrix.shape))))

            class_instance = class_name(matrix, K=K, alpha=0.01, beta=0.05, iterations=50)

            start_time = time.time()
            class_instance.train(print_errors=False)
            end_time = time.time()

            execution_time = end_time - start_time
            class_results[size] = execution_time

        results[class_name.__name__] = class_results

        print(f"{class_name.__name__} done")

    return results


classes = [MF_Numpy, MF_python, MF_Cython]
size_range = range(10,101,10) 

execution_times_low = benchmark_train(classes, size_range)

for class_name, times in execution_times_low.items():
    size = times.keys()
    exec_time = times.values()
    plt.plot(size, exec_time, label=class_name)

plt.xlabel('Matrix Size')
plt.ylabel('Execution Time')
plt.xticks(size_range)
plt.legend()
plt.subplots_adjust(top=0.98)
plt.savefig('low_size_benchmark.png')
plt.close()

# with larger value, logscale

# size_range_high = np.logspace(1, 3, base=10, dtype=int)
size_range_high = np.logspace(np.log(10), np.log(1001), num=5, base=np.e, dtype=int)

execution_times_high = benchmark_train(classes, size_range_high)

for class_name, times in execution_times_high.items():
    size = times.keys()
    exec_time = times.values()
    plt.plot(size, exec_time, label=class_name)

plt.xlabel('Matrix Size (nat log scale)')
plt.ylabel('Execution Time (nat log scale)')
plt.xticks(size_range_high)
plt.legend()
plt.yscale('log')
plt.subplots_adjust(top=0.98)
plt.savefig('high_size_benchmark.png')



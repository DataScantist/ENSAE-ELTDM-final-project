import time
import numpy as np
from MF_Numpy import MF_Numpy, MF_python, MF_python_float32
from MF_Cython import MF_Cython, MF_Cython_float32


def gen_matrix(shape, correlation, sparsity):

    """
    Generate a matrix with specified shape, column correlation, and sparsity.

    Parameters:
        shape (tuple): The shape of the matrix in the form (num_rows, num_cols).
        correlation (float): The correlation coefficient among columns
        sparsity (float): The proportion of zero values in the matrix

    Returns:
        numpy.ndarray: The generated matrix with the specified shape, 
                       column correlation, and sparsity.
    """  

    num_rows, num_cols = shape

    # Generate a random covariance matrix based on the correlation
    cov_matrix = np.full((num_cols, num_cols), correlation)
    np.fill_diagonal(cov_matrix, 1)

    # Generate random values from a multivariate normal distribution
    values = np.random.multivariate_normal(mean=np.zeros(num_cols),
                                           cov=cov_matrix, size=num_rows)

    # Scale the non-zero values to the desired range (0 to 5)
    values = (values - values.min()) / (values.max() - values.min()) * 5

    # Introduce additional zero values randomly
    random_zeros = np.random.choice([0, 1],
                                    size=(num_rows, num_cols),
                                    p=[sparsity, 1-sparsity])
    values = values * random_zeros

    # Round the values to the nearest integer
    values = np.round(values).astype(int)

    return values


if __name__ == "__main__":

    shape = (100, 100)
    correlation=0.3
    sparsity=0.5

    R = gen_matrix(shape=shape, correlation=correlation, sparsity=sparsity)
    K = int(np.round(np.log(max(R.shape))))



    print("\n\nTest Python simple :\n\n")
    start = time.time()

    mf = MF_python(R, K=K, alpha=0.01, beta=0.05, iterations=50)
    training_process = mf.train()

    print()
    print(f"K (latent space) of dimension {K}")
    if R.shape[0] < 11 and R.shape[1] < 11:
        print("Initial matrix of ratings:")
        print(R)
        print()
        print("P x Q, rate prediction, line = users:")
        print(mf.full_matrix())
        print()
        print("User bias:")
        print(mf.b_u)
        print()
        print("Item bias:")
        print(mf.b_i)
    print()
    stop = time.time()

    print("\nTemps de fonctionnement :", stop - start)




    print("\n\nTest Numpy :\n\n")
    start = time.time()

    mf_numpy = MF_Numpy(R, K=K, alpha=0.01, beta=0.05, iterations=50)
    training_process = mf_numpy.train()

    print()
    print(f"K (latent space) of dimension {K}")
    if R.shape[0] < 11 and R.shape[1] < 11:
        print("Initial matrix of ratings:")
        print(R)
        print()
        print("P x Q, rate prediction, line = users:")
        print(mf_numpy.full_matrix())
        print()
        print("User bias:")
        print(mf_numpy.b_u)
        print()
        print("Item bias:")
        print(mf_numpy.b_i)
    print()

    stop = time.time()
    print("\nTemps de fonctionnement :", stop - start)




    print("\n\nTest Cython :\n\n")
    start = time.time()

    mf_cython = MF_Cython(R, K=K, alpha=0.01, beta=0.05, iterations=50)
    training_process = mf_cython.train()

    print()
    print(f"K (latent space) of dimension {K}")
    if R.shape[0] < 11 and R.shape[1] < 11:
        print("Initial matrix of ratings:")
        print(R)
        print()
        print("P x Q, rate prediction, line = users:")
        print(mf_cython.full_matrix())
        print()
        print("User bias:")
        print(mf_cython.b_u)
        print()
        print("Item bias:")
        print(mf_cython.b_i)
    print()

    stop = time.time()
    print("\nTemps de fonctionnement :", stop - start)
    
    pass




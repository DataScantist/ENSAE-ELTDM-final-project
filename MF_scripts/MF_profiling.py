import numpy as np
from MF_Numpy import MF_Numpy, MF_python
from MF_Cython import MF_Cython, MF_Cython_variant

# It is necessary to add the decorator @profile on top of the function we want to
# profile, within the MF_cython and MF_numpy.py scripts.


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


# Profiling training time
if __name__ == "__main__":
    shape= (100, 100)
    correlation=0.3
    sparsity=0.5

    R = gen_matrix(shape=shape, correlation=correlation, sparsity=sparsity)
    K = int(np.round(np.log(max(R.shape))))

    mf_python = MF_python(R, K=K, alpha=0.01, beta=0.05, iterations=50)
    mf_numpy = MF_Numpy(R, K=K, alpha=0.01, beta=0.05, iterations=50)
    mf_cython = MF_Cython(R, K=K, alpha=0.01, beta=0.05, iterations=50)
    mf_cython_variant = MF_Cython_variant(R, K=K, alpha=0.01, beta=0.05, iterations=50)

    mf_python.train(print_errors=False)
    mf_numpy.train(print_errors=False)
    mf_cython.train(print_errors=False)
    mf_cython_variant.train(print_errors=False)


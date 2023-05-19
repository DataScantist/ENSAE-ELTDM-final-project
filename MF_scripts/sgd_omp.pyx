cimport cython
from cython.parallel cimport prange

@cython.boundscheck(False)
@cython.wraparound(False)
def sgd_parallel(double[:, ::1] P, 
                double[:, ::1] Q,
                double[::1] b_u, 
                double[::1] b_i,
                double[:, ::1] samples,
                double alpha, 
                double beta, 
                double b):

    cdef int i, j
    cdef double r, prediction, e
    cdef int num_samples = samples.shape[0]
    

    # use prange to do parallel SGD
    for k in prange(num_samples, nogil=True, schedule=static):
        i = <int>samples[k, 0]
        j = <int>samples[k, 1]
        r = samples[k, 2]

        # Computer prediction and error
        prediction = b + b_u[i] + b_i[j]

        for d in range(Q.shape[1]):
            prediction += P[i, d] * Q[j, d]
        e = (r - prediction)

        # Update biases
        b_u[i] += alpha * (e - beta * b_u[i])
        b_i[j] += alpha * (e - beta * b_i[j])
        
        # Update user and item latent feature matrices
        for d in range(Q.shape[1]):
            P[i, d] += alpha * (e * Q[j, d] - beta * P[i, d])
            Q[j, d] += alpha * (e * P[i, d] - beta * Q[j, d])
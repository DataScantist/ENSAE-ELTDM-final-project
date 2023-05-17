import numpy as np
cimport numpy as np

def sgd(np.ndarray[np.float32_t, ndim=2] R, 
        np.ndarray[np.float32_t, ndim=2] P, 
        np.ndarray[np.float32_t, ndim=2] Q,
        np.ndarray[np.float32_t] b_u, 
        np.ndarray[np.float32_t] b_i,
        int K,
        double alpha, 
        double beta, 
        double b, 
        list samples):

    cdef int i, j
    cdef double r, prediction, e
    cdef int num_samples = len(samples)

    for k in range(num_samples):
        i, j, r = samples[k]

        # Computer prediction and error
        prediction = b + b_u[i] + b_i[j]

        for d in range(K):
            prediction += P[i, d] * Q[j, d]
        e = (r - prediction)

        # Update biases
        b_u[i] += alpha * (e - beta * b_u[i])
        b_i[j] += alpha * (e - beta * b_i[j])
        
        # Update user and item latent feature matrices
        for d in range(K):
            P[i, d] += alpha * (e * Q[j, d] - beta * P[i, d])
            Q[j, d] += alpha * (e * P[i, d] - beta * Q[j, d])
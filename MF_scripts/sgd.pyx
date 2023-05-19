import numpy as np
cimport numpy as np

def sgd_float64(np.ndarray[np.float64_t, ndim=2] P, 
                np.ndarray[np.float64_t, ndim=2] Q,
                np.ndarray[np.float64_t, ndim=1] b_u, 
                np.ndarray[np.float64_t, ndim=1] b_i,
                list samples,
                double alpha, 
                double beta, 
                double b):

    cdef int i, j
    cdef double r, prediction, e
    cdef int num_samples = len(samples)

    for k in range(num_samples):
        i, j, r = samples[k]

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





def sgd_variant(np.ndarray[np.float64_t, ndim=2] P, 
              np.ndarray[np.float64_t, ndim=2] Q,
              np.ndarray[np.float64_t, ndim=1] b_u, 
              np.ndarray[np.float64_t, ndim=1] b_i,
              list samples,
              double alpha, 
              double beta, 
              double b):

    cdef int i, j
    cdef double r, prediction, e
    cdef int num_samples = len(samples)

    for k in range(num_samples):
        i = samples[k][0]
        j = samples[k][1]
        r = samples[k][2]

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





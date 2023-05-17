#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 17:02:23 2023

@author: Mamee
"""
import cython
import numpy as np
cimport numpy as np

def sgd(np.ndarray[np.float64_t, ndim=2] P,
        np.ndarray[np.float64_t, ndim=2] Q,
        np.ndarray[np.float64_t, ndim=1] b_u,
        np.ndarray[np.float64_t, ndim=1] b_i,
        np.ndarray[np.float64_t, ndim=2] R,
        np.ndarray[np.float64_t, ndim=1] samples,
        double alpha, double beta):

    cdef double e, prediction
    cdef int i, j, r

    for k in range(samples.shape[0]):
        i, j, r = samples[k]

        # Computer prediction and error
        prediction = 0
        for l in range(Q.shape[1]):
            prediction += P[i, l] * Q[j, l]
        prediction += b_u[i] + b_i[j]
        prediction += np.mean(R[np.where(R != 0)])

        e = r - prediction

        # Update biases
        b_u[i] += alpha * (e - beta * b_u[i])
        b_i[j] += alpha * (e - beta * b_i[j])

        # Update user and item latent feature matrices
        for l in range(Q.shape[1]):
            P[i, l] += alpha * (e * Q[j, l] - beta * P[i, l])
            Q[j, l] += alpha * (e * P[i, l] - beta * Q[j, l])



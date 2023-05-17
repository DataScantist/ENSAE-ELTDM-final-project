#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 17:55:44 2023

@author: Mamee
"""

from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy
from Cython.Distutils import build_ext

extensions = [
    Extension("sgd", ["MF_tests/sgd_2.pyx"], include_dirs=[numpy.get_include(),
                                                         './MF_tests'])
]

setup(
    ext_modules=cythonize(extensions),
    cmdclass={'build_ext': build_ext},
    script_args=["build_ext", "--build-lib", "./MF_tests"]
)
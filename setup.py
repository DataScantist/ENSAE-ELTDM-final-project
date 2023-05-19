from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy
from Cython.Distutils import build_ext

extensions = [
    Extension(
        "sgd",
        ["MF_scripts/sgd.pyx"],
        include_dirs=[numpy.get_include(),'./MF_scripts']
        )
]

# uncomment the following to (attempt) compiling parallelized version with OpenMP

# extensions = [
#     Extension(
#         "sgd",
#         ["MF_scripts/sgd.pyx"],
#         include_dirs=[numpy.get_include(),'./MF_scripts']
#         ),
#     Extension(
#         "sgd_omp",
#         ["MF_scripts/sgd_omp.pyx"],
#         include_dirs=[numpy.get_include(),'./MF_scripts'],
#         extra_compile_args=['-fopenmp'],
#         extra_link_args=['-fopenmp'],
#         )
# ]

setup(
    ext_modules=cythonize(extensions),
    cmdclass={'build_ext': build_ext},
    script_args=["build_ext", "--build-lib", "./MF_scripts"]
)
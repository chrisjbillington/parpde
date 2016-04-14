# Run this setup script like so:
# python setup.py build_ext --inplace

# To produce html annotation for a cython file, instead run:
# cython -a myfile.pyx

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("_finite_differences", ["_finite_differences.pyx"])]
setup(
    name = "_finite_differences",
    cmdclass = {"build_ext": build_ext},
    ext_modules = ext_modules
)

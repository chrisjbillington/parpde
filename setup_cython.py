# Run this setup script like so:
# python setup.py build_ext --inplace

# To produce html annotation for a cython file, instead run:
# cython -a myfile.pyx

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("_cython_functions", ["_cython_functions.pyx"])]
setup(
    name = "_cython_functions",
    cmdclass = {"build_ext": build_ext},
    ext_modules = ext_modules
)

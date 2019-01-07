# To just build the Cython extension for testing in the development folder, run:
# python setup.py build_ext --inplace

# To upload a version to PyPI, run:
#
#    python setup.py sdist upload
#
# If the package is not registered with PyPI yet, do so with:
#
# python setup.py register

import sys
import os

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext


VERSION = '1.0.1'

# Auto generate a __version__ package for the package to import
with open(os.path.join('parPDE', '__version__.py'), 'w') as f:
    f.write("__version__ = '%s'\n" % VERSION)

dependencies = ['numpy', 'scipy', 'h5py', 'cython', 'mpi4py']

if sys.version_info.major == 2:
    dependencies.append('enum34')

extra_args = ["-Ofast", "-march=native"]

ext_modules = [
    Extension(
        "parPDE.finite_differences",
        [os.path.join('parPDE', 'finite_differences.pyx')],
        extra_compile_args=extra_args,
    )
]

setup(
    name='parPDE',
    version=VERSION,
    description="A parallel solver for partial differential equations using MPI",
    author='Chris Billington',
    author_email='chrisjbillington@gmail.com',
    url='https://bitbucket.org/cbillington/parPDE/',
    license="BSD",
    packages=['parPDE'],
    install_requires=dependencies,
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules,
)

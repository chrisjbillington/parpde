import os
import shutil

this_folder = os.path.dirname(os.path.abspath(__file__))
extension_name = os.path.join(this_folder, '_cython_functions.so')
if not os.path.exists(extension_name):
    # Compile the Cython extension. Only one MPI process should do this!
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
    if not rank:
        current_folder = os.getcwd()
        try:
            os.chdir(this_folder)
            os.system("python setup_cython.py build_ext --inplace")
            shutil.rmtree('build')
            os.unlink('_cython_functions.c')
        finally:
            os.chdir(current_folder)
    MPI.COMM_WORLD.Barrier()

from _cython_functions import laplacian_edges, laplacian_interior, SOR_step_interior, SOR_step_edges

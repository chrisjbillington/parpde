import os
import shutil
from mpi4py import MPI


this_folder = os.path.dirname(os.path.abspath(__file__))
extension_name = '_finite_differences'
extension_pyx = os.path.join(this_folder, extension_name + '.pyx')
extension_so = os.path.join(this_folder, extension_name + '.so')
extension_c = os.path.join(this_folder, extension_name + '.c')

if not os.path.exists(extension_so) or os.path.getmtime(extension_so) < os.path.getmtime(extension_pyx):
    # Compile the Cython extension. Only one MPI process should do this!
    rank = MPI.COMM_WORLD.Get_rank()
    if not rank:
        current_folder = os.getcwd()
        try:
            os.chdir(this_folder)
            if os.system("python setup.py build_ext --inplace") != 0:
                raise ImportError("Couldn't compile cython extension")
            shutil.rmtree('build')
            os.unlink(extension_c)
        finally:
            os.chdir(current_folder)

MPI.COMM_WORLD.Barrier()

# from _finite_differences import laplacian_edges, laplacian_interior, SOR_step_interior, SOR_step_edges
from _finite_differences import SOR_step, apply_operator

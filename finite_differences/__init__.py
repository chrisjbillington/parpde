import sys
import os
import shutil
from distutils.sysconfig import get_config_var
from mpi4py import MPI


this_folder = os.path.dirname(os.path.abspath(__file__))
extension_name = '_finite_differences'
extension_pyx = os.path.join(this_folder, extension_name + '.pyx')
extension_c = os.path.join(this_folder, extension_name + '.c')

extension_so = os.path.join(this_folder, extension_name)
ext_suffix = get_config_var('EXT_SUFFIX')
if ext_suffix is not None:
    extension_so += ext_suffix
else:
    extension_so += '.so'


def compile_extension():
    # Compile the Cython extension. Only one MPI process should do this!
    rank = MPI.COMM_WORLD.Get_rank()
    if not rank:
        current_folder = os.getcwd()
        try:
            os.chdir(this_folder)
            if os.system(sys.executable + " setup.py build_ext --inplace") != 0:
                raise ImportError("Couldn't compile cython extension")
            shutil.rmtree('build')
            os.unlink(extension_c)
        finally:
            os.chdir(current_folder)

if not os.path.exists(extension_so) or os.path.getmtime(extension_so) < os.path.getmtime(extension_pyx):
    compile_extension()

MPI.COMM_WORLD.Barrier()

try:
    from ._finite_differences import SOR_step, apply_operator
except ImportError:
    # Compiled for a different Python version? Different Cython version? Whatever, recompile:
    try:
        os.unlink(extension_c)
    except OSError:
        pass
    compile_extension()
    from ._finite_differences import SOR_step, apply_operator




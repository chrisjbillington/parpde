import h5py
import sys
sys.path.insert(0, '../..') # The location of the modules we need to import

from parPDE import HDFOutput
for i, psi in HDFOutput.iterframes('groundstate_8', frames=[-1]):
        with h5py.File('initial_8.h5', 'w') as g:
            g.create_dataset('psi', data=psi)

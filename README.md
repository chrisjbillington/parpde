Parallel partial differential equation solver. Solves PDEs in parallel using MPI. Spatial derivatives are computed with simple finite differences and time evolution performed with forward difference methods.

Work in progress, currently only has the Laplacian operator implemented and can simulate only a single field.

Can evolve fields in time with fourth order Runge-Kutta, or solve systems of equations with successive overrelaxation.

Currently contains an example of a Bose-Einstein condensate. The example code finds the groundstate with successive overrelaxation before printing some vortices into the condensate and evolving it in time.

Try running the example with "mpirun -n <number of cores> python run_example.py" and then plotting it with "python plot_example.py".

Requires python, h5py, cython, numpy, matplotlib, mpi4py and an MPI implementation.
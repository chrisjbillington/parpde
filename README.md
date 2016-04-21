Parallel partial differential equation solver. Solves PDEs in parallel using
MPI. Spatial derivatives are computed with simple finite differences and time
evolution performed with forward difference methods. Spatial derivatives can
also be computed with Fourier transforms, but this limits execution to a
single process.

Currently only in 2D with first derivatives and second derivative operators
available to 2nd, 4th or 6th order. Cross derivatives (d^2/dxdy) are not
implemented.

Can evolve fields in time with fourth order Runge-Kutta, or solve systems of
equations with successive overrelaxation. Also has two Runge-Kutta variants
implemented and the split-step method popular for simulating Bose-Einstein
condensates (though this method requires Fourier transforms and is hence
limited to a single process)

Example of a Bose-Einstein condensate included. The example code finds the
groundstate with successive overrelaxation before printing some vortices into
the condensate and evolving it in time.

Try running the example with "mpirun -n <number of cores> python
run_example.py" and then plotting it with "python plot_example.py".

Works on Python 2 and 3 (tested on 2.7 and 3.4).
Requires python, numpy, scipy, h5py, cython, mpi4py an MPI implementation and
matplotlib (for the plot example). If using Python < 3.4, also requires the
backported 'enum34' module.

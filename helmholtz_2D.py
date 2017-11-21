from __future__ import division, print_function
import numpy as np
from .parPDE import HDFOutput, format_float, LAPLACIAN


class Helmholtz2D(object):
    def __init__(self, simulator, use_ffts=False):
        self.simulator = simulator
        self.dx = simulator.dx
        self.dy = simulator.dy
        self.use_ffts = use_ffts

    def compute_mu_convergence(self, t, psi, mu_target):
        """Computes the RMS discrepancy from the target eigenvalue"""
        grad2psi = self.simulator.par_operator(LAPLACIAN, psi, use_ffts=self.use_ffts)
        residuals = grad2psi - mu_target*psi
        sum_squared_residuals = self.simulator.par_vdot(residuals, residuals).real
        denominator = mu_target*psi
        sum_squared_denominator =  self.simulator.par_vdot(denominator, denominator).real
        convergence = np.sqrt(sum_squared_residuals/sum_squared_denominator)
        return convergence

    def compute_number(self, psi):
        ncalc = self.simulator.par_vdot(psi, psi).real * self.dx * self.dy
        return ncalc

    def normalise(self, psi, N_2D):
        """Normalise psi to the 2D normalisation constant N_2D, which has
        units of a linear density. Modifies psi in-place and returns pre-
        update normalisation."""
        # imposing normalisation on the wavefunction:
        ncalc = self.compute_number(psi)
        psi[:] *= np.sqrt(N_2D/ncalc)
        return ncalc

    def solve(self, mu, psi, relaxation_parameter=1.7, convergence=1e-13,
              output_interval=100, output_directory=None, convergence_check_interval=10,
              boundary_mask=None):
        """Find the solution to the Helmholtz equation with k^2 = mu using
        sucessive overrelaxation. Requires mu, and an initial guess of the
        wavefunction. The relaxation parameter must be between 0 and 2, and
        generally somewhere between 1.5 and 2 gives fastest convergence The
        more points per MPI task, the higher the relaxation parameter can be.
        The calculation will stop when mu is correct to within the given
        convergence. To save time, this will only be computed every
        convergence_check_interval steps. If output_directory is None, the
        output callback will still be called every output_interval steps, but
        it will just print statistics and not output anything to file.
        output_interval can also be a list of integers for which steps output
        should be saved."""

        if not self.simulator.MPI_rank: # Only rank 0 should print
            print('\n==========')
            print('Beginning successive overrelaxation')
            print("Target mu is: " + repr(mu))
            print("Convergence criterion is: {}".format(convergence))
            print('==========')

        def groundstate_system(psi):
            """The system of equations Ax = b to be solved with sucessive
            overrelaxation to find the groundstate. For us this is H*psi = mu*psi.
            Here we compute b, the diagonal part of A, and the coefficients for
            representing the nondiagonal part of A as a sum of operators to be
            evaluated by the solver."""
            return np.ones(psi.shape), LAPLACIAN, mu*psi

        def output_callback(i, t, psi, infodict):
            time_per_step = infodict['time per step']
            convergence_calc = infodict['convergence']

            output_log_dtype = [('step_number', int), ('convergence', float), ('time_per_step', float)]

            output_log_data = np.array((i, convergence_calc, time_per_step), dtype=output_log_dtype)

            if output_directory is not None:
                hdf_output.save(psi, output_log_data)
            message =  ('step: %d'%i +
                        ' | convergence: %E'%convergence_calc +
                        ' | time per step: {}'.format(format_float(time_per_step, units='s')))
            if not self.simulator.MPI_rank: # Only rank 0 should print
                print(message)

        if output_directory is not None:
            hdf_output = HDFOutput(self.simulator, output_directory)

        self.simulator.successive_overrelaxation(groundstate_system, psi, boundary_mask, relaxation_parameter,
                                                 convergence, output_interval, output_callback, post_step_callback=None,
                                                 convergence_check_interval=convergence_check_interval)
        if not self.simulator.MPI_rank: # Only rank 0 should print
            print('Convergence reached')
        return psi
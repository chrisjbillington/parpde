from __future__ import division, print_function
import time
import numpy as np
from parPDE import rk4, rk4ilip, successive_overrelaxation, HDFOutput, format_float


class BEC2D(object):
    def __init__(self, simulator, natural_units=False):
        self.simulator = simulator
        self.natural_units = natural_units
        self.dx = simulator.dx
        self.dy = simulator.dy

        if natural_units:
            self.hbar = 1
            self.time_units = 'time units'
        else:
            self.hbar = 1.054571726e-34
            self.time_units = 's'

    def compute_mu(self, t, psi, H):
        """Computes approximate chemical potential (for use as a global energy
        offset, for example) corresponding a given Hamiltonian, where H is a
        function that returns the result of that Hamiltonian applied to psi."""
        ncalc = self.compute_number(psi)
        K_psi, H_local_lin, H_local_nonlin = H(t, psi)
        H_psi = K_psi + (H_local_lin + H_local_nonlin)*psi
        mucalc = self.simulator.par_vdot(psi, H_psi) * self.dx * self.dy / ncalc
        return mucalc.real

    def compute_energy(self, t, psi, H):
        """Computes approximate chemical potential (for use as a global energy
        offset, for example) corresponding a given Hamiltonian, where H is a
        function that returns the result of that Hamiltonian applied to psi."""
        K_psi, H_local_lin, H_local_nonlin = H(t, psi)
        # Total energy operator. Differs from total Hamiltonian in that the
        # nonlinear term is halved in order to avoid double counting the
        # interaction energy:
        E_total_psi = K_psi + (H_local_lin + 0.5 * H_local_nonlin) * psi
        Ecalc = self.simulator.par_vdot(psi, E_total_psi).real * self.dx * self.dy
        return Ecalc

    def compute_number(self, psi):
        ncalc = self.simulator.par_vdot(psi, psi).real * self.dx * self.dy
        return ncalc

    def normalise(self, psi, N_2D):
        """Normalise psi to the 2D normalisation constant N_2D, which has
        units of a linear density. Modifies psi in-place and returns None."""
        # imposing normalisation on the wavefunction:
        ncalc = self.compute_number(psi)
        psi[:] *= np.sqrt(N_2D/ncalc)

    def find_groundstate(self, system, H, mu, psi, relaxation_parameter=1.7, convergence=1e-13,
                         output_interval=100, output_directory=None, convergence_check_interval=10):
        """Find the groundstate of a condensate with sucessive overrelaxation.
        A function for the system of equations being solved is required, as is
        the Hamiltonian (used only for computing the chemical potential to
        print statistics), and an initial guess. If the initial guess for psi
        is real, then real arithmetic will be used throughout the computation.
        Otherwise it should be complex. The relaxation parameter must be
        between 0 and 2, and generally somewhere between 1.5 and 2 gives
        fastest convergence The more points per MPI task, the higher the
        relaxation parameter can be. The calculation will stop when the
        chemical potential is correct to within the given convergence. To save
        time, this will only be computed every convergence_check_interval
        steps. If output_directory is None, the output callback will still be
        called every output_interval steps, but it will just print statistics
        and not output anything to file. output_interval can also be a list of
        integers for which steps output should be saved."""

        """Find the groundstate of a condensate using sucessive overrelaxation"""

        if not self.simulator.MPI_rank: # Only rank 0 should print
            print('\n==========')
            print('Beginning successive over relaxation')
            print("Target chemical potential is: " + repr(mu))
            print("Convergence criterion is: {}".format(convergence))
            print('==========')

        def output_callback(i, t, psi):
            mucalc = self.compute_mu(t, psi, H)
            convergence_calc = abs((mucalc - mu)/mu)
            time_per_step = (time.time() - start_time)/i if i else np.nan

            output_log_dtype = [('step_number', int), ('mucalc', float),
                                ('convergence', float), ('time_per_step', float)]

            output_log_data = np.array((i, mucalc, convergence_calc, time_per_step), dtype=output_log_dtype)

            if output_directory is not None:
                hdf_output.save(psi, output_log_data)
            message =  ('step: %d'%i +
                        '  mucalc: ' + repr(mucalc) +
                        '  convergence: %E'%convergence_calc +
                        '  time per step: {}'.format(format_float(time_per_step, units='s')))
            if not self.simulator.MPI_rank: # Only rank 0 should print
                print(message)

        if output_directory is not None:
            hdf_output = HDFOutput(self.simulator, output_directory)

        # Start the relaxation:
        start_time = time.time()
        successive_overrelaxation(self.simulator, system, psi, relaxation_parameter, convergence,
                                  output_interval, output_callback, post_step_callback=None,
                                  convergence_check_interval=convergence_check_interval)
        if not self.simulator.MPI_rank: # Only rank 0 should print
            print('Convergence reached')
        return psi


    def evolve(self, dt, t_final, H, psi, mu=0, method='rk4', imaginary_time=False,
               output_interval=100, output_directory=None, post_step_callback=None):

        """Evolve a wavefunction in time. Timestep, final time, the
        Hamiltonian H and the initial wavefunction are required. mu is
        optional, but will be subtracted from the Hamiltonian if provided -
        this is important in the case of imaginary time evolution, as the
        wavefunction will relax toward this chemical potential. method can be
        either of 'rk4' or 'rk4ilip'. If output_directory is None, the output
        callback will still be called every output_interval steps, but it will
        just print statistics and not output anything to file. output_interval
        can also be a list of integers for which steps output should be
        saved."""
        if not self.simulator.MPI_rank: # Only one process prints to stdout:
            print('\n==========')
            if imaginary_time:
                print("Beginning {}{} of imaginary time evolution".format(format_float(t_final), self.time_units))
            else:
                print("Beginning {}{} of time evolution".format(format_float(t_final), self.time_units))
            print('Using dt = {}{}'.format(format_float(dt), self.time_units))
            print('==========')

        if imaginary_time and method == 'rk4ilip':
            omega_imag_provided=True

            def dpsi_dt(t, psi):
                """The differential equation for psi in imaginary time, as
                well as the angular frequencies corresponding to the spatial
                part of the Hamiltonian for use with the RK4ILIP method"""
                K_psi, H_local_lin, H_local_nonlin = H(t, psi)
                omega_imag = -(H_local_lin + H_local_nonlin - mu)/self.hbar
                d_psi_dt = -1 / self.hbar * K_psi + omega_imag * psi
                return d_psi_dt, omega_imag

        elif method == 'rk4ilip':
            omega_imag_provided=False

            def dpsi_dt(t, psi):
                """The differential equation for psi, as well as the angular
                frequencies corresponding to the spatial part of the
                Hamiltonian for use with the RK4ILIP method"""
                K_psi, H_local_lin, H_local_nonlin = H(t, psi)
                omega = (H_local_lin + H_local_nonlin - mu)/self.hbar
                d_psi_dt = -1j / self.hbar * K_psi -1j*omega * psi
                return d_psi_dt, omega

        elif imaginary_time and method == 'rk4':

            def dpsi_dt(t, psi):
                """The differential equation for psi in imaginary time"""
                K_psi, H_local_lin, H_local_nonlin = H(t, psi)
                d_psi_dt = -1 / self.hbar * (K_psi + (H_local_lin + H_local_nonlin - mu) * psi)
                return d_psi_dt

        elif method == 'rk4':

            def dpsi_dt(t, psi):
                """The differential equation for psi"""
                K_psi, H_local_lin, H_local_nonlin = H(t, psi)
                d_psi_dt = -1j / self.hbar * (K_psi + (H_local_lin + H_local_nonlin - mu) * psi)
                return d_psi_dt

        else:
            raise ValueError(method)

        def output_callback(i, t, psi):
            energy_err = self.compute_energy(t, psi, H) / E_initial - 1
            number_err = self.compute_number(psi) / n_initial - 1
            time_per_step = (time.time() - start_time)/i if i else np.nan

            output_log_dtype = [('step_number', int), ('time', float),
                                ('number_err', float), ('energy_err', float), ('time_per_step', float)]
            output_log_data = np.array((i, t, number_err, energy_err, time_per_step), dtype=output_log_dtype)
            if output_directory is not None:
                hdf_output.save(psi, output_log_data)

            log_time_units = '' if self.natural_units else self.time_units
            message = ('step: %d' % i +
                      '  t = {}'.format(format_float(t, units=log_time_units)) +
                      '  number_err: %+.02E' % number_err +
                      '  energy_err: %+.02E' % energy_err +
                      '  time per step: {}'.format(format_float(time_per_step, units='s')))
            if not self.simulator.MPI_rank: # Only rank 0 should print
                print(message)

        if output_directory is not None:
            hdf_output = HDFOutput(self.simulator, output_directory)

        E_initial = self.compute_energy(0, psi, H)
        n_initial = self.compute_number(psi)
        start_time = time.time()

        # Start the integration:
        if method == 'rk4':
            rk4(dt, t_final, dpsi_dt, psi,
                output_interval=output_interval, output_callback=output_callback, post_step_callback=post_step_callback)
        elif method == 'rk4ilip':
            rk4ilip(dt, t_final, dpsi_dt, psi, omega_imag_provided,
                    output_interval=output_interval, output_callback=output_callback, post_step_callback=post_step_callback)

        return psi

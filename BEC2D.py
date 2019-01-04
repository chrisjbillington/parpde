from __future__ import division, print_function
import numpy as np
from parPDE import HDFOutput, format_float


class BEC2D(object):
    def __init__(self, simulator, natural_units=False, use_ffts=False):
        self.simulator = simulator
        self.natural_units = natural_units
        self.dx = simulator.dx
        self.dy = simulator.dy
        self.use_ffts = use_ffts

        if natural_units:
            self.hbar = 1
            self.time_units = 'timeunits'
        else:
            self.hbar = 1.054571726e-34
            self.time_units = 's'

    def compute_mu(self, t, psi, H):
        """Computes approximate chemical potential (for use as a global energy
        offset, for example) corresponding a given Hamiltonian, where H is a
        function that returns the result of that Hamiltonian applied to psi."""
        ncalc = self.compute_number(psi)
        K, H_local_lin, H_local_nonlin = H(t, psi)
        K_psi = self.simulator.par_operator(K, psi, use_ffts=self.use_ffts)
        H_psi = K_psi + (H_local_lin + H_local_nonlin)*psi
        mucalc = self.simulator.par_vdot(psi, H_psi) * self.dx * self.dy / ncalc
        return mucalc.real

    def compute_mu_convergence(self, t, psi, H, mu_target):
        """Computes the RMS discrepancy from the target chemical potential"""

        K, H_local_lin, H_local_nonlin = H(t, psi)
        K_psi = self.simulator.par_operator(K, psi, use_ffts=self.use_ffts)
        H_psi = K_psi + (H_local_lin + H_local_nonlin)*psi

        residuals = H_psi - mu_target*psi
        sum_squared_residuals = self.simulator.par_vdot(residuals, residuals).real
        denominator = mu_target*psi
        sum_squared_denominator =  self.simulator.par_vdot(denominator, denominator).real
        convergence = np.sqrt(sum_squared_residuals/sum_squared_denominator)
        return convergence

    def compute_energy(self, t, psi, H):
        """Computes approximate chemical potential (for use as a global energy
        offset, for example) corresponding a given Hamiltonian, where H is a
        function that returns the result of that Hamiltonian applied to psi."""
        K, H_local_lin, H_local_nonlin = H(t, psi)
        K_psi = self.simulator.par_operator(K, psi, use_ffts=self.use_ffts)
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
        units of a linear density. Modifies psi in-place and returns pre-
        update normalisation."""
        # imposing normalisation on the wavefunction:
        ncalc = self.compute_number(psi)
        psi[:] *= np.sqrt(N_2D/ncalc)
        return ncalc

    def find_groundstate(self, H, mu, psi, relaxation_parameter=1.7, convergence=1e-13,
                         output_interval=100, output_directory=None, convergence_check_interval=10,
                         boundary_mask=None):
        """Find the groundstate of a condensate with sucessive overrelaxation.
        Requires the Hamiltonian, an initial guess of the wavefunction, and
        the desired chemical potential. The groundstate of the given chemical
        potential will be found, regardless of atom number.  If the initial
        guess for psi is real, then real arithmetic will be used throughout
        the computation. Otherwise it should be complex. The relaxation
        parameter must be between 0 and 2, and generally somewhere between 1.5
        and 2 gives fastest convergence The more points per MPI task, the
        higher the relaxation parameter can be. The calculation will stop when
        the chemical potential is correct to within the given convergence. To
        save time, this will only be computed every convergence_check_interval
        steps. If output_directory is None, the output callback will still be
        called every output_interval steps, but it will just print statistics
        and not output anything to file. output_interval can also be a list of
        integers for which steps output should be saved."""

        if not self.simulator.MPI_rank: # Only rank 0 should print
            print('\n==========')
            print('Beginning successive overrelaxation')
            print("Target chemical potential is: " + repr(mu))
            print("Convergence criterion is: {}".format(convergence))
            print('==========')

        def groundstate_system(psi):
            """The system of equations Ax = b to be solved with sucessive
            overrelaxation to find the groundstate. For us this is H*psi = mu*psi.
            Here we compute b, the diagonal part of A, and the coefficients for
            representing the nondiagonal part of A as a sum of operators to be
            evaluated by the solver."""
            K, H_local_lin, H_local_nonlin = H(0, psi)
            A_diag = H_local_lin + H_local_nonlin
            A_nondiag = K
            b = mu*psi
            return A_diag, A_nondiag, b

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


    def evolve(self, dt, t_final, H, psi, mu=0, method='rk4', imaginary_time=False,
               output_interval=100, output_directory=None, post_step_callback=None, flush_output=True,
               estimate_error=True):

        """Evolve a wavefunction in time. Timestep, final time, the
        Hamiltonian H and the initial wavefunction are required. mu is
        optional, but will be subtracted from the Hamiltonian if provided -
        this is important in the case of imaginary time evolution, as the
        wavefunction will relax toward this chemical potential. Method can be
        either of 'rk4' or 'rk4ilip'. If output_directory is None, the output
        callback will still be called every output_interval steps, but it will
        just print statistics and not output anything to file. output_interval
        can also be a list of integers for which steps output should be saved.
        If imaginary_time=True, the wavefunction is not normalised after each
        step - it's assumed you want to relax it to the provided chemical
        potential. If you want to normalise it every step then provide a
        post_step_callback such as lambda i, t, psi, infodict: BEC2D.normalise(psi, N_2D)"""
        if not self.simulator.MPI_rank: # Only one process prints to stdout:
            print('\n==========')
            if imaginary_time:
                print("Beginning {}{} of imaginary time evolution".format(format_float(t_final), self.time_units))
            else:
                print("Beginning {}{} of time evolution".format(format_float(t_final), self.time_units))
            print('Using method: {} with dt = {}{}'.format(method, format_float(dt), self.time_units))
            print('==========')

        # Pick a differential equation based on the requirements of the method
        # being used, and whether we are evolving in imaginary time or not:
        if method == 'rk4':
            if imaginary_time:

                def dpsi_dt(t, psi):
                    """The differential equation for psi in imaginary time"""
                    K, H_local_lin, H_local_nonlin = H(t, psi)
                    K_psi = self.simulator.par_operator(K, psi, use_ffts=self.use_ffts)
                    return -1 / self.hbar * (K_psi + (H_local_lin + H_local_nonlin - mu) * psi)

            else:

                def dpsi_dt(t, psi):
                    """The differential equation for psi"""
                    K, H_local_lin, H_local_nonlin = H(t, psi)
                    K_psi = self.simulator.par_operator(K, psi, use_ffts=self.use_ffts)
                    d_psi_dt = -1j / self.hbar * (K_psi + (H_local_lin + H_local_nonlin - mu) * psi)
                    return d_psi_dt

        elif method in ['rk4ip', 'fss2', 'fss4']:
            if imaginary_time:
                K, _, _ = H(0, psi)
                nonlocal_operator = -1/self.hbar * K

                def local_operator(t, psi):
                    K, H_local_lin, H_local_nonlin = H(t, psi)
                    local_operator = -1/self.hbar * (H_local_lin + H_local_nonlin - mu)
                    return local_operator

            else:
                K, _, _ = H(0, psi)
                nonlocal_operator = -1j/self.hbar * K

                def local_operator(t, psi):
                    K, H_local_lin, H_local_nonlin = H(t, psi)
                    local_operator = -1j/self.hbar * (H_local_lin + H_local_nonlin - mu)
                    return local_operator

        elif method == 'rk4ilip':
            if imaginary_time:
                omega_imag_provided=True

                def dpsi_dt(t, psi):
                    """The differential equation for psi in imaginary time, as
                    well as the angular frequencies corresponding to the spatial
                    part of the Hamiltonian for use with the RK4ILIP method"""
                    K, H_local_lin, H_local_nonlin = H(t, psi)
                    K_psi = self.simulator.par_operator(K, psi, use_ffts=self.use_ffts)
                    omega_imag = -(H_local_lin + H_local_nonlin - mu)/self.hbar
                    d_psi_dt = -1 / self.hbar * K_psi + omega_imag * psi
                    return d_psi_dt, omega_imag
            else:
                omega_imag_provided=False

                def dpsi_dt(t, psi):
                    """The differential equation for psi, as well as the angular
                    frequencies corresponding to the spatial part of the
                    Hamiltonian for use with the RK4ILIP method"""
                    K, H_local_lin, H_local_nonlin = H(t, psi)
                    K_psi = self.simulator.par_operator(K, psi, use_ffts=self.use_ffts)
                    omega = (H_local_lin + H_local_nonlin - mu)/self.hbar
                    d_psi_dt = -1j / self.hbar * K_psi -1j*omega * psi
                    return d_psi_dt, omega

        else:
            msg = "method must be one of 'rk4', 'rk4ilip', 'rk4ip', 'fss2' or 'fss4'"
            raise ValueError(msg)

        def output_callback(i, t, psi, infodict):
            energy_err = self.compute_energy(t, psi, H) / E_initial - 1
            number_err = self.compute_number(psi) / n_initial - 1
            time_per_step = infodict['time per step']
            step_err = infodict['step error']

            if imaginary_time:
                convergence = self.compute_mu_convergence(t, psi, H, mu)
                output_log_dtype = [('step', int), ('time', float),
                                    ('dN/N', float), ('convergence', float),
                                    ('step err', float), ('time per step', float)]

                output_log_data = np.array((i, t, number_err, convergence, step_err, time_per_step),
                                           dtype=output_log_dtype)
            else:
                output_log_dtype = [('step', int), ('time', float),
                                    ('dN/N', float), ('dE/E', float),
                                    ('step err', float), ('time per step', float)]

                output_log_data = np.array((i, t, number_err, energy_err, step_err, time_per_step),
                                           dtype=output_log_dtype)
            if output_directory is not None:
                hdf_output.save(psi, output_log_data)

            message = ('step: %d' % i +
                      ' | t = {}'.format(format_float(t, units=self.time_units)) +
                      ' | dN/N: %+.02E' % number_err +
                     ((' | convergence: %E' % convergence)  if imaginary_time else (' | dE/E: %+.02E' % energy_err)) +
                      ' | step err: %.03E' % step_err +
                      ' | time per step: {}'.format(format_float(time_per_step, units='s')))
            if not self.simulator.MPI_rank: # Only rank 0 should print
                print(message)

        if output_directory is not None:
            hdf_output = HDFOutput(self.simulator, output_directory, flush_output=flush_output)

        E_initial = self.compute_energy(0, psi, H)
        n_initial = self.compute_number(psi)

        # Start the integration:
        if method == 'rk4':
            self.simulator.rk4(dt, t_final, dpsi_dt, psi, output_interval=output_interval,output_callback=output_callback,
                               post_step_callback=post_step_callback, estimate_error=estimate_error)
        elif method == 'rk4ip':
            self.simulator.rk4ip(dt, t_final, nonlocal_operator, local_operator, psi,
                                 output_interval=output_interval, output_callback=output_callback,
                                 post_step_callback=post_step_callback, estimate_error=estimate_error)
        elif method == 'rk4ilip':
            self.simulator.rk4ilip(dt, t_final, dpsi_dt, psi, omega_imag_provided, output_interval=output_interval,
                                   output_callback=output_callback, post_step_callback=post_step_callback,
                                   estimate_error=estimate_error)
        elif method == 'fss2':
            self.simulator.split_step(dt, t_final, nonlocal_operator, local_operator, psi, method_order=2,
                                      output_interval=output_interval, output_callback=output_callback,
                                      post_step_callback=post_step_callback,
                                      estimate_error=estimate_error)
        elif method == 'fss4':
            self.simulator.split_step(dt, t_final, nonlocal_operator, local_operator, psi, method_order=4,
                                      output_interval=output_interval, output_callback=output_callback,
                                      post_step_callback=post_step_callback,
                                      estimate_error=estimate_error)

        return psi

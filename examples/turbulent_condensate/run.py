# An example of a turbulent BEC in a harmonic trap. The groundstate is found
# and then some vortices randomly printed about with a phase printing. Some
# evolution in imaginary time is then performed to smooth things out before
# evolving the BEC in time.

# Run with 'mpirun -n <N CPUs> python run_example.py'

from __future__ import division, print_function
import sys
# sys.path.insert(0, '../..') # The location of the modules we need to import

import numpy as np

from parPDE import Simulator2D, LAPLACIAN
from parPDE.BEC2D import BEC2D


def get_number_and_trap(rhomax, R):
    """Gives the 2D normalisation constant and trap frequency required for the
    specified maximum density and radius of a single-component condensate in
    the Thomas-Fermi approximation"""
    N = pi * rhomax * R**2 / 2
    omega = np.sqrt(2 * g * rhomax / (m * R**2))
    return N, omega


# Constants:
pi = np.pi
hbar = 1.054571726e-34                        # Reduced Planck's constant
a_0  = 5.29177209e-11                         # Bohr radius
u    = 1.660539e-27                           # unified atomic mass unit
m  = 86.909180*u                              # 87Rb atomic mass
a  = 98.98*a_0                                # 87Rb |2,2> scattering length
g  = 4*pi*hbar**2*a/m                         # 87Rb self interaction constant

rhomax = 2.5e14 * 1e6                         # Desired peak condensate density
R = 7.5e-6                                    # Desired condensate radius
mu = g * rhomax                                # Approximate chemical potential for desired max density
                                              # (assuming all population is in in mF=+1 or mF=-1)
N_2D, omega = get_number_and_trap(rhomax, R)  # 2D normalisation constant and trap frequency
                                              # required for specified radius and peak density

# Space:
nx_global = ny_global = 256
x_max_global = y_max_global = 10e-6

simulator = Simulator2D(-x_max_global, x_max_global, -y_max_global, y_max_global, nx_global, ny_global,
                        periodic_x=True, periodic_y=True, operator_order=6)
bec2d = BEC2D(simulator, natural_units=False, use_ffts=True)

x = simulator.x
y = simulator.y
dx = simulator.dx
dy = simulator.dy

r2 = x**2.0 + y**2.0
r  = np.sqrt(r2)

# A Harmonic trap:
V = 0.5 * m * omega**2 * R**2.0 * (r/R)**2

dispersion_timescale = dx**2 * m / (pi * hbar)
chemical_potential_timescale = 2*pi*hbar/mu
potential_timescale = 2*pi*hbar/V.max()

K = -hbar**2/(2*m)*LAPLACIAN

def H(t, psi):
    """The Hamiltonian for single-component wavefunction psi. Returns the
    kinetic term as an OperatorSum instance, and the local terms separately."""
    H_local_lin = V
    H_local_nonlin = g * abs(psi)**2
    return K, H_local_lin, H_local_nonlin


if __name__ == '__main__':
    # The initial Thomas-Fermi guess:
    psi = rhomax * (1 - (x**2 + y**2) / R**2)
    psi[psi < 0] = 0
    psi = np.sqrt(psi)

    # Find the groundstate:
    psi = bec2d.find_groundstate(H, mu, psi, relaxation_parameter=1.7, convergence=1e-13,
                                 output_interval=100, output_directory='groundstate', convergence_check_interval=10)

    # psi is real so far, convert it to complex:
    psi = np.array(psi, dtype=complex)

    # Print some vortices, seeding the pseudorandom number generator so that
    # MPI processes all agree on where the vortices are:
    np.random.seed(42)
    for i in range(30):
        sign = np.sign(np.random.normal())
        x_vortex = np.random.normal(0, scale=R)
        y_vortex = np.random.normal(0, scale=R)
        psi[:] *= np.exp(sign * 1j*np.arctan2(x - y_vortex, y - x_vortex))

    # Smooth it a bit in imaginary time:
    psi = bec2d.evolve(dt=dispersion_timescale/2, t_final=chemical_potential_timescale,
                       H=H, psi=psi, mu=mu, method='rk4', imaginary_time=True,
                       output_interval=100, output_directory='smoothing')

    # And evolve it in time for 10ms:
    psi = bec2d.evolve(dt=dispersion_timescale/2, t_final=10e-3,
                       H=H, psi=psi, mu=mu, method='rk4', imaginary_time=False,
                       output_interval=100, output_directory='evolution')

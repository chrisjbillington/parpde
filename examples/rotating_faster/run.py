# Example file that finds the groundstate of a condensate in a rotating frame.
# Takes quite some time to run so you can just stop it when you run out of
# patience and run the plotting script.

# Run with 'mpirun -n <N CPUs> python run.py'

from __future__ import division, print_function
import sys
sys.path.insert(0, '../..') # The location of the modules we need to import

import numpy as np

from parPDE import Simulator2D, LAPLACIAN, GRADX, GRADY
from BEC2D import BEC2D


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

R = 7.5e-6                                    # Desired condensate radius
N = 50                                       # desired number of vortices
Omega = N*hbar/(m*R**2)                       # Rotation speed required for this number of vortices

fractional_rotation_speed = 0.9999
omega = 1/fractional_rotation_speed * Omega             # Trap frequency
omega_eff = np.sqrt(omega**2 - Omega**2)                # effective trap frequenct inc. centrifugal force
mu = 0.5 * m * omega_eff**2 * R**2.0 + hbar*Omega       # Chemical potential required for Thomas-Fermi radius to equal R
rhomax = (mu - hbar*Omega) /g                           # Peak condensate density in TF approx
healing_length = 2*pi*hbar/np.sqrt(2*m*g*rhomax)


# Space:
nx_global = ny_global = 768
x_max_global = y_max_global = 30e-6

simulator = Simulator2D(-x_max_global, x_max_global, -y_max_global, y_max_global, nx_global, ny_global,
                        periodic_x=False, periodic_y=False, operator_order=6)
bec2d = BEC2D(simulator, natural_units=False, use_ffts=False)

print(2*x_max_global/healing_length)



x = simulator.x
y = simulator.y
dx = simulator.dx
dy = simulator.dy

r2 = x**2.0 + y**2.0
r  = np.sqrt(r2)

assert False

# The harmonic trap
alpha = 2
V = 0.5 * m * omega**2 * r**2

# The kinetic and rotation terms of the Hamiltonian:
K = -hbar**2/(2*m)*LAPLACIAN - 1j*hbar*Omega*(y * GRADX - x * GRADY)


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

    psi = np.array(psi, dtype=complex)

    # Some random vortices to give the relaxation something to start with:
    np.random.seed(42) # must be seeded so that all MPI processes agree
    for i in range(100):
        x_vortex = np.random.normal(0, scale=R)
        y_vortex = np.random.normal(0, scale=R)
        psi[:] *= np.exp(1j*np.arctan2(x - y_vortex, y - x_vortex))

    # Find the groundstate:
    psi = bec2d.find_groundstate(H, mu, psi, boundary_mask=None, relaxation_parameter=1.9, convergence=1e-9,
                                 output_interval=500, output_directory='groundstate', convergence_check_interval=10)


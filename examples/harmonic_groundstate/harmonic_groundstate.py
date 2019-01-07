# Run with 'mpirun -n <N CPUs> python run_example.py'

from __future__ import division, print_function
import numpy as np

from parPDE import Simulator2D, LAPLACIAN
from BEC2D import BEC2D


nx_global = ny_global = 500
x_max_global = y_max_global = 10/np.sqrt(2)

simulator = Simulator2D(-x_max_global, x_max_global, -y_max_global, y_max_global, nx_global, ny_global,
                        periodic_x = True, periodic_y=True, operator_order=2)
bec2d = BEC2D(simulator, natural_units=True)

x = simulator.x
y = simulator.y
dx = simulator.dx
dy = simulator.dy

r = np.sqrt(x**2.0 + y**2.0)
V = 0.5 * r**2


def H(t, psi):
    """The Hamiltonian for single-component wavefunction psi. Returns the
    kinetic term acting on psi and the local terms (not acting on psi)
    separately."""
    grad2psi = simulator.par_laplacian(psi)
    H_local_lin = V
    K_psi = -0.5*grad2psi
    return K_psi, H_local_lin, 0


def groundstate_system(psi):
    """The system of equations Ax = b to be solved with sucessive
    overrelaxation to find the groundstate. For us this is H*psi = mu*psi.
    Here we compute b, the diagonal part of A, and the coefficients for
    representing the nondiagonal part of A as a sum of operators to be
    evaluated by the solver."""
    A_diag = V
    A_nondiag = -0.5*LAPLACIAN
    b = psi
    return A_diag, A_nondiag, b


if __name__ == '__main__':
    # The initial Thomas-Fermi guess:

    psi_0_1D = np.pi**(-0.25)*np.exp(-x**2/2)*np.ones(r.shape)/(np.sqrt(2*x_max_global))

    psi_0 = 1/np.sqrt(np.pi)*np.exp(-r**2/2)

    psi_1 = np.sqrt(2)*x/np.sqrt(np.pi)*np.exp(-r**2/2)

    psi = 1/np.sqrt(2) * (psi_0 + psi_1)

    psi = psi_0

    sum_integral = np.abs(psi**2).sum()*dx*dy
    print('Integral:', repr(sum_integral))
    print('FD energy:', bec2d.compute_energy(0, psi, H))

    grad2psi = simulator.fft_laplacian(psi)
    E_psi = (-0.5*grad2psi + V*psi)
    energy_density = (psi.conj()*E_psi).real
    print('FFT energy:', energy_density.sum()*dx*dy)

    err = np.abs((bec2d.compute_energy(0, psi, H) - 1.0)/1.0)

    print('err:', err)

    print('for comparison with paper:', 1.5*(1-1.5*err))

    # import matplotlib.pyplot as plt
    # plt.subplot(131)
    # plt.imshow(psi_0, interpolation='nearest')
    # plt.subplot(132)
    # plt.imshow(psi_1, interpolation='nearest')
    # plt.subplot(133)
    # plt.imshow(psi, interpolation='nearest')
    # plt.show()
    import sys
    sys.exit(0)
    assert False

    psi = np.ones(r.shape)
    # Find the groundstate:
    psi = bec2d.find_groundstate(groundstate_system, H, 1.0, psi, relaxation_parameter=1.0, convergence=1e-14,
                                 output_interval=100, output_directory='groundstate', convergence_check_interval=10)

    # psi is real so far, convert it to complex:
    # psi = np.array(psi, dtype=complex)


    # # Print some vortices, seeding the pseudorandom number generator so that
    # # MPI processes all agree on where the vortices are:
    # np.random.seed(42)
    # for i in range(30):
    #     sign = np.sign(np.random.normal())
    #     x_vortex = np.random.normal(0, scale=R)
    #     y_vortex = np.random.normal(0, scale=R)
    #     psi[:] *= np.exp(sign * 1j*np.arctan2(x - y_vortex, y - x_vortex))

    # psi_initial = psi.copy()
    # METHOD = 'fourier'
    # Smooth it a bit in imaginary time:
    # for i in range(10):
    #     psi = bec2d.evolve(dt=0.01, t_final=1,
    #                        H=H, psi=psi, mu=1, method='rk4', imaginary_time=True,
    #                        output_interval=100, output_directory='smoothing', post_step_callback= lambda i, t, psi: bec2d.normalise(psi, 1))

    #     print(bec2d.compute_energy(0, psi, H))
    # # And evolve it in time for 10ms:
    # psi = bec2d.evolve(dt=dispersion_timescale/2, t_final=10e-3,
    #                    H=H, psi=psi, mu=mu, method='rk4', imaginary_time=False,
    #                    output_interval=100, output_directory='evolution')

    # gradx_psi_fourier = simulator.fft_gradx(psi_fourier)
    # grady_psi_fourier = simulator.fft_grady(psi_fourier)
    # jx_fourier = (-1j*psi_fourier.conj()*gradx_psi_fourier).real
    # jy_fourier = (-1j*psi_fourier.conj()*grady_psi_fourier).real

    # import matplotlib.pyplot as plt
    # plt.subplot(211)
    # plt.title('jx of psi')
    # plt.imshow(jx_fourier.transpose(), origin='lower', interpolation='nearest')

    # plt.subplot(212)
    # plt.title('jy of psi')
    # plt.imshow(jy_fourier.transpose(), origin='lower', interpolation='nearest')

    # plt.show()

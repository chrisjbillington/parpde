# An example of a wave equation. The initial condition is a Gaussian disturbance, which
# propagates outward. The bottom third of the spatial region has a slower wave speed by
# a factor of two, so refraction can be observed.

# Run with 'mpirun -n <N CPUs> python run.py'

import numpy as np
from parPDE import Simulator2D, LAPLACIAN, HDFOutput, format_float

# Space:
nx_global = ny_global = 256
x_max_global = y_max_global = 10  # Metres

c_vac = 10  # Wave speed in 'vacuum', metres per second
n_med = 2  # Refractive index of medium
y_boundary = -3.33  # y position of boundary between vacuum and medium
sigma_0 = 0.5  # stdev of initial condution Gaussian impulse

# Create a simulator object:
simulator = Simulator2D(
    -x_max_global,
    x_max_global,
    -y_max_global,
    y_max_global,
    nx_global,
    ny_global,
    periodic_x=False,
    periodic_y=False,
    operator_order=6,
)

# An object for saving output.
hdf_output = HDFOutput(simulator, "evolution")

# Note: nx, ny, x, and y are for the region of space being simulated on this core
# only, as opposed to the 'global' equivalents which cover the entire spatial region
# being simulated over all MPI processes:
nx = simulator.nx
ny = simulator.ny
x = simulator.x
y = simulator.y
dx = simulator.dx
dy = simulator.dy

# The wave speed as a function of space:
c = np.full((nx, ny), c_vac)
c[:, (y <= y_boundary).reshape(simulator.ny)] =  c_vac / n_med

# How long it takes the wave to move one gridpoint. Can use this to pick a sensible
# timestep for rk4:
propagation_timescale = dx / c_vac


def dpsi_dt(t, psi):
    """The differential equation for our vector psi, which contains the wavefield u and
    its first derivative, du_dt"""

    # Extract u and du_dt from psi:
    u = psi[0]
    du_dt = psi[1]

    # Construct the output array dpsi_dt, containing the time derivatives of the input
    # array:
    dpsi_dt = np.empty_like(psi)

    # du_dt:
    dpsi_dt[0] = du_dt
    # d2u_dt2:
    dpsi_dt[1] = c ** 2 * simulator.par_operator(LAPLACIAN, u)

    return dpsi_dt


# Function to run every time we output to file:
def output_callback(i, t, psi, infodict):
    # The extra info we want to save in the log:
    time_per_step = infodict['time per step']
    step_err = infodict['step error']

    output_log_dtype = [
        ('step', int),
        ('time', float),
        ('step_err', float),
        ('time per step', float),
    ]
    output_log_data = np.array((i, t, step_err, time_per_step), dtype=output_log_dtype)
    # Save the field and the log data:
    hdf_output.save(psi, output_log_data)

    # Print some info:
    message = (
        f'step: {i}'
        + f' | t = {format_float(t, units="s")}'
        + f' | step err: {step_err:.03E}'
        + f' | time per step: {format_float(time_per_step, units="s")}'
    )
    if not simulator.MPI_rank:  # Ensure only one process (rank 0) prints
        print(message)


if __name__ == '__main__':
    # The initial condition for u and du_dt:
    u = np.exp(-(x ** 2 + y ** 2) / (2 * sigma_0 ** 2))
    du_dt = 0

    # Packed into a single array:
    psi = np.empty((2, simulator.nx, simulator.ny))
    psi[0] = u
    psi[1] = du_dt

    # Evolve in time for 2s:
    simulator.rk4(
        dt=propagation_timescale / 10,
        t_final=2,
        dpsi_dt=dpsi_dt,
        psi=psi,
        output_interval=20,
        output_callback=output_callback,
        post_step_callback=None,
        estimate_error=True,
)

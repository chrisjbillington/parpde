from __future__ import print_function


cimport cython
import numpy as np

ctypedef fused psi_t:
    double
    double complex

@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def rk4_step(
    double t, double dt, psi_t[:, :] psi,  dpsi_dt,
    psi_t[:, :] k1_temp, psi_t[:, :] k2_temp, psi_t[:, :] k3_temp, psi_t[:, :] k4_temp, psi_t[:, :] psi_temp):
    cdef int i
    cdef int j
    cdef int nx = psi.shape[0]
    cdef int ny = psi.shape[1]

    # Numpy arrays of the right shape to pass to the differentiual equation.
    # They point to the same memory as the 1D cython memoryview arrays:
    psi_ndarray =  np.asarray(psi)
    psi_temp_ndarray =  np.asarray(psi_temp)
    k1_temp_ndarray = np.asarray(k1_temp)
    k2_temp_ndarray = np.asarray(k1_temp)
    k3_temp_ndarray = np.asarray(k1_temp)
    k4_temp_ndarray = np.asarray(k1_temp)

    # First evaluation of dpsi_dt:
    dpsi_dt(t, psi_ndarray, k1_temp_ndarray)

    for i in range(nx):
        for j in range(ny):
            # First prediction of psi, at the midpoint:
            psi_temp[i, j] = psi[i, j] + 0.5*dt * k1_temp[i, j]

    # Second evaluation of dpsi_dt:
    dpsi_dt(t + 0.5*dt, psi_temp_ndarray, k1_temp_ndarray)

    for i in range(nx):
        for j in range(ny):
            # Second prediction of psi, at the midpoint:
            psi_temp[i, j] = psi[i, j] + 0.5*dt * k2_temp[i, j]

    # Third evaluation of dpsi_dt:
    dpsi_dt(t + 0.5*dt, psi_temp_ndarray, k2_temp_ndarray)

    for i in range(nx):
        for j in range(ny):
            # Third prediction of psi, at the endpoint:
            psi_temp[i, j] = psi[i, j] + dt * k3_temp[i, j]

    # Fourth evaluation of dpsi_dt:
    dpsi_dt(t + dt, psi_temp_ndarray, k3_temp_ndarray)

    for i in range(nx):
        for j in range(ny):
            # The final prediction for psi, including this last subste:
            psi[i, j] = psi[i, j] + dt/6.0 * (k1_temp[i, j] + 2*k2_temp[i, j] + 2*k3_temp[i, j] + k4_temp[i, j])



#def rk4_step(t, dt, psi, dpsi_dt, k_temp, psi_temp1, psi_temp2):
#    _rk4_step(t, dt, psi.ravel(), psi.shape, dpsi_dt, k_temp.ravel(), psi_temp1.ravel(), psi_temp2.ravel())

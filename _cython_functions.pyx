from __future__ import print_function


cimport cython
import numpy as np

LAPLACIAN = 0

EMPTY_2D_ARRAY = np.zeros((1, 1))


ctypedef fused double_or_complex:
    double
    double complex


cdef extern from "complex.h":
     double cabs(double complex) nogil


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void _laplacian(double_or_complex [:, :] psi, double dx, double dy, double_or_complex [:, :] out) nogil:
    """Compute the laplacian of the array psi using finite differences. Assumes zero boundary conditions."""

    cdef int i
    cdef int j
    cdef int nx
    cdef int ny
    nx = psi.shape[0]
    ny = psi.shape[1]

    cdef double over_dx2 = 1/dx**2
    cdef double over_dy2 = 1/dy**2

    cdef double_or_complex Lx_ij
    cdef double_or_complex Ly_ij
    for i in range(nx):
        for j in range(ny):
            Lx_ij = -2*psi[i, j]
            Ly_ij = -2*psi[i, j]
            if i != 0:
                Lx_ij += psi[i-1, j]
            if i != nx - 1:
                Lx_ij += psi[i+1, j]
            if j != 0:
                Ly_ij += psi[i, j-1]
            if j != ny - 1:
                Ly_ij += psi[i, j+1]
            out[i, j] = Lx_ij*over_dx2 + Ly_ij*over_dy2


cdef inline void complex_laplacian(double complex [:, :] A, double dx, double dy, double complex [:, :] out) nogil:
    _laplacian(A, dx, dy, out)


cdef inline void real_laplacian(double [:, :] A, double dx, double dy, double [:, :] out) nogil:
    _laplacian(A, dx, dy, out)


def laplacian(psi, dx, dy):
    out = np.empty(psi.shape, dtype=psi.dtype)
    if psi.dtype == np.complex128:
        complex_laplacian(psi, dx, dy, out)
    elif psi.dtype == np.float64:
        real_laplacian(psi, dx, dy, out)
    return out


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void _SOR_step_interior(double_or_complex [:, :] psi, double_or_complex [:, :] A_diag,
                                    double_or_complex [:, :] b, double dx, double dy, double relaxation_parameter,
                                    int use_laplacian, double_or_complex [:, :] laplacian_coefficient,
                                    double * squared_error_ptr, int compute_error) nogil:
    cdef int i
    cdef int j
    cdef int i_prime = 0
    cdef int j_prime = 0
    cdef int nx
    cdef int ny
    nx = psi.shape[0]
    ny = psi.shape[1]

    cdef int laplacian_coefficient_has_x = laplacian_coefficient.shape[0] > 1
    cdef int laplacian_coefficient_has_y = laplacian_coefficient.shape[1] > 1

    cdef double over_dx2 = 1/dx**2
    cdef double over_dy2 = 1/dy**2
    cdef double residual

    cdef double_or_complex A_hollow_psi
    cdef double_or_complex A_diag_total
    cdef double_or_complex psi_GS

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            # Only index the laplacian coefficient array in the directions it varies:
            if laplacian_coefficient_has_x:
                i_prime = i
            if laplacian_coefficient_has_y:
                j_prime = j
            # Compute the total diagonals of A. This is the sum of the
            # diagonal operator given by the user, and the diagonals of the
            # non-diagonal operators (currently only the Laplacian
            # implemented):
            A_diag_total = A_diag[i, j]
            if use_laplacian:
                A_diag_total -= 2*laplacian_coefficient[i_prime, j_prime]*(over_dx2 + over_dy2)

            # Compute the off diagonals of A, operating on psi:
            A_hollow_psi = 0
            if use_laplacian:
                A_hollow_psi += laplacian_coefficient[i_prime, j_prime]*(psi[i-1, j] + psi[i+1, j])*over_dx2
                A_hollow_psi += laplacian_coefficient[i_prime, j_prime]*(psi[i, j-1] + psi[i, j+1])*over_dy2

            if compute_error:
                # Add the squared residuals of the existing solution to the total:
                residual = cabs(A_hollow_psi + A_diag_total*psi[i, j] - b[i, j])
                squared_error_ptr[0] = squared_error_ptr[0] + residual*residual

            # The Gauss-Seidel prediction for psi:
            psi_GS = (b[i, j] - A_hollow_psi)/A_diag_total
            psi[i, j] = psi[i, j] + relaxation_parameter*(psi_GS - psi[i, j])



@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void _SOR_step_edges(double_or_complex [:, :] psi, double_or_complex [:, :] A_diag,
                                    double_or_complex [:, :] b, double dx, double dy, double relaxation_parameter,
                                    int use_laplacian, double_or_complex [:, :] laplacian_coefficient,
                                    double_or_complex[:, :] left_edge_buffer, double_or_complex[:, :] right_edge_buffer,
                                    double_or_complex[:, :] bottom_edge_buffer, double_or_complex[:, :] top_edge_buffer,
                                    double * squared_error_ptr, int compute_error) nogil:
    cdef int i
    cdef int j
    cdef int i_prime = 0
    cdef int j_prime = 0
    cdef int nx
    cdef int ny
    nx = psi.shape[0]
    ny = psi.shape[1]

    cdef int laplacian_coefficient_has_x = laplacian_coefficient.shape[0] > 1
    cdef int laplacian_coefficient_has_y = laplacian_coefficient.shape[1] > 1

    cdef double over_dx2 = 1/dx**2
    cdef double over_dy2 = 1/dy**2
    cdef double residual

    cdef double_or_complex A_hollow_psi
    cdef double_or_complex A_diag_total
    cdef double_or_complex psi_GS

    cdef double_or_complex psi_left
    cdef double_or_complex psi_right
    cdef double_or_complex psi_top
    cdef double_or_complex psi_bottom


    # Start in the lower left corner:
    i = 0
    j = 0
    while True:
        if i == 0:
            psi_left = left_edge_buffer[0, j]
        else:
            psi_left = psi[i - 1, j]
        if i == nx - 1:
            psi_right = right_edge_buffer[0, j]
        else:
            psi_right = psi[i + 1, j]
        if j == 0:
            psi_bottom = bottom_edge_buffer[i, 0]
        else:
            psi_bottom = psi[i, j - 1]
        if j == ny - 1:
            psi_top = top_edge_buffer[i, 0]
        else:
            psi_top = psi[i, j + 1]

        # Only index the laplacian coefficient array in the directions it varies:
        if laplacian_coefficient_has_x:
            i_prime = i
        if laplacian_coefficient_has_y:
            j_prime = j
        # Compute the total diagonals of A. This is the sum of the
        # diagonal operator given by the user, and the diagonals of the
        # non-diagonal operators (currently only the Laplacian
        # implemented):
        A_diag_total = A_diag[i, j]
        if use_laplacian:
            A_diag_total -= 2*laplacian_coefficient[i_prime, j_prime]*(over_dx2 + over_dy2)

        # Compute the off diagonals of A, operating on psi:
        A_hollow_psi = 0
        if use_laplacian:
            A_hollow_psi += laplacian_coefficient[i_prime, j_prime]*(psi_left + psi_right)*over_dx2
            A_hollow_psi += laplacian_coefficient[i_prime, j_prime]*(psi_bottom + psi_top)*over_dy2

        if compute_error:
            # Add the squared residuals of the existing solution to the total:
            residual = cabs(A_hollow_psi + A_diag_total*psi[i, j] - b[i, j])
            squared_error_ptr[0] = squared_error_ptr[0] + residual*residual

        # The Gauss-Seidel prediction for psi:
        psi_GS = (b[i, j] - A_hollow_psi)/A_diag_total
        psi[i, j] = psi[i, j] + relaxation_parameter*(psi_GS - psi[i, j])

        # Run up the left edge:
        if i == 0 and j < ny - 1:
            j += 1
        # Then go to to lower right corner:
        elif i == 0 and j == ny - 1:
            i = nx - 1
            j = 0
        # And run up the right edge:
        elif i == nx - 1 and  j < ny - 1:
            j += 1
        # Then jump one point to the right of the lower left corner:
        elif i == nx - 1 and j == ny - 1:
            i = 1
            j = 0
        # And run along the bottom edge, stopping short one point short of the
        # lower right corner:
        elif j == 0 and i < nx - 2:
            i += 1
        # Then jump one point to the right of the top edge:
        elif j == 0 and i == nx - 2:
            i = 1
            j = ny - 1
        # And run along the top edge, stopping short one point short of the
        # upper right corner:
        elif j == ny - 1 and i < nx - 2:
            i += 1
        else:
            # Done!
            break


cdef inline void _SOR_step_interior_complex(double complex [:, :] psi, double complex [:, :] A_diag,
                                    double complex [:, :] b, double dx, double dy, double relaxation_parameter,
                                    int use_laplacian, double complex [:, :] laplacian_coefficient,
                                    double * squared_error_ptr, int compute_error) nogil:
    _SOR_step_interior(psi, A_diag, b, dx, dy, relaxation_parameter, use_laplacian, laplacian_coefficient,
                       squared_error_ptr, compute_error)


cdef inline void _SOR_step_interior_real(double [:, :] psi, double [:, :] A_diag,
                                    double [:, :] b, double dx, double dy, double relaxation_parameter,
                                    int use_laplacian, double [:, :] laplacian_coefficient,
                                    double * squared_error_ptr, int compute_error) nogil:
    _SOR_step_interior(psi, A_diag, b, dx, dy, relaxation_parameter, use_laplacian, laplacian_coefficient,
                       squared_error_ptr, compute_error)

cdef inline void _SOR_step_edges_complex(double complex [:, :] psi, double complex [:, :] A_diag,
                                    double complex [:, :] b, double dx, double dy, double relaxation_parameter,
                                    int use_laplacian, double complex [:, :] laplacian_coefficient,
                                    double complex [:, :] left_edge_buffer, double complex [:, :] right_edge_buffer,
                                    double complex [:, :] bottom_edge_buffer, double complex [:, :] top_edge_buffer,
                                    double * squared_error_ptr, int compute_error) nogil:
    _SOR_step_edges(psi, A_diag, b, dx, dy, relaxation_parameter, use_laplacian, laplacian_coefficient,
                    left_edge_buffer, right_edge_buffer, bottom_edge_buffer, top_edge_buffer, squared_error_ptr,
                    compute_error)


cdef inline void _SOR_step_edges_real(double [:, :] psi, double [:, :] A_diag,
                                    double [:, :] b, double dx, double dy, double relaxation_parameter,
                                    int use_laplacian, double [:, :] laplacian_coefficient,
                                    double [:, :] left_edge_buffer, double [:, :] right_edge_buffer,
                                    double [:, :] bottom_edge_buffer, double [:, :] top_edge_buffer,
                                    double * squared_error_ptr, int compute_error) nogil:
    _SOR_step_edges(psi, A_diag, b, dx, dy, relaxation_parameter, use_laplacian, laplacian_coefficient,
                    left_edge_buffer, right_edge_buffer, bottom_edge_buffer, top_edge_buffer, squared_error_ptr,
                    compute_error)


def SOR_step_interior(psi, A_diag, A_nondiag, b, dx, dy, relaxation_parameter, compute_error):
    cdef int use_laplacian = LAPLACIAN in A_nondiag
    cdef double squared_error = 0
    cdef int compute_error_cint = compute_error
    if use_laplacian:
        laplacian_coefficient = A_nondiag[LAPLACIAN]
    else:
        laplacian_coefficient = EMPTY_2D_ARRAY
    if psi.dtype == np.complex128:
        if laplacian_coefficient.dtype == np.float64:
            laplacian_coefficient = np.array(laplacian_coefficient, dtype=complex)
        if A_diag.dtype == np.float64:
            A_diag = np.array(A_diag, dtype=complex)
        _SOR_step_interior_complex(psi, A_diag, b, dx, dy, relaxation_parameter, use_laplacian, laplacian_coefficient,
                                   &squared_error, compute_error_cint)
    elif psi.dtype == np.float64:
        _SOR_step_interior_real(psi, A_diag, b, dx, dy, relaxation_parameter, use_laplacian, laplacian_coefficient,
                                &squared_error, compute_error_cint)

    return float(squared_error)

def SOR_step_edges(psi, A_diag, A_nondiag, b, dx, dy, relaxation_parameter,
                   left_edge_buffer, right_edge_buffer, bottom_edge_buffer, top_edge_buffer, squared_error, compute_error):
    cdef int use_laplacian = LAPLACIAN in A_nondiag
    cdef int compute_error_cint = compute_error
    cdef double squared_error_cdouble = squared_error
    if use_laplacian:
        laplacian_coefficient = A_nondiag[LAPLACIAN]
    else:
        laplacian_coefficient = EMPTY_2D_ARRAY
    if psi.dtype == np.complex128:
        if laplacian_coefficient.dtype == np.float64:
            laplacian_coefficient = np.array(laplacian_coefficient, dtype=complex)
        if A_diag.dtype == np.float64:
            A_diag = np.array(A_diag, dtype=complex)
        _SOR_step_edges_complex(psi, A_diag, b, dx, dy, relaxation_parameter, use_laplacian, laplacian_coefficient,
                                left_edge_buffer, right_edge_buffer, bottom_edge_buffer, top_edge_buffer,
                                &squared_error_cdouble, compute_error_cint)
    elif psi.dtype == np.float64:
        _SOR_step_edges_real(psi, A_diag, b, dx, dy, relaxation_parameter, use_laplacian, laplacian_coefficient,
                             left_edge_buffer, right_edge_buffer, bottom_edge_buffer, top_edge_buffer,
                             &squared_error_cdouble, compute_error_cint)
    return float(squared_error_cdouble)


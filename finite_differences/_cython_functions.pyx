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


# Constants for central finite differences:
DEF D_2ND_ORDER_1 = 1.0/5.0

DEF D_4TH_ORDER_1 = 2.0/3.0
DEF D_4TH_ORDER_2 = 1.0/12.0

DEF D_6TH_ORDER_1 = 3.0/4.0
DEF D_6TH_ORDER_2 = -3.0/20.0
DEF D_6TH_ORDER_3 = 1.0/60.0

DEF D2_2ND_ORDER_0 = -2.0
DEF D2_2ND_ORDER_1 = 1.0

DEF D2_4TH_ORDER_0 = -5.0/2.0
DEF D2_4TH_ORDER_1 = 4.0/3.0
DEF D2_4TH_ORDER_2 = -1.0/12.0

DEF D2_6TH_ORDER_0 = -49.0/18.0
DEF D2_6TH_ORDER_1 = 3.0/2.0
DEF D2_6TH_ORDER_2 = -3.0/20.0
DEF D2_6TH_ORDER_3 = 1.0/90.0

@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void iter_edges(int * i, int * j, int nx, int ny, int order) nogil:
    """Increment i and j so as to iterate over the points within order/2 of
    the edge of the array. Caller is responsible for not calling any more when
    i = nx - 1 and j = ny - 1"""
    cdef int npts = order // 2
    if i[0] < npts:
        if j[0] < ny - 1:
            j[0] += 1
        else:
            i[0] += 1
            j[0] = 0
    elif i[0] < nx - npts:
        if j[0] < npts - 1:
            j[0] += 1
        elif j[0] == npts - 1:
            j[0] = ny - npts
        elif j[0] < ny - 1:
            j[0] += 1
        else:
            j[0] = 0
            i[0] += 1
    elif j[0] < ny - 1:
        j[0] += 1
    else:
        i[0] += 1
        j[0] = 0


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline double_or_complex _grad2x_single_point_interior(
    int i, int j, double_or_complex [:, :] psi, double over_dx2, int order, int hollow) nogil:
    """Compute the second x derivative at a single point i, j. No bounds
    checking is performed, so one must be sure that i is at least order/2 away
    fromt the edges. 1/dx^2 must be provided. If "hollow" is true, the central
    point is excluded from the calculation."""
    cdef double_or_complex Lx
    Lx = 0
    if order == 2:
        if not hollow:
            # Central point:
            Lx = D2_2ND_ORDER_0 * psi[i, j]
        # Nearest neighbor:
        Lx += D2_2ND_ORDER_1 * (psi[i-1, j] + psi[i+1, j])
    elif order == 4:
        if not hollow:
            # Central point:
            Lx = D2_4TH_ORDER_0 * psi[i, j]
        # Nearest neighbor:
        Lx += D2_4TH_ORDER_1 * (psi[i-1, j] + psi[i+1, j])
        # Next nearest neighbor:
        Lx += D2_4TH_ORDER_2 * (psi[i-2, j] + psi[i+2, j])
    elif order == 6:
        if not hollow:
            # Central point:
            Lx = D2_6TH_ORDER_0 * psi[i, j]
        # Nearest neighbor:
        Lx += D2_6TH_ORDER_1 * (psi[i-1, j] +  psi[i+1, j])
        # Next nearest neighbor:
        Lx += D2_6TH_ORDER_2 * (psi[i-2, j] + psi[i+2, j])
        # Next next nearest neighbor:
        Lx += D2_6TH_ORDER_3 * (psi[i-3, j] + psi[i+3, j])
    return Lx * over_dx2


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline double_or_complex _grad2y_single_point_interior(
    int i, int j, double_or_complex [:, :] psi, double over_dy2, int order, int hollow) nogil:
    """Compute the second y derivative at a single point i, j. No bounds
    checking is performed, so one must be sure that j is at least order/2 away
    fromt the edges. 1/dy^2 must be provided. If "hollow" is true, the central
    point is excluded from the calculation."""
    cdef double_or_complex Ly
    Ly = 0
    if order == 2:
        if not hollow:
            # Central point:
            Ly = D2_2ND_ORDER_0 * psi[i, j]
        # Nearest neighbor:
        Ly += D2_2ND_ORDER_1 * (psi[i, j-1] + psi[i, j+1])
    elif order == 4:
        if not hollow:
            # Central point:
            Ly = D2_4TH_ORDER_0 * psi[i, j]
        # Nearest neighbor:
        Ly += D2_4TH_ORDER_1 * (psi[i, j-1] + psi[i, j+1])
        # Next nearest neighbor:
        Ly += D2_4TH_ORDER_2 * (psi[i, j-2] + psi[i, j+2])
    elif order == 6:
        if not hollow:
            # Central point:
            Ly = D2_6TH_ORDER_0 * psi[i, j]
        # Nearest neighbor:
        Ly += D2_6TH_ORDER_1 * (psi[i, j-1] + psi[i, j+1])
        # Next nearest neighbor:
        Ly += D2_6TH_ORDER_2 * (psi[i, j-2] + psi[i, j+2])
        # Next next nearest neighbor:
        Ly += D2_6TH_ORDER_3 * (psi[i, j-3] + psi[i, j+3])

    return Ly * over_dy2

@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline double_or_complex _laplacian_single_point_interior(
    int i, int j, double_or_complex [:, :] psi, double over_dx2, double over_dy2, int order, int hollow) nogil:
    """Compute the Laplacian at a single point i, j. No bounds checking is
    performed, so one must be sure that i and j are at least order/2 away
    fromt the edges. 1/dx^2 and 1/dy^2 must be provided. If "hollow" is true,
    the central point is excluded from the calculation."""
    return (_grad2x_single_point_interior(i, j, psi, over_dx2, order, hollow) +
            _grad2y_single_point_interior(i, j, psi, over_dy2, order, hollow))


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void _laplacian_interior(
     double_or_complex [:, :] psi, double_or_complex [:, :] out, double dx, double dy, int order, int hollow) nogil:
    """Compute the laplacian of the array of a given order finite difference
    scheme. Only operates on interior points, that is, points at least order/2
    points away from the edge of the array."""

    cdef int i
    cdef int j
    cdef int nx
    cdef int ny
    cdef int npts_edge = order // 2
    nx = psi.shape[0]
    ny = psi.shape[1]

    cdef double over_dx2 = 1/dx**2
    cdef double over_dy2 = 1/dy**2

    cdef double_or_complex Lx_ij
    cdef double_or_complex Ly_ij
    for i in range(npts_edge, nx - npts_edge):
        for j in range(npts_edge, ny - npts_edge):
            out[i, j] =  _laplacian_single_point_interior(i, j, psi, over_dx2, over_dy2, order, hollow)


cdef inline void complex_laplacian_interior(
    double complex [:, :] psi, double complex [:, :] out, double dx, double dy, int order, int hollow) nogil:
    _laplacian_interior(psi, out, dx, dy, order, hollow)

cdef inline void real_laplacian_interior(
     double [:, :] psi, double [:, :] out, double dx, double dy, int order, int hollow) nogil:
    _laplacian_interior(psi, out, dx, dy, order, hollow)

def laplacian_interior(psi, double dx, double dy):
    out = np.empty(psi.shape, dtype=psi.dtype)
    if psi.dtype == np.float64:
        real_laplacian_interior(psi, out, dx, dy, order=2, hollow=0)
    elif psi.dtype == np.complex128:
        complex_laplacian_interior(psi, out, dx, dy, order=2, hollow=0)
    return out


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline double_or_complex _grad2x_single_point_edges(
    int i, int j, int nx, double_or_complex [:, :] psi,
    double_or_complex [:, :] left_buffer, double_or_complex [:, :] right_buffer,
    double over_dx2, int order, int hollow) nogil:
    """Compute the second x derivative at a single point i, j near the edges
    of the array such that the buffers neighboring points are required. No
    bounds checking is performed so the caller must ensure that indices are in
    bounds. 1/dx^2 must be provided. If "hollow" is true, the central point is
    excluded from the calculation."""
    cdef double_or_complex Lx
    Lx = 0
    if order == 2:
        if not hollow:
            # Central point:
            Lx = D2_2ND_ORDER_0 * psi[i, j]
        # Nearest neighbors:
        if i == 0:
            Lx += D2_2ND_ORDER_1 * (left_buffer[0, j] + psi[i+1, j])
        elif i == nx - 1:
            Lx += D2_2ND_ORDER_1 * (psi[i-1, j] + right_buffer[0, j])
    elif order == 4:
        if not hollow:
            # Central point:
            Lx = D2_4TH_ORDER_0 * psi[i, j]
        if i == 0:
            # Nearest neighbor:
            Lx += D2_4TH_ORDER_1 * (left_buffer[1, j] + psi[i+1, j])
            # Next nearest neighbor:
            Lx += D2_4TH_ORDER_2 * (left_buffer[0, j] + psi[i+2, j])
        elif i == 1:
            # Nearest neighbor:
            Lx += D2_4TH_ORDER_1 * (psi[i-1, j] + psi[i+1, j])
            # Next nearest neighbor:
            Lx += D2_4TH_ORDER_2 * (left_buffer[1, j] + psi[i+2, j])
        elif i == nx - 2:
            # Nearest neighbor:
            Lx += D2_4TH_ORDER_1 * (psi[i-1, j] + psi[i+1, j])
            # Next nearest neighbor:
            Lx += D2_4TH_ORDER_2 * (psi[i-2, j] + right_buffer[0, j])
        elif i == nx - 1:
            # Nearest neighbor:
            Lx += D2_4TH_ORDER_1 * (psi[i-1, j] + right_buffer[0, j])
            # Next nearest neighbor:
            Lx += D2_4TH_ORDER_2 * (psi[i-2, j] + right_buffer[1, j])
    elif order == 6:
        if not hollow:
            # Central point:
            Lx = D2_6TH_ORDER_0 * psi[i, j]
        if i == 0:
            # Nearest neighbor:
            Lx += D2_6TH_ORDER_1 * (left_buffer[2, j] +  psi[i+1, j])
            # Next nearest neighbor:
            Lx += D2_6TH_ORDER_2 * (left_buffer[1, j] + psi[i+2, j])
            # Next next nearest neighbor:
            Lx += D2_6TH_ORDER_3 * (left_buffer[0, j] + psi[i+3, j])
        elif i == 1:
            # Nearest neighbor:
            Lx += D2_6TH_ORDER_1 * (psi[i-1, j] +  psi[i+1, j])
            # Next nearest neighbor:
            Lx += D2_6TH_ORDER_2 * (left_buffer[2, j] + psi[i+2, j])
            # Next next nearest neighbor:
            Lx += D2_6TH_ORDER_3 * (left_buffer[1, j] + psi[i+3, j])
        elif i == 2:
            # Nearest neighbor:
            Lx += D2_6TH_ORDER_1 * (psi[i-1, j] +  psi[i+1, j])
            # Next nearest neighbor:
            Lx += D2_6TH_ORDER_2 * (psi[i-2, j] + psi[i+2, j])
            # Next next nearest neighbor:
            Lx += D2_6TH_ORDER_3 * (left_buffer[0, j] + psi[i+3, j])
        elif i == nx - 3:
            # Nearest neighbor:
            Lx += D2_6TH_ORDER_1 * (psi[i-1, j] +  psi[i+1, j])
            # Next nearest neighbor:
            Lx += D2_6TH_ORDER_2 * (psi[i-2, j] + psi[i+2, j])
            # Next next nearest neighbor:
            Lx += D2_6TH_ORDER_3 * (psi[i-3, j] + right_buffer[0, j])
        elif i == nx - 2:
            # Nearest neighbor:
            Lx += D2_6TH_ORDER_1 * (psi[i-1, j] +  psi[i+1, j])
            # Next nearest neighbor:
            Lx += D2_6TH_ORDER_2 * (psi[i-2, j] + right_buffer[0, j])
            # Next next nearest neighbor:
            Lx += D2_6TH_ORDER_3 * (psi[i-3, j] + right_buffer[1, j])
        elif i == nx - 1:
            # Nearest neighbor:
            Lx += D2_6TH_ORDER_1 * (psi[i-1, j] +  right_buffer[0, j])
            # Next nearest neighbor:
            Lx += D2_6TH_ORDER_2 * (psi[i-2, j] + right_buffer[1, j])
            # Next next nearest neighbor:
            Lx += D2_6TH_ORDER_3 * (psi[i-3, j] + right_buffer[2, j])

    return Lx * over_dx2


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline double_or_complex _grad2y_single_point_edges(
    int i, int j, int ny, double_or_complex [:, :] psi,
    double_or_complex [:, :] bottom_buffer, double_or_complex [:, :] top_buffer,
    double over_dy2, int order, int hollow):
    """Compute the second x derivative at a single point i, j near the edges
    of the array such that the buffers neighboring points are required. No
    bounds checking is performed so the caller must ensure that indices are in
    bounds. 1/dx^2 must be provided. If "hollow" is true, the central point is
    excluded from the calculation."""
    cdef double_or_complex Ly
    Ly = 0
    if order == 2:
        if not hollow:
            # Central point:
            Ly = D2_2ND_ORDER_0 * psi[i, j]
        # Nearest neighbors:
        if j == 0:
            Ly += D2_2ND_ORDER_1 * (bottom_buffer[i, 0] + psi[i, j+1])
        elif j == ny - 1:
            Ly += D2_2ND_ORDER_1 * (psi[i, j-1] + top_buffer[i, 0])
    elif order == 4:
        if not hollow:
            # Central point:
            Ly = D2_4TH_ORDER_0 * psi[i, j]
        if j == 0:
            # Nearest neighbor:
            Ly += D2_4TH_ORDER_1 * (bottom_buffer[i, 1] + psi[i, j+1])
            # Next nearest neighbor:
            Ly += D2_4TH_ORDER_2 * (bottom_buffer[i, 0] + psi[i, j+2])
        elif j == 1:
            # Nearest neighbor:
            Ly += D2_4TH_ORDER_1 * (psi[i, j-1] + psi[i, j+1])
            # Next nearest neighbor:
            Ly += D2_4TH_ORDER_2 * (bottom_buffer[i, 1] + psi[i, j+2])
        elif j == ny - 2:
            # Nearest neighbor:
            Ly += D2_4TH_ORDER_1 * (psi[i, j-1] + psi[i, j+1])
            # Next nearest neighbor:
            Ly += D2_4TH_ORDER_2 * (psi[i, j-2] + top_buffer[i, 0])
        elif j == ny - 1:
            # Nearest neighbor:
            Ly += D2_4TH_ORDER_1 * (psi[i, j-1] + top_buffer[i, 0])
            # Next nearest neighbor:
            Ly += D2_4TH_ORDER_2 * (psi[i, j-2] + top_buffer[i, 1])
    elif order == 6:
        if not hollow:
            # Central point:
            Ly = D2_6TH_ORDER_0 * psi[i, j]
        if j == 0:
            # Nearest neighbor:
            Ly += D2_6TH_ORDER_1 * (bottom_buffer[i, 2] +  psi[i, j+1])
            # Next nearest neighbor:
            Ly += D2_6TH_ORDER_2 * (bottom_buffer[i, 1] + psi[i, j+2])
            # Next next nearest neighbor:
            Ly += D2_6TH_ORDER_3 * (bottom_buffer[i, 0] + psi[i, j+3])
        elif j == 1:
            # Nearest neighbor:
            Ly += D2_6TH_ORDER_1 * (psi[i, j-1] +  psi[i, j+1])
            # Next nearest neighbor:
            Ly += D2_6TH_ORDER_2 * (bottom_buffer[i, 2] + psi[i, j+2])
            # Next next nearest neighbor:
            Ly += D2_6TH_ORDER_3 * (bottom_buffer[i, 1] + psi[i, j+3])
        elif j == 2:
            # Nearest neighbor:
            Ly += D2_6TH_ORDER_1 * (psi[i, j-1] +  psi[i, j+1])
            # Next nearest neighbor:
            Ly += D2_6TH_ORDER_2 * (psi[i, j-2] + psi[i, j+2])
            # Next next nearest neighbor:
            Ly += D2_6TH_ORDER_3 * (bottom_buffer[i, 0] + psi[i, j+3])
        elif j == ny - 3:
            # Nearest neighbor:
            Ly += D2_6TH_ORDER_1 * (psi[i, j-1] +  psi[i, j+1])
            # Next nearest neighbor:
            Ly += D2_6TH_ORDER_2 * (psi[i, j-2] + psi[i, j+2])
            # Next next nearest neighbor:
            Ly += D2_6TH_ORDER_3 * (psi[i, j-3] + top_buffer[i, 0])
        elif j == ny - 2:
            # Nearest neighbor:
            Ly += D2_6TH_ORDER_1 * (psi[i, j-1] +  psi[i, j+1])
            # Next nearest neighbor:
            Ly += D2_6TH_ORDER_2 * (psi[i, j-2] + top_buffer[i, 0])
            # Next next nearest neighbor:
            Ly += D2_6TH_ORDER_3 * (psi[i, j-3] + top_buffer[i, 1])
        elif j == ny - 1:
            # Nearest neighbor:
            Ly += D2_6TH_ORDER_1 * (psi[i, j-1] +  top_buffer[i, 0])
            # Next nearest neighbor:
            Ly += D2_6TH_ORDER_2 * (psi[i, j-2] + top_buffer[i, 1])
            # Next next nearest neighbor:
            Ly += D2_6TH_ORDER_3 * (psi[i, j-3] + top_buffer[i, 2])
    else:
        print('invalid')

    return Ly * over_dy2

@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline double_or_complex _laplacian_single_point_edges(
    int i, int j, int nx, int ny, double_or_complex [:, :] psi,
    double_or_complex [:, :] left_buffer, double_or_complex [:, :] right_buffer,
    double_or_complex [:, :] bottom_buffer, double_or_complex [:, :] top_buffer,
    double over_dx2, double over_dy2, int order, int hollow):
    """Compute the Laplacian at a single point i, j. No bounds checking is
    performed, so one must be sure that i and j are at least order/2 away
    fromt the edges. 1/dx^2 and 1/dy^2 must be provided. If "hollow" is true,
    the central point is excluded from the calculation."""
    cdef double_or_complex Lx
    cdef double_or_complex Ly
    cdef int npts = order // 2
    if i < npts or i > nx - npts - 1:
        Lx = _grad2x_single_point_edges(i, j, nx, psi, left_buffer, right_buffer, over_dx2, order, hollow)
    else:
        Lx = _grad2x_single_point_interior(i, j, psi, over_dx2, order, hollow)
    if j < npts or j > ny - npts - 1:
        Ly = _grad2y_single_point_edges(i, j, ny, psi, bottom_buffer, top_buffer, over_dy2, order, hollow)
    else:
        Ly = _grad2y_single_point_interior(i, j, psi, over_dy2, order, hollow)
    return Lx + Ly



@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void _laplacian_edges(
     double_or_complex [:, :] psi, double_or_complex[:, :] out,
     double_or_complex [:, :] left_buffer, double_or_complex [:, :] right_buffer,
     double_or_complex [:, :] bottom_buffer, double_or_complex [:, :] top_buffer,
     double dx, double dy, int order, int hollow):
    """Compute the laplacian of the array of a given order finite difference
    scheme. Only operates on edge points, that is, points at most order/2
    points away from the edge of the array."""

    cdef int i
    cdef int j
    cdef int nx
    cdef int ny
    cdef int npts_edge = order // 2
    nx = psi.shape[0]
    ny = psi.shape[1]

    cdef double over_dx2 = 1/dx**2
    cdef double over_dy2 = 1/dy**2

    cdef double_or_complex Lx_ij
    cdef double_or_complex Ly_ij
    i = 0
    j = 0
    while True:
        out[i, j] =  _laplacian_single_point_edges(i, j, nx, ny, psi,
                                                   left_buffer, right_buffer, bottom_buffer, top_buffer,
                                                   over_dx2, over_dy2, order, hollow)
        iter_edges(&i, &j, nx, ny, order)
        if i > nx - 1 or j > ny - 1:
            break

cdef inline void complex_laplacian_edges(
    double complex [:, :] psi, double complex [:, :] out,
    double complex [:, :] left_buffer, double complex [:, :] right_buffer,
    double complex [:, :] bottom_buffer, double complex [:, :] top_buffer,
    double dx, double dy, int order, int hollow):
    _laplacian_edges(psi, out, left_buffer, right_buffer, bottom_buffer, top_buffer, dx, dy, order, hollow)

cdef inline void real_laplacian_edges(
     double [:, :] psi, double [:, :] out,
     double [:, :] left_buffer, double [:, :] right_buffer,
     double [:, :] bottom_buffer, double [:, :] top_buffer,
     double dx, double dy, int order, int hollow):
    _laplacian_edges(psi, out, left_buffer, right_buffer, bottom_buffer, top_buffer, dx, dy, order, hollow)

def laplacian_edges(psi, out, left_buffer, right_buffer, bottom_buffer, top_buffer, dx, dy):
    if psi.dtype == np.float64:
        real_laplacian_edges(psi, out, left_buffer, right_buffer, bottom_buffer, top_buffer, dx, dy, order=2, hollow=0)
    elif psi.dtype == np.complex128:
        complex_laplacian_edges(psi, out, left_buffer, right_buffer, bottom_buffer, top_buffer, dx, dy, order=2, hollow=0)
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


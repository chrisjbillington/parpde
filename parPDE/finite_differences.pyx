# cython: language_level=3

cimport cython
import numpy as np

ctypedef fused psi_t:
    double
    double complex

ctypedef fused A_diag_t:
    double
    double complex

ctypedef fused b_t:
    double
    double complex

ctypedef fused gradx_coeff_t:
    double
    double complex

ctypedef fused grady_coeff_t:
    double
    double complex

ctypedef fused grad2x_coeff_t:
    double
    double complex

ctypedef fused grad2y_coeff_t:
    double
    double complex

cdef extern from "complex.h":
    double cabs(double complex) nogil


# Constants for central finite differences:
DEF D_2ND_ORDER_1 = 1.0/2.0

DEF D_4TH_ORDER_1 = 2.0/3.0
DEF D_4TH_ORDER_2 = -1.0/12.0

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
cdef inline void iter_edges(int * i, int * j, int nx, int ny, int npts) nogil:
    """Increment i and j so as to iterate over the points within npts of the
    edge of the array. Caller should use i = j = 0 for the first point before
    calling this function to get the second point. Sets i = j = -1 when
    done."""
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
    elif i[0] < nx - 1:
        i[0] += 1
        j[0] = 0
    else:
        i[0] = -1
        j[0] = -1


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void iter_interior(int * i, int * j, int nx, int ny, int npts) nogil:
    """Increment i and j so as to iterate over the points interior, that is,
    at least npts_edges points the edge of the array. Caller should use i = j
    = npts for the first point before calling this function to get the second
    point. Sets i = j = -1 when done."""
    if j[0] < ny - npts - 1:
        j[0] += 1
    elif i[0] < nx - npts - 1:
        i[0] += 1
        j[0] = npts
    else:
        i[0] = -1
        j[0] = -1


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void iter_all(int * i, int * j, int nx, int ny, int npts) nogil:
    """Increment i and j so as to iterate over all points in the array. Caller
    should use i = j = 0 for the first point before calling this function to
    get the second point. Sets i = j = -1 when done."""
    if j[0] < ny - 1:
        j[0] += 1
    elif i[0] < nx - 1:
        i[0] += 1
        j[0] = 0
    else:
        i[0] = -1
        j[0] = -1


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline psi_t gradx_single_point(
    int i, int j, int nx, psi_t [:, :] psi, double over_dx, int order,
    int hollow, psi_t * diagonal, psi_t [:, :] left_buffer, psi_t [:, :] right_buffer) nogil:
    """Compute the first x derivative at a single point i, j. If "hollow" is
    true, the central point is excluded from the calculation, and 'diagonal'
    set to the operator's value there."""
    cdef psi_t Lx
    Lx = 0
    if hollow:
        diagonal[0] = 0
    if order == 2:
        if 0 < i < nx - 1:
            Lx += D_2ND_ORDER_1 * (psi[i+1, j] - psi[i-1, j])
        elif i == 0:
            Lx += D_2ND_ORDER_1 * (psi[i+1, j] - left_buffer[0, j])
        elif i == nx - 1:
            Lx += D_2ND_ORDER_1 * (right_buffer[0, j] - psi[i-1, j])
    elif order == 4:
        if 1 < i < nx - 2:
            Lx += D_4TH_ORDER_1 * (psi[i+1, j] - psi[i-1, j])
            Lx += D_4TH_ORDER_2 * (psi[i+2, j] - psi[i-2, j])
        elif i == 0:
            Lx += D_4TH_ORDER_1 * (psi[i+1, j] - left_buffer[1, j])
            Lx += D_4TH_ORDER_2 * (psi[i+2, j] - left_buffer[0, j])
        elif i == 1:
            Lx += D_4TH_ORDER_1 * (psi[i+1, j] - psi[i-1, j])
            Lx += D_4TH_ORDER_2 * (psi[i+2, j] - left_buffer[1, j])
        elif i == nx - 2:
            Lx += D_4TH_ORDER_1 * (psi[i+1, j] - psi[i-1, j])
            Lx += D_4TH_ORDER_2 * (right_buffer[0, j] - psi[i-2, j])
        elif i == nx - 1:
            Lx += D_4TH_ORDER_1 * (right_buffer[0, j] - psi[i-1, j])
            Lx += D_4TH_ORDER_2 * (right_buffer[1, j] - psi[i-2, j])
    elif order == 6:
        if 2 < i < nx - 3:
            Lx += D_6TH_ORDER_1 * (psi[i+1, j] - psi[i-1, j])
            Lx += D_6TH_ORDER_2 * (psi[i+2, j] - psi[i-2, j])
            Lx += D_6TH_ORDER_3 * (psi[i+3, j] - psi[i-3, j])
        elif i == 0:
            Lx += D_6TH_ORDER_1 * (psi[i+1, j] - left_buffer[2, j])
            Lx += D_6TH_ORDER_2 * (psi[i+2, j] - left_buffer[1, j])
            Lx += D_6TH_ORDER_3 * (psi[i+3, j] - left_buffer[0, j])
        elif i == 1:
            Lx += D_6TH_ORDER_1 * (psi[i+1, j] - psi[i-1, j])
            Lx += D_6TH_ORDER_2 * (psi[i+2, j] - left_buffer[2, j])
            Lx += D_6TH_ORDER_3 * (psi[i+3, j] - left_buffer[1, j])
        elif i == 2:
            Lx += D_6TH_ORDER_1 * (psi[i+1, j] - psi[i-1, j])
            Lx += D_6TH_ORDER_2 * (psi[i+2, j] - psi[i-2, j])
            Lx += D_6TH_ORDER_3 * (psi[i+3, j] - left_buffer[2, j])
        elif i == nx - 3:
            Lx += D_6TH_ORDER_1 * (psi[i+1, j] - psi[i-1, j])
            Lx += D_6TH_ORDER_2 * (psi[i+2, j] - psi[i-2, j])
            Lx += D_6TH_ORDER_3 * (right_buffer[0, j] - psi[i-3, j])
        elif i == nx - 2:
            Lx += D_6TH_ORDER_1 * (psi[i+1, j] - psi[i-1, j])
            Lx += D_6TH_ORDER_2 * (right_buffer[0, j] - psi[i-2, j])
            Lx += D_6TH_ORDER_3 * (right_buffer[1, j] - psi[i-3, j])
        elif i == nx - 1:
            Lx += D_6TH_ORDER_1 * (right_buffer[0, j] - psi[i-1, j])
            Lx += D_6TH_ORDER_2 * (right_buffer[1, j] - psi[i-2, j])
            Lx += D_6TH_ORDER_3 * (right_buffer[2, j] - psi[i-3, j])

    return Lx * over_dx


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline psi_t grad2x_single_point(
    int i, int j, int nx, psi_t [:, :] psi, double over_dx2, int order,
    int hollow, psi_t * diagonal, psi_t [:, :] left_buffer, psi_t [:, :] right_buffer) nogil:
    """Compute the second x derivative at a single point i, j. If "hollow" is
    true, the central point is excluded from the calculation, and 'diagonal'
    set to the operator's value there."""
    cdef psi_t Lx
    Lx = 0
    if order == 2:
        if hollow:
            diagonal[0] = D2_2ND_ORDER_0 * over_dx2
        else:
            Lx = D2_2ND_ORDER_0 * psi[i, j]
        if 0 < i < nx - 1:
            Lx += D2_2ND_ORDER_1 * (psi[i-1, j] + psi[i+1, j])
        elif i == 0:
            Lx += D2_2ND_ORDER_1 * (left_buffer[0, j] + psi[i+1, j])
        elif i == nx - 1:
            Lx += D2_2ND_ORDER_1 * (psi[i-1, j] + right_buffer[0, j])
    elif order == 4:
        if hollow:
            diagonal[0] = D2_4TH_ORDER_0 * over_dx2
        else:
            Lx = D2_4TH_ORDER_0 * psi[i, j]
        if 1 < i < nx - 2:
            Lx += D2_4TH_ORDER_1 * (psi[i-1, j] + psi[i+1, j])
            Lx += D2_4TH_ORDER_2 * (psi[i-2, j] + psi[i+2, j])
        elif i == 0:
            Lx += D2_4TH_ORDER_1 * (left_buffer[1, j] + psi[i+1, j])
            Lx += D2_4TH_ORDER_2 * (left_buffer[0, j] + psi[i+2, j])
        elif i == 1:
            Lx += D2_4TH_ORDER_1 * (psi[i-1, j] + psi[i+1, j])
            Lx += D2_4TH_ORDER_2 * (left_buffer[1, j] + psi[i+2, j])
        elif i == nx - 2:
            Lx += D2_4TH_ORDER_1 * (psi[i-1, j] + psi[i+1, j])
            Lx += D2_4TH_ORDER_2 * (psi[i-2, j] + right_buffer[0, j])
        elif i == nx - 1:
            Lx += D2_4TH_ORDER_1 * (psi[i-1, j] + right_buffer[0, j])
            Lx += D2_4TH_ORDER_2 * (psi[i-2, j] + right_buffer[1, j])
    elif order == 6:
        if hollow:
            diagonal[0] = D2_6TH_ORDER_0 * over_dx2
        else:
            Lx = D2_6TH_ORDER_0 * psi[i, j]
        if 2 < i < nx - 3:
            Lx += D2_6TH_ORDER_1 * (psi[i-1, j] +  psi[i+1, j])
            Lx += D2_6TH_ORDER_2 * (psi[i-2, j] + psi[i+2, j])
            Lx += D2_6TH_ORDER_3 * (psi[i-3, j] + psi[i+3, j])
        elif i == 0:
            Lx += D2_6TH_ORDER_1 * (left_buffer[2, j] +  psi[i+1, j])
            Lx += D2_6TH_ORDER_2 * (left_buffer[1, j] + psi[i+2, j])
            Lx += D2_6TH_ORDER_3 * (left_buffer[0, j] + psi[i+3, j])
        elif i == 1:
            Lx += D2_6TH_ORDER_1 * (psi[i-1, j] +  psi[i+1, j])
            Lx += D2_6TH_ORDER_2 * (left_buffer[2, j] + psi[i+2, j])
            Lx += D2_6TH_ORDER_3 * (left_buffer[1, j] + psi[i+3, j])
        elif i == 2:
            Lx += D2_6TH_ORDER_1 * (psi[i-1, j] +  psi[i+1, j])
            Lx += D2_6TH_ORDER_2 * (psi[i-2, j] + psi[i+2, j])
            Lx += D2_6TH_ORDER_3 * (left_buffer[2, j] + psi[i+3, j])
        elif i == nx - 3:
            Lx += D2_6TH_ORDER_1 * (psi[i-1, j] +  psi[i+1, j])
            Lx += D2_6TH_ORDER_2 * (psi[i-2, j] + psi[i+2, j])
            Lx += D2_6TH_ORDER_3 * (psi[i-3, j] + right_buffer[0, j])
        elif i == nx - 2:
            Lx += D2_6TH_ORDER_1 * (psi[i-1, j] +  psi[i+1, j])
            Lx += D2_6TH_ORDER_2 * (psi[i-2, j] + right_buffer[0, j])
            Lx += D2_6TH_ORDER_3 * (psi[i-3, j] + right_buffer[1, j])
        elif i == nx - 1:
            Lx += D2_6TH_ORDER_1 * (psi[i-1, j] +  right_buffer[0, j])
            Lx += D2_6TH_ORDER_2 * (psi[i-2, j] + right_buffer[1, j])
            Lx += D2_6TH_ORDER_3 * (psi[i-3, j] + right_buffer[2, j])

    return Lx * over_dx2


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline psi_t grady_single_point(
    int i, int j, int ny, psi_t [:, :] psi, double over_dy, int order,
    int hollow, psi_t * diagonal, psi_t [:, :] bottom_buffer, psi_t [:, :] top_buffer) nogil:
    """Compute the second y derivative at a single point i, j. If "hollow" is
    true, the central point is excluded from the calculation, and 'diagonal'
    set to the operator's value there."""
    cdef psi_t Ly
    Ly = 0
    if hollow:
        diagonal[0] = 0
    if order == 2:
        if 0 < j < ny - 1:
            Ly += D_2ND_ORDER_1 * (psi[i, j+1] - psi[i, j-1])
        elif j == 0:
            Ly += D_2ND_ORDER_1 * (psi[i, j+1] - bottom_buffer[i, 0])
        elif j == ny - 1:
            Ly += D_2ND_ORDER_1 * (top_buffer[i, 0] - psi[i, j-1])
    elif order == 4:
        if 1 < j < ny - 2:
            Ly += D_4TH_ORDER_1 * (psi[i, j+1] - psi[i, j-1])
            Ly += D_4TH_ORDER_2 * (psi[i, j+2] - psi[i, j-2])
        elif j == 0:
            Ly += D_4TH_ORDER_1 * (psi[i, j+1] - bottom_buffer[i, 1])
            Ly += D_4TH_ORDER_2 * (psi[i, j+2] - bottom_buffer[i, 0])
        elif j == 1:
            Ly += D_4TH_ORDER_1 * (psi[i, j+1] - psi[i, j-1])
            Ly += D_4TH_ORDER_2 * (psi[i, j+2] - bottom_buffer[i, 1])
        elif j == ny - 2:
            Ly += D_4TH_ORDER_1 * (psi[i, j+1] - psi[i, j-1])
            Ly += D_4TH_ORDER_2 * (top_buffer[i, 0] - psi[i, j-2])
        elif j == ny - 1:
            Ly += D_4TH_ORDER_1 * (top_buffer[i, 0] - psi[i, j-1])
            Ly += D_4TH_ORDER_2 * (top_buffer[i, 1] - psi[i, j-2])
    elif order == 6:
        if 2 < j < ny - 3:
            Ly += D_6TH_ORDER_1 * (psi[i, j+1] - psi[i, j-1])
            Ly += D_6TH_ORDER_2 * (psi[i, j+2] - psi[i, j-2])
            Ly += D_6TH_ORDER_3 * (psi[i, j+3] - psi[i, j-3])
        if j == 0:
            Ly += D_6TH_ORDER_1 * (psi[i, j+1] - bottom_buffer[i, 2])
            Ly += D_6TH_ORDER_2 * (psi[i, j+2] - bottom_buffer[i, 1])
            Ly += D_6TH_ORDER_3 * (psi[i, j+3] - bottom_buffer[i, 0])
        elif j == 1:
            Ly += D_6TH_ORDER_1 * (psi[i, j+1] - psi[i, j-1])
            Ly += D_6TH_ORDER_2 * (psi[i, j+2] - bottom_buffer[i, 2])
            Ly += D_6TH_ORDER_3 * (psi[i, j+3] - bottom_buffer[i, 1])
        elif j == 2:
            Ly += D_6TH_ORDER_1 * (psi[i, j+1] - psi[i, j-1])
            Ly += D_6TH_ORDER_2 * (psi[i, j+2] - psi[i, j-2])
            Ly += D_6TH_ORDER_3 * (psi[i, j+3] - bottom_buffer[i, 2])
        elif j == ny - 3:
            Ly += D_6TH_ORDER_1 * (psi[i, j+1] - psi[i, j-1])
            Ly += D_6TH_ORDER_2 * (psi[i, j+2] - psi[i, j-2])
            Ly += D_6TH_ORDER_3 * (top_buffer[i, 0] - psi[i, j-3])
        elif j == ny - 2:
            Ly += D_6TH_ORDER_1 * (psi[i, j+1] - psi[i, j-1])
            Ly += D_6TH_ORDER_2 * (top_buffer[i, 0] - psi[i, j-2])
            Ly += D_6TH_ORDER_3 * (top_buffer[i, 1] - psi[i, j-3])
        elif j == ny - 1:
            Ly += D_6TH_ORDER_1 * (top_buffer[i, 0] - psi[i, j-1])
            Ly += D_6TH_ORDER_2 * (top_buffer[i, 1] - psi[i, j-2])
            Ly += D_6TH_ORDER_3 * (top_buffer[i, 2] - psi[i, j-3])

    return Ly * over_dy


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline psi_t grad2y_single_point(
    int i, int j, int ny, psi_t [:, :] psi, double over_dy2, int order,
    int hollow, psi_t * diagonal, psi_t [:, :] bottom_buffer, psi_t [:, :] top_buffer) nogil:
    """Compute the second y derivative at a single point i, j. If "hollow" is
    true, the central point is excluded from the calculation, and 'diagonal'
    set to the operator's value there."""
    cdef psi_t Ly
    Ly = 0
    if order == 2:
        if hollow:
            diagonal[0] = D2_2ND_ORDER_0 * over_dy2
        else:
            Ly = D2_2ND_ORDER_0 * psi[i, j]
        if 0 < j < ny - 1:
            Ly += D2_2ND_ORDER_1 * (psi[i, j-1] + psi[i, j+1])
        elif j == 0:
            Ly += D2_2ND_ORDER_1 * (bottom_buffer[i, 0] + psi[i, j+1])
        elif j == ny - 1:
            Ly += D2_2ND_ORDER_1 * (psi[i, j-1] + top_buffer[i, 0])
    elif order == 4:
        if hollow:
            diagonal[0] = D2_4TH_ORDER_0 * over_dy2
        else:
            Ly = D2_4TH_ORDER_0 * psi[i, j]
        if 1 < j < ny - 2:
            Ly += D2_4TH_ORDER_1 * (psi[i, j-1] + psi[i, j+1])
            Ly += D2_4TH_ORDER_2 * (psi[i, j-2] + psi[i, j+2])
        elif j == 0:
            Ly += D2_4TH_ORDER_1 * (bottom_buffer[i, 1] + psi[i, j+1])
            Ly += D2_4TH_ORDER_2 * (bottom_buffer[i, 0] + psi[i, j+2])
        elif j == 1:
            Ly += D2_4TH_ORDER_1 * (psi[i, j-1] + psi[i, j+1])
            Ly += D2_4TH_ORDER_2 * (bottom_buffer[i, 1] + psi[i, j+2])
        elif j == ny - 2:
            Ly += D2_4TH_ORDER_1 * (psi[i, j-1] + psi[i, j+1])
            Ly += D2_4TH_ORDER_2 * (psi[i, j-2] + top_buffer[i, 0])
        elif j == ny - 1:
            Ly += D2_4TH_ORDER_1 * (psi[i, j-1] + top_buffer[i, 0])
            Ly += D2_4TH_ORDER_2 * (psi[i, j-2] + top_buffer[i, 1])
    elif order == 6:
        if hollow:
            diagonal[0] = D2_6TH_ORDER_0 * over_dy2
        else:
            Ly = D2_6TH_ORDER_0 * psi[i, j]
        if 2 < j < ny - 3:
            Ly += D2_6TH_ORDER_1 * (psi[i, j-1] +  psi[i, j+1])
            Ly += D2_6TH_ORDER_2 * (psi[i, j-2] + psi[i, j+2])
            Ly += D2_6TH_ORDER_3 * (psi[i, j-3] + psi[i, j+3])
        if j == 0:
            Ly += D2_6TH_ORDER_1 * (bottom_buffer[i, 2] +  psi[i, j+1])
            Ly += D2_6TH_ORDER_2 * (bottom_buffer[i, 1] + psi[i, j+2])
            Ly += D2_6TH_ORDER_3 * (bottom_buffer[i, 0] + psi[i, j+3])
        elif j == 1:
            Ly += D2_6TH_ORDER_1 * (psi[i, j-1] +  psi[i, j+1])
            Ly += D2_6TH_ORDER_2 * (bottom_buffer[i, 2] + psi[i, j+2])
            Ly += D2_6TH_ORDER_3 * (bottom_buffer[i, 1] + psi[i, j+3])
        elif j == 2:
            Ly += D2_6TH_ORDER_1 * (psi[i, j-1] +  psi[i, j+1])
            Ly += D2_6TH_ORDER_2 * (psi[i, j-2] + psi[i, j+2])
            Ly += D2_6TH_ORDER_3 * (bottom_buffer[i, 2] + psi[i, j+3])
        elif j == ny - 3:
            Ly += D2_6TH_ORDER_1 * (psi[i, j-1] +  psi[i, j+1])
            Ly += D2_6TH_ORDER_2 * (psi[i, j-2] + psi[i, j+2])
            Ly += D2_6TH_ORDER_3 * (psi[i, j-3] + top_buffer[i, 0])
        elif j == ny - 2:
            Ly += D2_6TH_ORDER_1 * (psi[i, j-1] +  psi[i, j+1])
            Ly += D2_6TH_ORDER_2 * (psi[i, j-2] + top_buffer[i, 0])
            Ly += D2_6TH_ORDER_3 * (psi[i, j-3] + top_buffer[i, 1])
        elif j == ny - 1:
            Ly += D2_6TH_ORDER_1 * (psi[i, j-1] +  top_buffer[i, 0])
            Ly += D2_6TH_ORDER_2 * (psi[i, j-2] + top_buffer[i, 1])
            Ly += D2_6TH_ORDER_3 * (psi[i, j-3] + top_buffer[i, 2])

    return Ly * over_dy2


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline psi_t operators_single_point(
    int i, int j, int nx, int ny, psi_t [:, :] psi,
    gradx_coeff_t [:, :] gradx_coeff, grady_coeff_t [:, :] grady_coeff,
    grad2x_coeff_t [:, :] grad2x_coeff, grad2y_coeff_t [:, :] grad2y_coeff,
    double over_dx2, double over_dy2, double over_dx, double over_dy, int order, int hollow, psi_t * diagonal,
    psi_t [:, :] left_buffer, psi_t [:, :] right_buffer, psi_t [:, :] bottom_buffer, psi_t [:, :] top_buffer) nogil:
    # Do not generate code for invalid types:
    if psi_t is cython.double and (gradx_coeff_t is cython.complex or
                                   grady_coeff_t is cython.complex or
                                   grad2x_coeff_t is cython.complex or
                                   grad2y_coeff_t is cython.complex):
        return 0
    cdef psi_t L_psi = 0
    cdef psi_t diagonal_temp = 0
    cdef int i_prime
    cdef int j_prime
    if hollow:
        diagonal[0] = 0

    if gradx_coeff is not None:
        if gradx_coeff.shape[0] > 1:
            i_prime = i
        else:
            i_prime = 0
        if gradx_coeff.shape[1] > 1:
            j_prime = j
        else:
            j_prime = 0
        L_psi += gradx_coeff[i_prime, j_prime] * gradx_single_point(i, j, nx, psi, over_dx, order, hollow,
                                                                    &diagonal_temp, left_buffer, right_buffer)
        if hollow:
            diagonal[0] += gradx_coeff[i_prime, j_prime] * diagonal_temp

    if grady_coeff is not None:
        if grady_coeff.shape[0] > 1:
            i_prime = i
        else:
            i_prime = 0
        if grady_coeff.shape[1] > 1:
            j_prime = j
        else:
            j_prime = 0
        L_psi += grady_coeff[i_prime, j_prime] * grady_single_point(i, j, ny, psi, over_dy, order, hollow,
                                                                    &diagonal_temp, bottom_buffer, top_buffer)
        if hollow:
            diagonal[0] += grady_coeff[i_prime, j_prime] * diagonal_temp

    if grad2x_coeff is not None:
        if grad2x_coeff.shape[0] > 1:
            i_prime = i
        else:
            i_prime = 0
        if grad2x_coeff.shape[1] > 1:
            j_prime = j
        else:
            j_prime = 0
        L_psi += grad2x_coeff[i_prime, j_prime] * grad2x_single_point(i, j, nx, psi, over_dx2, order, hollow,
                                                                      &diagonal_temp, left_buffer, right_buffer)
        if hollow:
            diagonal[0] += grad2x_coeff[i_prime, j_prime] * diagonal_temp

    if grad2y_coeff is not None:
        if grad2y_coeff.shape[0] > 1:
            i_prime = i
        else:
            i_prime = 0
        if grad2y_coeff.shape[1] > 1:
            j_prime = j
        else:
            j_prime = 0
        L_psi += grad2y_coeff[i_prime, j_prime] * grad2y_single_point(i, j, ny, psi, over_dy2, order, hollow,
                                                                      &diagonal_temp, bottom_buffer, top_buffer)
        if hollow:
            diagonal[0] += grad2y_coeff[i_prime, j_prime] * diagonal_temp

    return L_psi

@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def SOR_step(
    psi_t [:, :] psi, A_diag_t [:, :] A_diag,
    gradx_coeff_t [:, :] gradx_coeff, grady_coeff_t [:, :] grady_coeff,
    grad2x_coeff_t [:, :] grad2x_coeff, grad2y_coeff_t [:, :] grad2y_coeff,
    b_t [:, :] b, double dx, double dy, double relaxation_parameter, int operator_order,
    psi_t [:, :] left_buffer=None, psi_t [:, :] right_buffer=None,
    psi_t [:, :] bottom_buffer=None, psi_t [:, :] top_buffer=None,
    double squared_error=0, int interior=1, int edges=1, char [:, :] boundary_mask=None):

    # Invalid types:
    if psi_t is cython.double and (A_diag_t is cython.complex or
                                   b_t is cython.complex or
                                   gradx_coeff_t is cython.complex or
                                   grady_coeff_t is cython.complex or
                                   grad2x_coeff_t is cython.complex or
                                   grad2y_coeff_t is cython.complex):
        raise TypeError("Cannot use complex arrays when psi is real")

    cdef int i
    cdef int j
    cdef int nx = psi.shape[0]
    cdef int ny = psi.shape[1]

    cdef double over_dx2 = 1/dx**2
    cdef double over_dy2 = 1/dy**2
    cdef double over_dx = 1/dx
    cdef double over_dy = 1/dy

    cdef int npts_edge = operator_order // 2

    cdef psi_t operator_diag
    cdef psi_t hollow_operator_result
    cdef psi_t A_hollow_psi
    cdef psi_t A_diag_total
    cdef psi_t psi_GS

    if edges:
        # If doing edges and interior, or only edges, we start in the corner:
        i = 0
        j = 0
    elif interior:
        # If doing only interior, we start away from the edge:
        i = npts_edge
        j = npts_edge
    else:
        msg = "Must choose either edges, interior or both"
        raise ValueError(msg)

    # We loop over psi with a while loop, incrementing i and j each
    # loop depending on the region being computed, and breaking when done.
    while True:
        if boundary_mask is None or boundary_mask[i, j]:
            # Compute the result of A*psi excluding the diagonals:
            A_hollow_psi = operators_single_point(
                               i, j, nx, ny, psi, gradx_coeff, grady_coeff, grad2x_coeff, grad2y_coeff,
                               over_dx2, over_dy2, over_dx, over_dy, operator_order, 1, &operator_diag,
                               left_buffer, right_buffer, bottom_buffer, top_buffer)
            # Compute the total diagonals of A. This is the sum of the
            # diagonal operator given by the user, and the diagonals of the
            # non-diagonal operators:
            A_diag_total = A_diag[i, j] + operator_diag

            # Record the squared residuals for error checking:
            squared_error += cabs(A_hollow_psi + A_diag_total*psi[i, j] - b[i, j])**2

            # The Gauss-Seidel prediction for psi at this point:
            psi_GS = (b[i, j] - A_hollow_psi)/A_diag_total

            # Update psi with overrelaxation at this point:
            psi[i, j] = psi[i, j] + relaxation_parameter*(psi_GS - psi[i, j])

        if edges and not interior:
            iter_edges(&i, &j, nx, ny, npts_edge)
        elif edges:
            iter_all(&i, &j, nx, ny, npts_edge)
        elif interior:
            iter_interior(&i, &j, nx, ny, npts_edge)
        if i == -1:
            break

    return float(squared_error)


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def apply_operator(
    psi_t [:, :] psi,
    gradx_coeff_t [:, :] gradx_coeff, grady_coeff_t [:, :] grady_coeff,
    grad2x_coeff_t [:, :] grad2x_coeff, grad2y_coeff_t [:, :] grad2y_coeff,
    double dx, double dy, int operator_order, psi_t [:, :] out=None,
    psi_t [:, :] left_buffer=None, psi_t [:, :] right_buffer=None,
    psi_t [:, :] bottom_buffer=None, psi_t [:, :] top_buffer=None,
    int interior=1, int edges=1):

    # Invalid types:
    if psi_t is cython.double and (gradx_coeff_t is cython.complex or
                                   grady_coeff_t is cython.complex or
                                   grad2x_coeff_t is cython.complex or
                                   grad2y_coeff_t is cython.complex):
        raise TypeError("Cannot use complex arrays when psi is real")

    cdef int i
    cdef int j
    cdef int nx = psi.shape[0]
    cdef int ny = psi.shape[1]

    cdef double over_dx2 = 1/dx**2
    cdef double over_dy2 = 1/dy**2
    cdef double over_dx = 1/dx
    cdef double over_dy = 1/dy
    cdef int npts_edge = operator_order // 2

    if out is None:
        if psi_t is cython.complex:
            out = np.zeros((nx, ny), dtype=np.complex128)
        else:
            out = np.zeros((nx, ny), dtype=np.float64)
    if edges:
        # If doing edges and interior, or only edges, we start in the corner:
        i = 0
        j = 0
    elif interior:
        # If doing only interior, we start away from the edge:
        i = npts_edge
        j = npts_edge
    else:
        msg = "Must choose either edges, interior or both"
        raise ValueError(msg)

    # We loop over our psi with a while loop, incrementing i and j each loop
    # depending on the region being computed, and breaking when done.
    while True:
        # Compute the result of the operators on psi at this point:
        out[i, j] = operators_single_point(i, j, nx, ny, psi, gradx_coeff, grady_coeff, grad2x_coeff, grad2y_coeff,
                                           over_dx2, over_dy2, over_dx, over_dy, operator_order, 0, NULL,
                                           left_buffer, right_buffer, bottom_buffer, top_buffer)
        if edges and not interior:
            iter_edges(&i, &j, nx, ny, npts_edge)
        elif edges:
            iter_all(&i, &j, nx, ny, npts_edge)
        elif interior:
            iter_interior(&i, &j, nx, ny, npts_edge)
        if i == -1:
            break

    return np.asarray(out)



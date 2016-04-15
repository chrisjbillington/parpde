from __future__ import print_function


cimport cython
import numpy as np

cdef enum Operator:
    GRADX = 0
    GRADY = 1
    GRAD2X = 2
    GRAD2Y = 3
    LAPLACIAN = 4


ZEROS_REAL_2D = np.zeros((1, 1), dtype=np.float64)
ZEROS_COMPLEX_2D = np.zeros((1, 1), dtype=np.complex128)


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
    the edge of the array. Caller is responsible for not calling any more after
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
cdef inline object process_operators(
    psi, A_nondiag, int * use_gradx, int * use_grady,int * use_grad2x, int * use_grad2y, int * use_laplacian):
    """Turn the dictionary of operators and their coefficient arrays A_nondiag into individual ints saying whether or
    not to use each operator, and coefficient arrays of the same datatype as psi"""
    if psi.dtype == np.float64:
        default_coeff = ZEROS_REAL_2D
    elif psi.dtype == np.float128:
        default_coeff = ZEROS_COMPLEX_2D
    else:
        raise TypeError(psi.dtype)
    use_gradx[0] = GRADX in A_nondiag
    use_grady[0] = GRADY in A_nondiag
    use_grad2x[0] = GRAD2X in A_nondiag
    use_grad2y[0] = GRAD2Y in A_nondiag
    use_laplacian[0] = LAPLACIAN in A_nondiag

    gradx_coeff = A_nondiag.get(GRADX, default_coeff)
    grady_coeff = A_nondiag.get(GRADY, default_coeff)
    grad2x_coeff = A_nondiag.get(GRAD2X, default_coeff)
    grad2y_coeff = A_nondiag.get(GRAD2Y, default_coeff)
    laplacian_coeff = A_nondiag.get(LAPLACIAN, default_coeff)

    if use_gradx[0] and gradx_coeff.dtype != psi.dtype:
        gradx_coeff = np.array(gradx_coeff, dtype=psi.dtype)
    if use_grady[0] and grady_coeff.dtype != psi.dtype:
        grady_coeff = np.array(grady_coeff, dtype=psi.dtype)
    if use_grad2x[0] and grad2x_coeff.dtype != psi.dtype:
        grad2x_coeff = np.array(grad2x_coeff, dtype=psi.dtype)
    if use_grad2y[0] and grad2y_coeff.dtype != psi.dtype:
        grad2y_coeff = np.array(grad2y_coeff, dtype=psi.dtype)
    if use_laplacian[0] and laplacian_coeff.dtype != psi.dtype:
        laplacian_coeff = np.array(laplacian_coeff, dtype=psi.dtype)

    return gradx_coeff, grady_coeff, grad2x_coeff, grad2y_coeff, laplacian_coeff


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline double_or_complex _grad2x_single_point_interior(
    int i, int j, double_or_complex [:, :] psi, double over_dx2, int order,
    int hollow, double_or_complex * diagonal) nogil:
    """Compute the second x derivative at a single point i, j. No bounds
    checking is performed, so one must be sure that i is at least order/2 away
    fromt the edges. 1/dx^2 must be provided. If "hollow" is true, the central
    point is excluded from the calculation, and 'diagonal' set to the
    operator's value there."""
    cdef double_or_complex Lx
    Lx = 0
    if order == 2:
        if hollow:
            diagonal[0] = D2_2ND_ORDER_0 * over_dx2
        else:
            # Central point:
            Lx = D2_2ND_ORDER_0 * psi[i, j]
        # Nearest neighbor:
        Lx += D2_2ND_ORDER_1 * (psi[i-1, j] + psi[i+1, j])
    elif order == 4:
        if hollow:
            diagonal[0] = D2_4TH_ORDER_0 * over_dx2
        else:
            # Central point:
            Lx = D2_4TH_ORDER_0 * psi[i, j]
        # Nearest neighbor:
        Lx += D2_4TH_ORDER_1 * (psi[i-1, j] + psi[i+1, j])
        # Next nearest neighbor:
        Lx += D2_4TH_ORDER_2 * (psi[i-2, j] + psi[i+2, j])
    elif order == 6:
        if hollow:
            diagonal[0] = D2_6TH_ORDER_0 * over_dx2
        else:
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
    int i, int j, double_or_complex [:, :] psi, double over_dy2, int order,
    int hollow, double_or_complex * diagonal) nogil:
    """Compute the second y derivative at a single point i, j. No bounds
    checking is performed, so one must be sure that j is at least order/2 away
    fromt the edges. 1/dy^2 must be provided. If "hollow" is true, the central
    point is excluded from the calculation, and 'diagonal' set to the
    operator's value there."""
    cdef double_or_complex Ly
    Ly = 0
    if order == 2:
        if hollow:
            diagonal[0] = D2_2ND_ORDER_0 * over_dy2
        else:
            # Central point:
            Ly = D2_2ND_ORDER_0 * psi[i, j]
        # Nearest neighbor:
        Ly += D2_2ND_ORDER_1 * (psi[i, j-1] + psi[i, j+1])
    elif order == 4:
        if hollow:
            diagonal[0] = D2_4TH_ORDER_0 * over_dy2
        else:
            # Central point:
            Ly = D2_4TH_ORDER_0 * psi[i, j]
        # Nearest neighbor:
        Ly += D2_4TH_ORDER_1 * (psi[i, j-1] + psi[i, j+1])
        # Next nearest neighbor:
        Ly += D2_4TH_ORDER_2 * (psi[i, j-2] + psi[i, j+2])
    elif order == 6:
        if hollow:
            diagonal[0] = D2_6TH_ORDER_0 * over_dy2
        else:
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
    int i, int j, double_or_complex [:, :] psi, double over_dx2, double over_dy2, int order,
    int hollow, double_or_complex * diagonal) nogil:
    """Compute the Laplacian at a single point i, j. No bounds checking is
    performed, so one must be sure that i and j are at least order/2 away
    fromt the edges. 1/dx^2 and 1/dy^2 must be provided. If "hollow" is true,
    the central point is excluded from the calculation, and 'diagonal'
    set to the operator's value there."""
    cdef double_or_complex Lx
    cdef double_or_complex Ly
    cdef double_or_complex diagonal_temp = 0
    Lx = _grad2x_single_point_interior(i, j, psi, over_dx2, order, hollow, &diagonal_temp)
    if hollow:
        diagonal[0] = diagonal_temp
    Ly = _grad2y_single_point_interior(i, j, psi, over_dy2, order, hollow, &diagonal_temp)
    if hollow:
        diagonal[0] += diagonal_temp
    return (Lx + Ly)


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void _laplacian_interior(
     double_or_complex [:, :] psi, double_or_complex [:, :] out, double dx, double dy, int order) nogil:
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
            out[i, j] =  _laplacian_single_point_interior(i, j, psi, over_dx2, over_dy2, order, 0, NULL)


cdef inline void complex_laplacian_interior(
    double complex [:, :] psi, double complex [:, :] out, double dx, double dy, int order) nogil:
    _laplacian_interior(psi, out, dx, dy, order)

cdef inline void real_laplacian_interior(
     double [:, :] psi, double [:, :] out, double dx, double dy, int order) nogil:
    _laplacian_interior(psi, out, dx, dy, order)

def laplacian_interior(psi, double dx, double dy):
    out = np.empty(psi.shape, dtype=psi.dtype)
    if psi.dtype == np.float64:
        real_laplacian_interior(psi, out, dx, dy, order=2)
    elif psi.dtype == np.complex128:
        complex_laplacian_interior(psi, out, dx, dy, order=2)
    return out


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline double_or_complex _grad2x_single_point_edges(
    int i, int j, int nx, double_or_complex [:, :] psi,
    double_or_complex [:, :] left_buffer, double_or_complex [:, :] right_buffer,
    double over_dx2, int order, int hollow, double_or_complex * diagonal) nogil:
    """Compute the second x derivative at a single point i, j near the edges
    of the array such that the buffers neighboring points are required. No
    bounds checking is performed so the caller must ensure that indices are in
    bounds. 1/dx^2 must be provided. If "hollow" is true, the central point is
    excluded from the calculation, and 'diagonal' set to the
    operator's value there."""
    cdef double_or_complex Lx
    Lx = 0
    if order == 2:
        if hollow:
            diagonal[0] = D2_2ND_ORDER_0 * over_dx2
        else:
            # Central point:
            Lx = D2_2ND_ORDER_0 * psi[i, j]
        # Nearest neighbors:
        if i == 0:
            Lx += D2_2ND_ORDER_1 * (left_buffer[0, j] + psi[i+1, j])
        elif i == nx - 1:
            Lx += D2_2ND_ORDER_1 * (psi[i-1, j] + right_buffer[0, j])
    elif order == 4:
        if hollow:
            diagonal[0] = D2_4TH_ORDER_0 * over_dx2
        else:
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
        if hollow:
            diagonal[0] = D2_6TH_ORDER_0 * over_dx2
        else:
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
    double over_dy2, int order, int hollow, double_or_complex * diagonal) nogil:
    """Compute the second x derivative at a single point i, j near the edges
    of the array such that the buffers neighboring points are required. No
    bounds checking is performed so the caller must ensure that indices are in
    bounds. 1/dx^2 must be provided. If "hollow" is true, the central point is
    excluded from the calculation, and 'diagonal' set to the
    operator's value there."""
    cdef double_or_complex Ly
    Ly = 0
    if order == 2:
        if hollow:
            diagonal[0] = D2_2ND_ORDER_0 * over_dy2
        else:
            # Central point:
            Ly = D2_2ND_ORDER_0 * psi[i, j]
        # Nearest neighbors:
        if j == 0:
            Ly += D2_2ND_ORDER_1 * (bottom_buffer[i, 0] + psi[i, j+1])
        elif j == ny - 1:
            Ly += D2_2ND_ORDER_1 * (psi[i, j-1] + top_buffer[i, 0])
    elif order == 4:
        if hollow:
            diagonal[0] = D2_4TH_ORDER_0 * over_dy2
        else:
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
        if hollow:
            diagonal[0] = D2_6TH_ORDER_0 * over_dy2
        else:
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
    double over_dx2, double over_dy2, int order, int hollow, double_or_complex * diagonal) nogil:
    """Compute the Laplacian at a single point i, j. No bounds checking is
    performed, so one must be sure that i and j are at least order/2 away
    fromt the edges. 1/dx^2 and 1/dy^2 must be provided. If "hollow" is true,
    the central point is excluded from the calculation, and 'diagonal'
    set to the operator's value there."""
    cdef double_or_complex Lx
    cdef double_or_complex Ly
    cdef double_or_complex diagonal_temp = 0
    cdef int npts = order // 2
    if i < npts or i > nx - npts - 1:
        Lx = _grad2x_single_point_edges(i, j, nx, psi, left_buffer, right_buffer, over_dx2, order, hollow, &diagonal_temp)
    else:
        Lx = _grad2x_single_point_interior(i, j, psi, over_dx2, order, hollow, &diagonal_temp)
    if hollow:
        diagonal[0] = diagonal_temp
    if j < npts or j > ny - npts - 1:
        Ly = _grad2y_single_point_edges(i, j, ny, psi, bottom_buffer, top_buffer, over_dy2, order, hollow, &diagonal_temp)
    else:
        Ly = _grad2y_single_point_interior(i, j, psi, over_dy2, order, hollow, &diagonal_temp)
    if hollow:
        diagonal[0] += diagonal_temp
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
     double dx, double dy, int order):
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
                                                   over_dx2, over_dy2, order, 0, NULL)
        iter_edges(&i, &j, nx, ny, order)
        if i > nx - 1 or j > ny - 1:
            break

cdef inline void complex_laplacian_edges(
    double complex [:, :] psi, double complex [:, :] out,
    double complex [:, :] left_buffer, double complex [:, :] right_buffer,
    double complex [:, :] bottom_buffer, double complex [:, :] top_buffer,
    double dx, double dy, int order):
    _laplacian_edges(psi, out, left_buffer, right_buffer, bottom_buffer, top_buffer, dx, dy, order)

cdef inline void real_laplacian_edges(
     double [:, :] psi, double [:, :] out,
     double [:, :] left_buffer, double [:, :] right_buffer,
     double [:, :] bottom_buffer, double [:, :] top_buffer,
     double dx, double dy, int order):
    _laplacian_edges(psi, out, left_buffer, right_buffer, bottom_buffer, top_buffer, dx, dy, order)

def laplacian_edges(psi, out, left_buffer, right_buffer, bottom_buffer, top_buffer, dx, dy):
    if psi.dtype == np.float64:
        real_laplacian_edges(psi, out, left_buffer, right_buffer, bottom_buffer, top_buffer, dx, dy, order=2)
    elif psi.dtype == np.complex128:
        complex_laplacian_edges(psi, out, left_buffer, right_buffer, bottom_buffer, top_buffer, dx, dy, order=2)
    return out


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void _SOR_step_interior(double_or_complex [:, :] psi, double_or_complex [:, :] A_diag,
                                    double_or_complex [:, :] b, double dx, double dy, double relaxation_parameter,
                                    int use_gradx, double_or_complex [:, :] gradx_coeff,
                                    int use_grady, double_or_complex [:, :] grady_coeff,
                                    int use_grad2x, double_or_complex [:, :] grad2x_coeff,
                                    int use_grad2y, double_or_complex [:, :] grad2y_coeff,
                                    int use_laplacian, double_or_complex [:, :] laplacian_coeff,
                                    int operator_order, double * squared_error_ptr, int compute_error) nogil:
    cdef int i
    cdef int j
    cdef int i_prime = 0
    cdef int j_prime = 0
    cdef int nx
    cdef int ny
    nx = psi.shape[0]
    ny = psi.shape[1]

    cdef int laplacian_coeff_has_x = laplacian_coeff.shape[0] > 1
    cdef int laplacian_coeff_has_y = laplacian_coeff.shape[1] > 1

    cdef double over_dx2 = 1/dx**2
    cdef double over_dy2 = 1/dy**2
    cdef double residual

    cdef double_or_complex operator_coeff
    cdef double_or_complex operator_diag
    cdef double_or_complex hollow_operator_result
    cdef double_or_complex A_hollow_psi
    cdef double_or_complex A_diag_total
    cdef double_or_complex psi_GS

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            # Compute the total diagonals of A. This is the sum of the
            # diagonal operator given by the user, and the diagonals of the
            # non-diagonal operators:
            A_diag_total = A_diag[i, j]
            # Compute the result of A*psi excluding the diagonals:
            A_hollow_psi = 0
            if use_laplacian:
                # Only index the operator in the directions it varies:
                if laplacian_coeff_has_x:
                    i_prime = i
                if laplacian_coeff_has_y:
                    j_prime = j
                operator_coeff = laplacian_coeff[i_prime, j_prime]
                hollow_operator_result = _laplacian_single_point_interior(
                                             i, j, psi, over_dx2, over_dy2, operator_order, 1, &operator_diag)
                A_hollow_psi += operator_coeff * hollow_operator_result
                A_diag_total += operator_coeff * operator_diag

            if compute_error:
                # Add the squared residuals of the existing solution to the total:
                residual = cabs(A_hollow_psi + A_diag_total*psi[i, j] - b[i, j])
                squared_error_ptr[0] = squared_error_ptr[0] + residual*residual

            # The Gauss-Seidel prediction for psi at this point:
            psi_GS = (b[i, j] - A_hollow_psi)/A_diag_total

            # Update psi with overrelaxation at this point:
            psi[i, j] = psi[i, j] + relaxation_parameter*(psi_GS - psi[i, j])



@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void _SOR_step_edges(double_or_complex [:, :] psi, double_or_complex [:, :] A_diag,
                                    double_or_complex [:, :] b, double dx, double dy, double relaxation_parameter,
                                    int use_gradx, double_or_complex [:, :] gradx_coeff,
                                    int use_grady, double_or_complex [:, :] grady_coeff,
                                    int use_grad2x, double_or_complex [:, :] grad2x_coeff,
                                    int use_grad2y, double_or_complex [:, :] grad2y_coeff,
                                    int use_laplacian, double_or_complex [:, :] laplacian_coeff,
                                    double_or_complex[:, :] left_buffer, double_or_complex[:, :] right_buffer,
                                    double_or_complex[:, :] bottom_buffer, double_or_complex[:, :] top_buffer,
                                    int operator_order, double * squared_error_ptr, int compute_error):
    cdef int i
    cdef int j
    cdef int i_prime = 0
    cdef int j_prime = 0
    cdef int nx
    cdef int ny
    nx = psi.shape[0]
    ny = psi.shape[1]

    cdef int laplacian_coeff_has_x = laplacian_coeff.shape[0] > 1
    cdef int laplacian_coeff_has_y = laplacian_coeff.shape[1] > 1

    cdef double over_dx2 = 1/dx**2
    cdef double over_dy2 = 1/dy**2
    cdef double residual

    cdef double_or_complex operator_coeff
    cdef double_or_complex operator_diag
    cdef double_or_complex hollow_operator_result
    cdef double_or_complex A_hollow_psi
    cdef double_or_complex A_diag_total
    cdef double_or_complex psi_GS
    i = 0
    j = 0
    while True:
        # Compute the total diagonals of A. This is the sum of the
        # diagonal operator given by the user, and the diagonals of the
        # non-diagonal operators:
        A_diag_total = A_diag[i, j]
        # Compute the result of A*psi excluding the diagonals:
        A_hollow_psi = 0
        if use_laplacian:
            if laplacian_coeff_has_x:
                i_prime = i
            if laplacian_coeff_has_y:
                j_prime = j
            operator_coeff = laplacian_coeff[i_prime, j_prime]
            hollow_operator_result = _laplacian_single_point_edges(
                                         i, j, nx, ny, psi, left_buffer, right_buffer, bottom_buffer, top_buffer,
                                         over_dx2, over_dy2, operator_order, 1, &operator_diag)
            A_hollow_psi += operator_coeff * hollow_operator_result
            A_diag_total += operator_coeff * operator_diag

        if compute_error:
            # Add the squared residuals of the existing solution to the total:
            residual = cabs(A_hollow_psi + A_diag_total*psi[i, j] - b[i, j])
            squared_error_ptr[0] = squared_error_ptr[0] + residual*residual

        # The Gauss-Seidel prediction for psi at this point:
        psi_GS = (b[i, j] - A_hollow_psi)/A_diag_total

        # Update psi with overrelaxation at this point:
        psi[i, j] = psi[i, j] + relaxation_parameter*(psi_GS - psi[i, j])

        iter_edges(&i, &j, nx, ny, 2)
        if i > nx - 1 or j > ny - 1:
            break


cdef inline void _SOR_step_interior_complex(double complex [:, :] psi, double complex [:, :] A_diag,
                                    double complex [:, :] b, double dx, double dy, double relaxation_parameter,
                                    int use_gradx, double complex [:, :] gradx_coeff,
                                    int use_grady, double complex [:, :] grady_coeff,
                                    int use_grad2x, double complex [:, :] grad2x_coeff,
                                    int use_grad2y, double complex [:, :] grad2y_coeff,
                                    int use_laplacian, double complex [:, :] laplacian_coeff,
                                    int operator_order, double * squared_error_ptr, int compute_error) nogil:
    _SOR_step_interior(psi, A_diag, b, dx, dy, relaxation_parameter,
                       use_gradx, gradx_coeff, use_grady, grady_coeff,
                       use_grad2x, grad2x_coeff, use_grad2y, grad2y_coeff, use_laplacian, laplacian_coeff,
                       operator_order, squared_error_ptr, compute_error)


cdef inline void _SOR_step_interior_real(double [:, :] psi, double [:, :] A_diag,
                                    double [:, :] b, double dx, double dy, double relaxation_parameter,
                                    int use_gradx, double [:, :] gradx_coeff, int use_grady, double [:, :] grady_coeff,
                                    int use_grad2x, double [:, :] grad2x_coeff, int use_grad2y, double [:, :] grad2y_coeff,
                                    int use_laplacian, double [:, :] laplacian_coeff,
                                    int operator_order, double * squared_error_ptr, int compute_error) nogil:
    _SOR_step_interior(psi, A_diag, b, dx, dy, relaxation_parameter,
                       use_gradx, gradx_coeff, use_grady, grady_coeff,
                       use_grad2x, grad2x_coeff, use_grad2y, grad2y_coeff, use_laplacian, laplacian_coeff,
                       operator_order, squared_error_ptr, compute_error)

cdef inline void _SOR_step_edges_complex(double complex [:, :] psi, double complex [:, :] A_diag,
                                    double complex [:, :] b, double dx, double dy, double relaxation_parameter,
                                    int use_gradx, double complex [:, :] gradx_coeff,
                                    int use_grady, double complex [:, :] grady_coeff,
                                    int use_grad2x, double complex [:, :] grad2x_coeff,
                                    int use_grad2y, double complex [:, :] grad2y_coeff,
                                    int use_laplacian, double complex [:, :] laplacian_coeff,
                                    double complex [:, :] left_buffer, double complex [:, :] right_buffer,
                                    double complex [:, :] bottom_buffer, double complex [:, :] top_buffer,
                                    int operator_order, double * squared_error_ptr, int compute_error):
    _SOR_step_edges(psi, A_diag, b, dx, dy, relaxation_parameter,
                    use_gradx, gradx_coeff, use_grady, grady_coeff,
                    use_grad2x, grad2x_coeff, use_grad2y, grad2y_coeff, use_laplacian, laplacian_coeff,
                    left_buffer, right_buffer, bottom_buffer, top_buffer,
                    operator_order, squared_error_ptr, compute_error)

cdef inline void _SOR_step_edges_real(double [:, :] psi, double [:, :] A_diag,
                                    double [:, :] b, double dx, double dy, double relaxation_parameter,
                                    int use_gradx, double [:, :] gradx_coeff, int use_grady, double [:, :] grady_coeff,
                                    int use_grad2x, double [:, :] grad2x_coeff, int use_grad2y, double [:, :] grad2y_coeff,
                                    int use_laplacian, double [:, :] laplacian_coeff,
                                    double [:, :] left_buffer, double [:, :] right_buffer,
                                    double [:, :] bottom_buffer, double [:, :] top_buffer,
                                    int operator_order, double * squared_error_ptr, int compute_error):
    _SOR_step_edges(psi, A_diag, b, dx, dy, relaxation_parameter,
                    use_gradx, gradx_coeff, use_grady, grady_coeff,
                    use_grad2x, grad2x_coeff, use_grad2y, grad2y_coeff, use_laplacian, laplacian_coeff,
                    left_buffer, right_buffer, bottom_buffer, top_buffer,
                    operator_order, squared_error_ptr, compute_error)


def SOR_step_interior(psi, A_diag, A_nondiag, b, dx, dy, relaxation_parameter, operator_order, compute_error):
    cdef int use_gradx
    cdef int use_grady
    cdef int use_grad2x
    cdef int use_grad2y
    cdef int use_laplacian
    cdef int compute_error_cint = compute_error
    cdef double squared_error = 0

    gradx_coeff, grady_coeff, grad2x_coeff, grad2y_coeff, laplacian_coeff = process_operators(
        psi, A_nondiag, &use_gradx, &use_grady, &use_grad2x, &use_grad2y, &use_laplacian)

    if psi.dtype == np.complex128:
        if A_diag.dtype == np.float64:
            A_diag = np.array(A_diag, dtype=complex)
        _SOR_step_interior_complex(psi, A_diag, b, dx, dy, relaxation_parameter,
                                   use_gradx, gradx_coeff, use_grady, grady_coeff,
                                   use_grad2x, grad2x_coeff, use_grad2y, grad2y_coeff, use_laplacian, laplacian_coeff,
                                   operator_order, &squared_error, compute_error_cint)
    elif psi.dtype == np.float64:
        _SOR_step_interior_real(psi, A_diag, b, dx, dy, relaxation_parameter,
                                use_gradx, gradx_coeff, use_grady, grady_coeff,
                                use_grad2x, grad2x_coeff, use_grad2y, grad2y_coeff, use_laplacian, laplacian_coeff,
                                operator_order, &squared_error, compute_error_cint)

    return float(squared_error)


def SOR_step_edges(psi, A_diag, A_nondiag, b, dx, dy, relaxation_parameter,
                   left_buffer, right_buffer, bottom_buffer, top_buffer,
                   operator_order, squared_error, compute_error):
    cdef int use_gradx
    cdef int use_grady
    cdef int use_grad2x
    cdef int use_grad2y
    cdef int use_laplacian
    cdef int compute_error_cint = compute_error
    cdef double squared_error_cdouble = squared_error

    gradx_coeff, grady_coeff, grad2x_coeff, grad2y_coeff, laplacian_coeff = process_operators(
        psi, A_nondiag, &use_gradx, &use_grady, &use_grad2x, &use_grad2y, &use_laplacian)

    if psi.dtype == np.complex128:
        if A_diag.dtype == np.float64:
            A_diag = np.array(A_diag, dtype=complex)
        _SOR_step_edges_complex(psi, A_diag, b, dx, dy, relaxation_parameter,
                                use_gradx, gradx_coeff, use_grady, grady_coeff,
                                use_grad2x, grad2x_coeff, use_grad2y, grad2y_coeff, use_laplacian, laplacian_coeff,
                                left_buffer, right_buffer, bottom_buffer, top_buffer,
                                operator_order, &squared_error_cdouble, compute_error_cint)
    elif psi.dtype == np.float64:
        _SOR_step_edges_real(psi, A_diag, b, dx, dy, relaxation_parameter,
                             use_gradx, gradx_coeff, use_grady, grady_coeff,
                             use_grad2x, grad2x_coeff, use_grad2y, grad2y_coeff, use_laplacian, laplacian_coeff,
                             left_buffer, right_buffer, bottom_buffer, top_buffer,
                             operator_order, &squared_error_cdouble, compute_error_cint)
    return float(squared_error_cdouble)


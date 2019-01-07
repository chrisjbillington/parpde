from __future__ import division, print_function
import sys
import os
import time
import enum
import numpy as np
from mpi4py import MPI
import h5py
from .finite_differences import SOR_step, apply_operator
from scipy.fftpack import fft2, ifft2


def format_float(x, sigfigs=4, units=''):
    """Returns a string of the float f with a limited number of sig figs and a metric prefix"""

    prefixes = { -24: u"y", -21: u"z", -18: u"a", -15: u"f", -12: u"p", -9: u"n", -6: u"u", -3: u"m",
        0: u"", 3: u"k", 6: u"M", 9: u"G", 12: u"T", 15: u"P", 18: u"E", 21: u"Z", 24: u"Y" }

    if np.isnan(x) or np.isinf(x):
        return str(x)

    if x != 0:
        exponent = int(np.floor(np.log10(np.abs(x))))
        # Only multiples of 10^3
        exponent = int(np.floor(exponent / 3) * 3)
    else:
        exponent = 0

    significand = x / 10 ** exponent
    pre_decimal, post_decimal = divmod(significand, 1)
    digits = sigfigs - len(str(int(pre_decimal)))
    significand = round(significand, digits)
    result = '%.0{}f'.format(digits) % significand
    if exponent:
        try:
            # If our number has an SI prefix then use it
            prefix = prefixes[exponent]
            result += ' ' + prefix
        except KeyError:
            # Otherwise display in scientific notation
            result += 'e' + str(exponent)
            if units:
                result += ' '
    elif units:
        result += ' '
    return result + units


# Constants to represent differential operators.
class Operators(enum.IntEnum):
    GRADX = 0
    GRADY = 1
    GRAD2X = 2
    GRAD2Y = 3


class OperatorSum(dict):
    """Class for representing a weighted sum of operators. Supports
    arithemetic operations, and coefficients can be numpy arrays for spatially
    varying coefficients."""
    # Tells numpy arrays to not try to use their arithmetic operations
    # elementwise on us, instead they should defer to this class's arithmetic
    # methods:
    __array_priority__ = 1.0
    def __add__(self, other):
        new = OperatorSum(self)
        for obj, coefficient in other.items():
            new[obj] = new.get(obj, 0) + coefficient
        return new

    def __sub__(self, other):
        new = OperatorSum(self)
        for obj, coefficient in other.items():
            new[obj] = new.get(obj, 0) - coefficient
        return new

    def __mul__(self, factor):
        new = OperatorSum(self)
        for obj, coefficient in new.items():
            new[obj] = coefficient*factor
        return new

    def __div__(self, factor):
        new = OperatorSum(self)
        for obj, coefficient in new.items():
            new[obj] = coefficient/factor
        return new

    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__
    __rdiv__ = __div__


# Objects representing operators, which can be added, subtracted etc from each
# other and multiplied by constants:
GRADX = OperatorSum({Operators.GRADX: np.ones((1, 1))})
GRADY = OperatorSum({Operators.GRADY: np.ones((1, 1))})
GRAD2X = OperatorSum({Operators.GRAD2X: np.ones((1, 1))})
GRAD2Y = OperatorSum({Operators.GRAD2Y: np.ones((1, 1))})
LAPLACIAN = GRAD2X + GRAD2Y


def get_factors(n):
    """return all the factors of n"""
    factors = set()
    for i in range(1, int(n**(0.5)) + 1):
        if not n % i:
            factors.update((i, n // i))
    return factors


def get_best_2D_segmentation(size_x, size_y, N_segments):
    """Returns (best_n_segments_x, best_n_segments_y), describing the optimal
    cartesian grid for splitting up a rectangle of size (size_x, size_y) into
    N_segments equal sized segments such as to minimise surface area between
    the segments."""
    lowest_surface_area = None
    for n_segments_x in get_factors(N_segments):
        n_segments_y = N_segments // n_segments_x
        surface_area = n_segments_x * size_y + n_segments_y * size_x
        if lowest_surface_area is None or surface_area < lowest_surface_area:
            lowest_surface_area = surface_area
            best_n_segments_x, best_n_segments_y = n_segments_x, n_segments_y
    return best_n_segments_x, best_n_segments_y


class Simulator2D(object):
    def __init__(self, x_min_global, x_max_global, y_min_global, y_max_global, nx_global, ny_global,
                 periodic_x=False, periodic_y=False, operator_order=4):
        """A class for solving partial differential equations in two dimensions on
        multiple cores using MPI"""
        self.x_min_global = x_min_global
        self.x_max_global = x_max_global
        self.y_min_global = y_min_global
        self.y_max_global = y_max_global
        self.nx_global = nx_global
        self.ny_global = ny_global
        self.periodic_x = periodic_x
        self.periodic_y = periodic_y
        self.operator_order = operator_order
        self.n_edge_pts = self.operator_order // 2
        if not self.operator_order in [2, 4, 6]:
            msg = "Only differential operators of order 2, 4, 6 supported."
            raise ValueError(msg)
        self.global_shape = (self.nx_global, self.ny_global)

        self._setup_MPI_grid()

        self.shape = (self.nx, self.ny)

        self.dx = (self.x_max_global - self.x_min_global)/(self.nx_global - 1)
        self.dy = (self.y_max_global - self.y_min_global)/(self.ny_global - 1)

        self.x_min = self.x_min_global + self.dx * self.global_first_x_index
        self.y_min = self.y_min_global + self.dy * self.global_first_y_index

        self.x_max = self.x_min + self.dx * (self.nx - 1)
        self.y_max = self.y_min + self.dy * (self.ny - 1)

        self.x = np.linspace(self.x_min, self.x_max, self.nx).reshape((self.nx, 1))
        self.y = np.linspace(self.y_min, self.y_max, self.ny).reshape((1, self.ny))

        self.kx = self.ky = self.f_gradx = self.grady = self.grad2x = self.grad2y = self.f_laplacian = None
        if self.MPI_size_x == 1:
            # For FFTs, which can be done only on a single node in periodic directions:
            if periodic_x:
                self.kx = 2 * np.pi * np.fft.fftfreq(self.nx, d=self.dx).reshape((self.nx, 1))
                # x derivative operator in Fourier space:
                self.f_gradx = 1j*self.kx
                self.f_grad2x = -self.kx**2
            if periodic_y:
                self.ky = 2 * np.pi * np.fft.fftfreq(self.ny, d=self.dy).reshape((1, self.ny))
                # y derivative operator in Fourier space:
                self.f_grady = 1j*self.ky
                self.f_grad2y = -self.ky**2
            if periodic_x and periodic_y:
                # Laplace operator in Fourier space:
                self.f_laplacian = self.f_grad2x + self.f_grad2y


    def _setup_MPI_grid(self):
        """Split space up according to the number of MPI tasks. Set instance
        attributes for spatial extent and number of points in this MPI task,
        and create buffers and persistent communication requests for sending
        data to adjacent processes"""

        self.MPI_size = MPI.COMM_WORLD.Get_size()
        self.MPI_size_x, self.MPI_size_y = get_best_2D_segmentation(self.nx_global, self.ny_global, self.MPI_size)
        self.MPI_comm = MPI.COMM_WORLD.Create_cart([self.MPI_size_x, self.MPI_size_y],
                                                   periods=[self.periodic_x, self.periodic_y], reorder=True)
        self.MPI_rank = self.MPI_comm.Get_rank()
        self.MPI_x_coord, self.MPI_y_coord = self.MPI_comm.Get_coords(self.MPI_rank)
        if self.MPI_x_coord > 0 or self.periodic_x:
            self.MPI_rank_left = self.MPI_comm.Get_cart_rank((self.MPI_x_coord - 1, self.MPI_y_coord))
        else:
            self.MPI_rank_left = MPI.PROC_NULL
        if self.MPI_x_coord < self.MPI_size_x -1 or self.periodic_x:
            self.MPI_rank_right = self.MPI_comm.Get_cart_rank((self.MPI_x_coord + 1, self.MPI_y_coord))
        else:
            self.MPI_rank_right = MPI.PROC_NULL
        if self.MPI_y_coord > 0 or self.periodic_y:
            self.MPI_rank_down = self.MPI_comm.Get_cart_rank((self.MPI_x_coord, self.MPI_y_coord - 1))
        else:
            self.MPI_rank_down = MPI.PROC_NULL
        if self.MPI_y_coord < self.MPI_size_y -1 or self.periodic_y:
            self.MPI_rank_up = self.MPI_comm.Get_cart_rank((self.MPI_x_coord, self.MPI_y_coord + 1))
        else:
            self.MPI_rank_up = MPI.PROC_NULL

        self.processor_name = MPI.Get_processor_name()

        # Share out the points between processes in each direction:
        self.nx, nx_remaining = divmod(self.nx_global, self.MPI_size_x)
        if self.MPI_x_coord < nx_remaining:
            # Give the remaining to the lowest ranked processes:
            self.nx += 1
        self.ny, ny_remaining = divmod(self.ny_global, self.MPI_size_y)
        if self.MPI_y_coord < ny_remaining:
            # Give the remaining to the lowest ranked processes:
            self.ny += 1

        # What are our coordinates in the global array?
        self.global_first_x_index = self.nx * self.MPI_x_coord
        # Be sure to count the extra points the lower ranked processes have:
        if self.MPI_x_coord >= nx_remaining:
            self.global_first_x_index += nx_remaining

        self.global_first_y_index = self.ny * self.MPI_y_coord
        # Be sure to count the extra points the lower ranked processes have:
        if self.MPI_y_coord >= ny_remaining:
            self.global_first_y_index += ny_remaining

        # We need to tag our data to have a way other than rank to distinguish
        # between multiple messages the two tasks might be sending each other
        # at the same time:
        TAG_LEFT_TO_RIGHT = 0
        TAG_RIGHT_TO_LEFT = 1
        TAG_DOWN_TO_UP = 2
        TAG_UP_TO_DOWN = 3

        # Buffers and MPI request objects for sending and receiving data to
        # and from other processes. Sorted by whether the datatype is real or
        # complex.
        self.MPI_send_buffers = {}
        self.MPI_receive_buffers = {}
        self.MPI_requests = {}
        for dtype in [np.float64, np.complex128]:
            x_edge_shape = (self.n_edge_pts, self.ny)
            y_edge_shape = (self.nx, self.n_edge_pts)
            left_send_buffer = np.zeros(x_edge_shape, dtype=dtype)
            left_receive_buffer = np.zeros(x_edge_shape, dtype=dtype)
            right_send_buffer = np.zeros(x_edge_shape, dtype=dtype)
            right_receive_buffer = np.zeros(x_edge_shape, dtype=dtype)
            bottom_send_buffer = np.zeros(y_edge_shape, dtype=dtype)
            bottom_receive_buffer = np.zeros(y_edge_shape, dtype=dtype)
            top_send_buffer = np.zeros(y_edge_shape, dtype=dtype)
            top_receive_buffer = np.zeros(y_edge_shape, dtype=dtype)


            send_left = self.MPI_comm.Send_init(left_send_buffer, self.MPI_rank_left, tag=TAG_RIGHT_TO_LEFT)
            send_right = self.MPI_comm.Send_init(right_send_buffer, self.MPI_rank_right, tag=TAG_LEFT_TO_RIGHT)
            send_bottom = self.MPI_comm.Send_init(bottom_send_buffer, self.MPI_rank_down, tag=TAG_UP_TO_DOWN)
            send_top = self.MPI_comm.Send_init(top_send_buffer, self.MPI_rank_up, tag=TAG_DOWN_TO_UP)
            receive_left = self.MPI_comm.Recv_init(left_receive_buffer, self.MPI_rank_left, tag=TAG_LEFT_TO_RIGHT)
            receive_right = self.MPI_comm.Recv_init(right_receive_buffer, self.MPI_rank_right, tag=TAG_RIGHT_TO_LEFT)
            receive_bottom = self.MPI_comm.Recv_init(bottom_receive_buffer, self.MPI_rank_down, tag=TAG_DOWN_TO_UP)
            receive_top = self.MPI_comm.Recv_init(top_receive_buffer, self.MPI_rank_up, tag=TAG_UP_TO_DOWN)

            self.MPI_send_buffers[dtype] = (left_send_buffer, right_send_buffer, bottom_send_buffer, top_send_buffer)

            self.MPI_receive_buffers[dtype] = (left_receive_buffer, right_receive_buffer,
                                               bottom_receive_buffer, top_receive_buffer)

            self.MPI_requests[dtype] = (send_left, send_right, send_bottom, send_top,
                                        receive_left, receive_right, receive_bottom, receive_top)
        self.pending_requests = None


    def MPI_send_at_edges(self, psi):
        """Start an asynchronous MPI send data from the edges of psi to all
        adjacent MPI processes."""
        left_buffer, right_buffer, bottom_buffer, top_buffer = self.MPI_send_buffers[psi.dtype.type]
        left_buffer[:] = psi[:self.n_edge_pts, :]
        right_buffer[:] = psi[-self.n_edge_pts:, :]
        bottom_buffer[:] = psi[:, :self.n_edge_pts]
        top_buffer[:] = psi[:, -self.n_edge_pts:]
        self.pending_requests = self.MPI_requests[psi.dtype.type]
        MPI.Prequest.Startall(self.pending_requests)

    def MPI_receive_at_edges(self):
        """Finalise an asynchronous MPI transfer from all adjacent MPI
        processes. Data remains in the receive buffers and can be accessed by
        the caller after this method returns."""
        MPI.Prequest.Waitall(self.pending_requests)
        self.pending_requests = None

    def par_sum(self, psi):
        """Sum the given field over all MPI processes"""
        local_sum = np.asarray(psi.sum())
        result = np.zeros_like(local_sum)
        self.MPI_comm.Allreduce(local_sum, result, MPI.SUM)
        return result

    def par_vdot(self, psi1, psi2):
        """"Dots two vectors (with complex comjucation of the first) and sums
        result over MPI processes"""
        local_dot = np.asarray(np.vdot(psi1, psi2))
        result = np.zeros_like(local_dot)
        self.MPI_comm.Allreduce(local_dot, result, MPI.SUM)
        return result

    def par_operator_init(self, operator, psi, out=None):
        self.MPI_send_at_edges(psi)
        gradx_coeff = operator.get(Operators.GRADX, None)
        grady_coeff = operator.get(Operators.GRADY, None)
        grad2x_coeff = operator.get(Operators.GRAD2X, None)
        grad2y_coeff = operator.get(Operators.GRAD2Y, None)
        return apply_operator(psi, gradx_coeff, grady_coeff, grad2x_coeff, grad2y_coeff,
                              self.dx,  self.dy, self.operator_order, out=out,
                              left_buffer=None, right_buffer=None,
                              bottom_buffer=None, top_buffer=None, interior=True, edges=False)

    def par_operator_finalise(self, operator, psi, out):
        gradx_coeff = operator.get(Operators.GRADX, None)
        grady_coeff = operator.get(Operators.GRADY, None)
        grad2x_coeff = operator.get(Operators.GRAD2X, None)
        grad2y_coeff = operator.get(Operators.GRAD2Y, None)
        left_buffer, right_buffer, bottom_buffer, top_buffer = self.MPI_receive_buffers[psi.dtype.type]
        self.MPI_receive_at_edges()
        return apply_operator(psi, gradx_coeff, grady_coeff, grad2x_coeff, grad2y_coeff,
                              self.dx,  self.dy, self.operator_order, out=out,
                              left_buffer=left_buffer, right_buffer=right_buffer,
                              bottom_buffer=bottom_buffer, top_buffer=top_buffer, interior=False, edges=True)

    def par_operator(self, operator, psi, out=None, use_ffts=False):
        if use_ffts:
            if out is not None:
                raise ValueError("out parameter not supported for fft operators")
            return self.apply_fourier_operator(operator, psi)
        out = self.par_operator_init(operator, psi, out)
        return self.par_operator_finalise(operator, psi, out)

    def make_fourier_operator(self, operator):
        """If the operator is diagonal in the Fourier basis, return its
        fourier representation as an array."""
        gradx_coeff = operator.get(Operators.GRADX, None)
        grady_coeff = operator.get(Operators.GRADY, None)
        grad2x_coeff = operator.get(Operators.GRAD2X, None)
        grad2y_coeff = operator.get(Operators.GRAD2Y, None)

        if self.MPI_size > 1 or not (self.periodic_x and self.periodic_y):
            msg = "FFTs can only be done on a single process with periodic boundary conditions"
            raise ValueError(msg)

        if any(op.shape != (1, 1) for op in operator.values()):
            msg = ("FFTs cannot be used to evaluate operators with spatially varying coefficients, " +
                   "as they are not diagonal in the Fourier basis.")
            raise ValueError(msg)

        # Compute the operator in Fourier space:
        f_operator = 0
        if gradx_coeff is not None:
            f_operator = f_operator + gradx_coeff * self.f_gradx
        if grady_coeff is not None:
            f_operator = f_operator + grady_coeff * self.f_grady
        if grad2x_coeff is not None:
            f_operator = f_operator + grad2x_coeff * self.f_grad2x
        if grad2y_coeff is not None:
            f_operator = f_operator + grad2y_coeff * self.f_grad2y
        return f_operator

    def apply_fourier_operator(self, operator, psi):
        """Applies an operator in the Fourier basis. If operator provided is
        an OperatorSum, it is converted to the Fourier basis. If operator is
        an array, it is assumed to already be the representation of the
        operator in the Fourier basis."""
        if isinstance(operator, OperatorSum):
            operator = self.make_fourier_operator(operator)
        elif not isinstance(operator, np.ndarray):
            raise TypeError(operator)
        result = ifft2(operator*fft2(psi))
        if psi.dtype == np.float64 and all(op.dtype == np.float64 for op in operator.values()):
            result = result.real
        return result

    def _pre_step_checks(self, i, t, psi, output_interval, output_callback, post_step_callback, infodict, final_call=False):
        if np.isnan(psi).any() or np.isinf(psi).any():
            sys.stdout.write('It exploded :(\n')
            # Call output funcs so user can see exactly when things went pear shaped:
            if post_step_callback is not None:
                post_step_callback(i, t, psi, infodict)
            if output_callback is not None and output_interval is not None:
                output_callback(i, t, psi, infodict)
            raise RuntimeError('It exploded :(')
        if post_step_callback is not None:
            post_step_callback(i, t, psi, infodict)
        output_callback_called = False
        if output_callback is not None and output_interval is not None:
            if np.iterable(output_interval):
                if i in output_interval:
                    output_callback(i, t, psi, infodict)
                    output_callback_called = True
            elif not i % output_interval:
                output_callback(i, t, psi, infodict)
                output_callback_called = True
            if final_call and not output_callback_called:
                output_callback(i, t, psi, infodict)

    def successive_overrelaxation(self, system, psi, boundary_mask=None, relaxation_parameter=1.7, convergence=1e-13,
                                   output_interval=100, output_callback=None, post_step_callback=None,
                                   convergence_check_interval=10):
        """Solve a system of equations A*psi=b using sucessive overrelaxation.
        The provided function for the system of equations should accept psi
        and return the diagonal part of A as an array, the non-diagonal part
        of A as an OperatorSum instance, and b as an array. This function
        requires an initial guess for psi, and optionally takes a boolean
        array boundary_mask, which specifies which region is in-bounds for the
        problem. Any points not selected by the mask will not be evolved, and
        as such the initial-guess value of psi there serves as boundary conditions."""
        i = 0
        start_time = time.time()
        convergence_calc = np.nan
        if boundary_mask is not None:
            boundary_mask = np.array(boundary_mask, dtype=np.uint8)
        while True:
            time_per_step = (time.time() - start_time)/i if i else np.nan
            infodict = {"convergence": convergence_calc, 'time per step': time_per_step}
            self._pre_step_checks(i, 0, psi, output_interval, output_callback, post_step_callback, infodict)
            self.MPI_send_at_edges(psi)
            A_diag, A_nondiag, b = system(psi)
            if not i % convergence_check_interval:
                # Only compute the error every convergence_check_interval steps to save time
                compute_error=True
                integral_b = self.par_vdot(b, b).real
            else:
                compute_error = False

            gradx_coeff = A_nondiag.get(Operators.GRADX, None)
            grady_coeff = A_nondiag.get(Operators.GRADY, None)
            grad2x_coeff = A_nondiag.get(Operators.GRAD2X, None)
            grad2y_coeff = A_nondiag.get(Operators.GRAD2Y, None)

            squared_error = SOR_step(psi, A_diag, gradx_coeff, grady_coeff, grad2x_coeff, grad2y_coeff, b,
                                     self.dx, self.dy, relaxation_parameter, self.operator_order,
                                     interior=True, edges=False, boundary_mask=boundary_mask)

            self.MPI_receive_at_edges()
            left_buffer, right_buffer, bottom_buffer, top_buffer = self.MPI_receive_buffers[psi.dtype.type]

            squared_error = SOR_step(psi, A_diag, gradx_coeff, grady_coeff, grad2x_coeff, grad2y_coeff, b,
                                     self.dx, self.dy, relaxation_parameter, self.operator_order,
                                     left_buffer, right_buffer, bottom_buffer, top_buffer,
                                     squared_error=squared_error, interior=False, edges=True,
                                     boundary_mask=boundary_mask)
            if compute_error:
                squared_error = np.asarray(squared_error).reshape(1)
                total_squared_error = np.zeros(1)
                self.MPI_comm.Allreduce(squared_error, total_squared_error, MPI.SUM)
                convergence_calc = np.sqrt(total_squared_error[0]/integral_b)
                if convergence_calc < convergence:
                    break
            i += 1
        time_per_step = (time.time() - start_time)/i if i else np.nan
        infodict = {"convergence": convergence_calc, 'time per step': time_per_step}
        self._pre_step_checks(i, 0, psi, output_interval, output_callback, post_step_callback, infodict, final_call=True)

    def _evolve(self, dt, t_final, psi, integration_step_func,
                output_interval, output_callback, post_step_callback, estimate_error, method_order):
        """common loop for time integration"""
        t = 0
        i = 0
        step_error = 0
        error_check_substep = None
        start_time = time.time()

        # The number of substeps we take depends on the order of the method.
        # We do whatever we epxect to reduce the error by a factor of 16:
        n_errcheck_substeps = int(np.ceil(16**(1.0/method_order)))
        while t < t_final or error_check_substep is not None:
            # The step before output is to occur, take n_errcheck_substeps
            # steps of size dt/n_errcheck_substeps, then step over the same
            # interval with a size of dt, and compare the solutions for an
            # estimate of the per-step error.
            if estimate_error and output_interval is not None:
                if error_check_substep is None:
                    # Do we need to do an error checking step?
                    if np.iterable(output_interval):
                        if (i + 1) in output_interval:
                            error_check_substep = 0
                    elif not (i +1) % output_interval:
                        error_check_substep = 0

                if error_check_substep == 0:
                    # Save the wavefunction and actual timestep before setting
                    # dt to one fifth its value:
                    psi_pre_errorcheck = psi.copy()
                    dt_unmodified = dt
                    dt = dt/n_errcheck_substeps
                if error_check_substep == n_errcheck_substeps:
                    # Ok, we've done our five small steps. Save the resulting wavefunction:
                    psi_accurate = psi.copy()
                    # Restore the wavefunction and dt:
                    psi[:] = psi_pre_errorcheck
                    dt = dt_unmodified
                if error_check_substep == n_errcheck_substeps + 1:
                    # Ok, we've completed the normal sized step. Compare wavefunctions:
                    sum_squared_error = self.par_sum(np.abs(psi - psi_accurate)**2)
                    psi_norm = self.par_vdot(psi_accurate, psi_accurate).real
                    step_error = np.sqrt(sum_squared_error/psi_norm)
                    # Finished checking error.
                    error_check_substep = None
                    # The missed increment of i from the normal sized
                    # timestep. This i should trigger output at the next call
                    # to _pre_step_checks:
                    i += 1
            time_per_step = (time.time() - start_time)/i if i else np.nan
            infodict = {'time per step': time_per_step, 'step error': step_error}
            if error_check_substep is None:
                self._pre_step_checks(i, t, psi, output_interval, output_callback, post_step_callback, infodict)

            # The actual integration step:
            integration_step_func(t, dt, psi)
            t += dt
            if error_check_substep is not None:
                error_check_substep += 1
            else:
                i += 1

        # Ensure we output at the end:
        time_per_step = (time.time() - start_time)/i if i else np.nan
        infodict = {'time per step': time_per_step, 'step error': step_error}
        self._pre_step_checks(i, t, psi, output_interval, output_callback, post_step_callback, infodict, final_call=True)

    def rk4(self, dt, t_final, dpsi_dt, psi,
            output_interval=100, output_callback=None, post_step_callback=None, estimate_error=False):
        """Fourth order Runge-Kutta. dpsi_dt should return an array for the time derivatives of psi."""

        def rk4_step(t, dt, psi):
            k1 = dpsi_dt(t, psi)
            k2 = dpsi_dt(t + 0.5*dt, psi + 0.5*k1*dt)
            k3 = dpsi_dt(t + 0.5*dt, psi + 0.5*k2*dt)
            k4 = dpsi_dt(t + dt, psi + k3*dt)
            psi[:] += dt/6*(k1 + 2*k2 + 2*k3 + k4)

        self._evolve(dt, t_final, psi, rk4_step,
                     output_interval, output_callback, post_step_callback, estimate_error, method_order=4)

    def rk4ilip(self, dt, t_final, dpsi_dt, psi, omega_imag_provided=False,
                output_interval=100, output_callback=None, post_step_callback=None, estimate_error=False):
        """Fourth order Runge-Kutta in an instantaneous local interaction picture.
        dpsi_dt should return both the derivative of psi, and an array omega,
        which is H_local/hbar at each point in space (Note that H_local should not
        be excluded from the calculation of dpsi_dt). If omega is purely
        imaginary, you can instead return the omega_imag comprising its imaginary
        part, in which case you should set omega_imag_provided to True. This means
        real arrays can be used for arithmetic instead of complex ones, which is
        faster."""

        def rk4ilip_step(t, dt, psi):
            if omega_imag_provided:
                # Omega is purely imaginary, and so omega_imag has been provided
                # instead so real arithmetic can be used:
                f1, omega_imag = dpsi_dt(t, psi)
                i_omega = -omega_imag
                # Have to take abs(dt) here in case we are taking a backward step (dt < 0):
                i_omega_clipped = i_omega.clip(-400/abs(dt), 400/abs(dt))
                U_half = np.exp(i_omega_clipped*0.5*dt)
            else:
                f1, omega = dpsi_dt(t, psi)
                i_omega = 1j*omega
                if omega.dtype == np.float64:
                    theta = omega*0.5*dt
                    U_half = np.cos(theta) + 1j*np.sin(theta) # faster than np.exp(1j*theta) when theta is real
                else:
                    i_omega_clipped = i_omega.real.clip(-400/abs(dt), 400/abs(dt)) + 1j*i_omega.imag
                    U_half = np.exp(1j*i_omega_clipped*0.5*dt)

            U_full = U_half**2
            U_dagger_half = 1/U_half
            U_dagger_full = 1/U_full

            k1 = f1 + i_omega*psi

            phi_1 = psi + 0.5*k1*dt
            psi_1 = U_dagger_half*phi_1
            f2, _ = dpsi_dt(t + 0.5*dt, psi_1)
            k2 = U_half*f2 + i_omega*phi_1

            phi_2 = psi + 0.5*k2*dt
            psi_2 = U_dagger_half*phi_2
            f3, _ = dpsi_dt(t + 0.5*dt, psi_2)
            k3 = U_half*f3 + i_omega*phi_2

            phi_3 = psi + k3*dt
            psi_3 = U_dagger_full*phi_3
            f4, _ = dpsi_dt(t + dt, psi_3)
            k4 = U_full*f4 + i_omega*phi_3

            phi_4 = psi + dt/6*(k1 + 2*k2 + 2*k3 + k4)
            psi[:] = U_dagger_full*phi_4

        self._evolve(dt, t_final, psi, rk4ilip_step,
                     output_interval, output_callback, post_step_callback, estimate_error, method_order=4)

    def split_step(self, dt, t_final, nonlocal_operator, local_operator, psi,
            output_interval=100, output_callback=None, post_step_callback=None,
            estimate_error=False, method_order=2):
        """Split step method. Nonlocal_operator is an operator diagonal in
        Fourier space. It can be either an OperatorSum instance if it is just
        a sum of differential operators, or can be given as an array
        representing the operator's diagonals in the Fourier basis."""

        if isinstance(nonlocal_operator, OperatorSum):
            fourier_operator = self.make_fourier_operator(nonlocal_operator)
        elif isinstance(nonlocal_operator, np.ndarray):
            fourier_operator = nonlocal_operator
        else:
            msg = "nonlocal_operator must be OperatorSum instance or array"
            raise TypeError(msg)

        # We cache these for different timestep sizes:
        fourier_unitaries = {}

        def U(dt, psi):
            try:
                unitary = fourier_unitaries[dt]
            except KeyError:
                unitary = np.exp(dt*fourier_operator)
                fourier_unitaries[dt] = unitary
            return self.apply_fourier_operator(unitary, psi)

        # The same operator is used at the end of one step as at the beginning
        # of the next (if they have the same dt), so we cache it:
        cached_local_unitary = {0: (None, None, None)}

        def split_step_2nd_order_step(t, dt, psi):
            previous_t, previous_dt, previous_local_unitary = cached_local_unitary[0]
            if previous_t == t and previous_dt == dt:
                local_unitary = previous_local_unitary
            else:
                local_unitary = np.exp(0.5*dt*local_operator(t, psi))
            # Real space half step:
            psi[:] *= local_unitary
            # Fourier space step:
            psi[:] =  U(dt, psi)
            # Real space half step:
            local_unitary = np.exp(0.5*dt*local_operator(t+dt, psi))
            cached_local_unitary[0] = (t+dt, dt, local_unitary)
            psi[:] *= local_unitary

        def split_step_4nd_order_step(t, dt, psi):
            p = 1/(4 - 4**(1.0/3.0))
            for subdt in [p * dt, p * dt, (1 - 4*p) * dt, p * dt, p * dt]:
                split_step_2nd_order_step(t, subdt, psi)
                t += subdt
                
        if method_order == 2:
            step_func = split_step_2nd_order_step
        elif method_order == 4:
            step_func = split_step_4nd_order_step
        else:
            msg = "method_order must be 2 or 4, not %s" % str(method_order)
            raise ValueError(msg)

        self._evolve(dt, t_final, psi, step_func,
                     output_interval, output_callback, post_step_callback,
                     estimate_error, method_order=method_order)

    def rk4ip(self, dt, t_final, nonlocal_operator, local_operator, psi,
              output_interval=100, output_callback=None, post_step_callback=None, estimate_error=False):
        """Fourth order Runge-Kutta in the interaction picture. Uses an
        interaction picture based on an operator diagonal in Fourier space,
        which should be passed in as nonlocal_operator. It can be either an
        OperatorSum instance if it is just a sum of differential operators, or
        can be given as an array representing the operator's diagonals in the
        Fourier basis. The rest of the equation is solved based on an operator
        diagonal in real space, which the function local_operator should
        return."""

        if isinstance(nonlocal_operator, OperatorSum):
            fourier_operator = self.make_fourier_operator(nonlocal_operator)
        elif isinstance(nonlocal_operator, np.ndarray):
            fourier_operator = nonlocal_operator
        else:
            msg = "nonlocal_operator must be OperatorSum instance or array"
            raise TypeError(msg)

        # We cache these for different timestep sizes:
        fourier_unitaries = {}

        def G(t, psi):
            return local_operator(t, psi)*psi

        def U(dt, psi):
            try:
                unitary = fourier_unitaries[dt]
            except KeyError:
                unitary = np.exp(0.5*dt*fourier_operator)
                fourier_unitaries[dt] = unitary
            return self.apply_fourier_operator(unitary, psi)

        def rk4ip_step(t, dt, psi):
            """The propagator for one step of the RK4IP method"""
            psi_I = U(dt, psi)
            k1 = U(dt, G(t, psi))
            k2 = G(t + 0.5*dt, psi_I + 0.5*dt*k1)
            k3 = G(t + 0.5*dt, psi_I + 0.5*dt*k2)
            k4 = G(t + dt, U(dt, psi_I + dt*k3))
            psi[:] = U(dt, psi_I + dt/6. * (k1 + 2*k2 + 2*k3)) + dt/6.*k4

        self._evolve(dt, t_final, psi, rk4ip_step,
                     output_interval, output_callback, post_step_callback, estimate_error, method_order=4)

class HDFOutput(object):
    def __init__(self, simulator, output_dir, flush_output=True):
        self.simulator = simulator
        self.output_dir = output_dir
        self.flush_output=flush_output
        if not simulator.MPI_rank and not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)
        self.basename = str(simulator.MPI_rank).zfill(len(str(simulator.MPI_size))) + '.h5'
        self.filepath = os.path.join(self.output_dir, self.basename)
        # Ensure output folder exists before other processes continue:
        simulator.MPI_comm.Barrier()
        self.file = h5py.File(self.filepath, 'w')

        self.file.attrs['x_min_global'] = simulator.x_min_global
        self.file.attrs['x_max_global'] = simulator.x_max_global
        self.file.attrs['y_min_global'] = simulator.y_min_global
        self.file.attrs['y_max_global'] = simulator.y_max_global
        self.file.attrs['nx_global'] = simulator.nx_global
        self.file.attrs['ny_global'] = simulator.ny_global
        self.file.attrs['global_shape'] = simulator.global_shape
        geometry_dtype = [('rank', int),
                          ('processor_name', 'a256'),
                          ('x_cart_coord', int),
                          ('y_cart_coord', int),
                          ('first_x_index', int),
                          ('first_y_index', int),
                          ('nx', int),
                          ('ny', int)]
        MPI_geometry_dset = self.file.create_dataset('MPI_geometry', shape=(1,), dtype=geometry_dtype)
        MPI_geometry_dset.attrs['MPI_size'] = simulator.MPI_size
        data = (simulator.MPI_rank, simulator.processor_name, simulator.MPI_x_coord, simulator.MPI_y_coord,
                simulator.global_first_x_index, simulator.global_first_y_index, simulator.nx, simulator.ny)
        MPI_geometry_dset[0] = data

    def save(self, psi, output_log_data, flush=True):
        if not 'psi' in self.file:
            self.file.create_dataset('psi', (0,) + psi.shape[:-2] + self.simulator.shape,
                                     maxshape=(None,) + psi.shape[:-2] + self.simulator.shape,
                                     dtype=psi.dtype)
        if not 'output_log' in self.file:
            self.file.create_dataset('output_log', (0,), maxshape=(None,), dtype=output_log_data.dtype)

        output_log_dataset = self.file['output_log']
        output_log_dataset.resize((len(output_log_dataset) + 1,))
        output_log_dataset[-1] = output_log_data
        psi_dataset = self.file['psi']
        psi_dataset.resize((len(psi_dataset) + 1,) + psi_dataset.shape[1:])
        psi_dataset[-1] = psi
        if self.flush_output:
            self.file.flush()

    @staticmethod
    def iterframes(directory, start=0, end=None, step=1, frames=None):
        with h5py.File(os.path.join(directory, '0.h5'), 'r') as master_file:
            MPI_size = master_file['MPI_geometry'].attrs['MPI_size']
            global_shape = master_file.attrs['global_shape']
            psi_other_dims = master_file['psi'].shape[1:-2]
            dtype = master_file['psi'].dtype
            psi_global_shape = tuple(psi_other_dims) + tuple(global_shape)
            if end is None:
                end = len(master_file['psi'])
        files = []
        for rank in range(MPI_size):
            basename = str(rank).zfill(len(str(MPI_size))) + '.h5'
            f = h5py.File(os.path.join(directory, basename), 'r')
            psi_dataset = f['psi']
            start_x = f['MPI_geometry']['first_x_index'][0]
            start_y = f['MPI_geometry']['first_y_index'][0]
            nx = f['MPI_geometry']['nx'][0]
            ny = f['MPI_geometry']['ny'][0]
            files.append((f, psi_dataset, start_x, start_y, nx, ny))

        for i in frames if frames is not None else range(start, end, step):
            psi = np.zeros(psi_global_shape, dtype=dtype)
            for f, psi_dataset, start_x, start_y, nx, ny in files:
                psi[..., start_x:start_x + nx, start_y:start_y + ny] = psi_dataset[i]
            yield i, psi






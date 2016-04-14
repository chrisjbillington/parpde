from __future__ import division, print_function
import os
import enum
import numpy as np
from mpi4py import MPI
import h5py
from cython_functions import laplacian, SOR_step_interior, SOR_step_edges
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
    result = str(significand)
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


# Some slice objects for conveniently slicing arrays:
LEFT_EDGE = np.s_[0]
RIGHT_EDGE = np.s_[-1]
BOTTOM_EDGE = np.s_[:, 0]
TOP_EDGE = np.s_[:, -1]


# Constants to represent differential operators:
class Operators(enum.IntEnum):
    LAPLACIAN = 0


class OperatorSum(dict):
    """Class for representing a weighted sum of operators"""
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



LAPLACIAN = OperatorSum({Operators.LAPLACIAN: np.ones((1, 1))})


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
                 periodic_x=False, periodic_y=False):
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

        self.global_shape = (self.nx_global, self.ny_global)

        self._setup_MPI_grid()

        self.shape = (self.nx, self.ny)

        self.dx = (self.x_max_global - self.x_min_global)/(self.nx_global + 1)
        self.dy = (self.y_max_global - self.y_min_global)/(self.nx_global + 1)

        self.x_min = self.x_min_global + self.dx * self.global_first_x_index
        self.y_min = self.y_min_global + self.dy * self.global_first_y_index

        self.x_max = self.x_min + self.dx * self.nx
        self.y_max = self.y_min + self.dy * self.ny

        self.x = np.linspace(self.x_min, self.x_max, self.nx).reshape((self.nx, 1))
        self.y = np.linspace(self.y_min, self.y_max, self.ny).reshape((1, self.ny))


        self.kx = self.ky = self.f_gradx = self.grady = self.f_laplacian = None
        if self.MPI_size_x == 1:
            # For FFTs, which can be done only on a single node in periodic directions:
            if periodic_x:
                self.kx = 2 * np.pi * np.fft.fftfreq(self.nx, d=self.dx).reshape((self.nx, 1))
                # x derivative operator in Fourier space:
                self.f_gradx = 1j*self.kx
            if periodic_y:
                self.ky = 2 * np.pi * np.fft.fftfreq(self.ny, d=self.dy).reshape((1, self.ny))
                # y derivative operator in Fourier space:
                self.f_grady = 1j*self.ky
            if periodic_x and periodic_y:
                # Laplace operator in Fourier space:
                self.f_laplacian = -(self.kx**2 + self.ky**2)


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

        # The data we want to send to adjacent processes isn't in contiguous
        # memory, so we need to copy it into and out of temporary buffers.
        # Buffers for sending real numbers:
        self.MPI_left_send_buffer_real = np.zeros(self.ny, dtype=np.float64)
        self.MPI_left_receive_buffer_real = np.zeros(self.ny, dtype=np.float64)
        self.MPI_right_send_buffer_real = np.zeros(self.ny, dtype=np.float64)
        self.MPI_right_receive_buffer_real = np.zeros(self.ny, dtype=np.float64)
        self.MPI_top_send_buffer_real = np.zeros(self.nx, dtype=np.float64)
        self.MPI_top_receive_buffer_real = np.zeros(self.nx, dtype=np.float64)
        self.MPI_bottom_send_buffer_real = np.zeros(self.nx, dtype=np.float64)
        self.MPI_bottom_receive_buffer_real = np.zeros(self.nx, dtype=np.float64)
        # Buffers for sending complex numbers:
        self.MPI_left_send_buffer_complex = np.zeros(self.ny, dtype=np.complex128)
        self.MPI_left_receive_buffer_complex = np.zeros(self.ny, dtype=np.complex128)
        self.MPI_right_send_buffer_complex = np.zeros(self.ny, dtype=np.complex128)
        self.MPI_right_receive_buffer_complex = np.zeros(self.ny, dtype=np.complex128)
        self.MPI_top_send_buffer_complex = np.zeros(self.nx, dtype=np.complex128)
        self.MPI_top_receive_buffer_complex = np.zeros(self.nx, dtype=np.complex128)
        self.MPI_bottom_send_buffer_complex = np.zeros(self.nx, dtype=np.complex128)
        self.MPI_bottom_receive_buffer_complex = np.zeros(self.nx, dtype=np.complex128)

        # We need to tag our data to have a way other than rank to distinguish
        # between multiple messages the two tasks might be sending each other
        # at the same time:
        TAG_LEFT_TO_RIGHT = 0
        TAG_RIGHT_TO_LEFT = 1
        TAG_DOWN_TO_UP = 2
        TAG_UP_TO_DOWN = 3

        # Create persistent requests for the data transfers we will regularly be doing:
        self.MPI_send_left_real = self.MPI_comm.Send_init(self.MPI_left_send_buffer_real,
                                                          self.MPI_rank_left, tag=TAG_RIGHT_TO_LEFT)
        self.MPI_send_right_real = self.MPI_comm.Send_init(self.MPI_right_send_buffer_real,
                                                           self.MPI_rank_right, tag=TAG_LEFT_TO_RIGHT)
        self.MPI_receive_left_real = self.MPI_comm.Recv_init(self.MPI_left_receive_buffer_real,
                                                             self.MPI_rank_left, tag=TAG_LEFT_TO_RIGHT)
        self.MPI_receive_right_real = self.MPI_comm.Recv_init(self.MPI_right_receive_buffer_real,
                                                              self.MPI_rank_right, tag=TAG_RIGHT_TO_LEFT)
        self.MPI_send_bottom_real = self.MPI_comm.Send_init(self.MPI_bottom_send_buffer_real,
                                                            self.MPI_rank_down, tag=TAG_UP_TO_DOWN)
        self.MPI_send_top_real = self.MPI_comm.Send_init(self.MPI_top_send_buffer_real,
                                                         self.MPI_rank_up, tag=TAG_DOWN_TO_UP)
        self.MPI_receive_bottom_real = self.MPI_comm.Recv_init(self.MPI_bottom_receive_buffer_real,
                                                               self.MPI_rank_down, tag=TAG_DOWN_TO_UP)
        self.MPI_receive_top_real = self.MPI_comm.Recv_init(self.MPI_top_receive_buffer_real,
                                                            self.MPI_rank_up, tag=TAG_UP_TO_DOWN)

        self.MPI_all_requests_real = [self.MPI_send_left_real, self.MPI_receive_left_real,
                                      self.MPI_send_right_real, self.MPI_receive_right_real,
                                      self.MPI_send_bottom_real, self.MPI_receive_bottom_real,
                                      self.MPI_send_top_real, self.MPI_receive_top_real]


        # Create persistent requests for the data transfers we will regularly be doing:
        self.MPI_send_left_complex = self.MPI_comm.Send_init(self.MPI_left_send_buffer_complex,
                                                             self.MPI_rank_left, tag=TAG_RIGHT_TO_LEFT)
        self.MPI_send_right_complex = self.MPI_comm.Send_init(self.MPI_right_send_buffer_complex,
                                                              self.MPI_rank_right, tag=TAG_LEFT_TO_RIGHT)
        self.MPI_receive_left_complex = self.MPI_comm.Recv_init(self.MPI_left_receive_buffer_complex,
                                                                self.MPI_rank_left, tag=TAG_LEFT_TO_RIGHT)
        self.MPI_receive_right_complex = self.MPI_comm.Recv_init(self.MPI_right_receive_buffer_complex,
                                                                 self.MPI_rank_right, tag=TAG_RIGHT_TO_LEFT)
        self.MPI_send_bottom_complex = self.MPI_comm.Send_init(self.MPI_bottom_send_buffer_complex,
                                                               self.MPI_rank_down, tag=TAG_UP_TO_DOWN)
        self.MPI_send_top_complex = self.MPI_comm.Send_init(self.MPI_top_send_buffer_complex,
                                                            self.MPI_rank_up, tag=TAG_DOWN_TO_UP)
        self.MPI_receive_bottom_complex = self.MPI_comm.Recv_init(self.MPI_bottom_receive_buffer_complex,
                                                                  self.MPI_rank_down, tag=TAG_DOWN_TO_UP)
        self.MPI_receive_top_complex = self.MPI_comm.Recv_init(self.MPI_top_receive_buffer_complex,
                                                               self.MPI_rank_up, tag=TAG_UP_TO_DOWN)

        self.MPI_all_requests_complex = [self.MPI_send_left_complex, self.MPI_receive_left_complex,
                                         self.MPI_send_right_complex, self.MPI_receive_right_complex,
                                         self.MPI_send_bottom_complex, self.MPI_receive_bottom_complex,
                                         self.MPI_send_top_complex, self.MPI_receive_top_complex]
        self.pending_requests = None


    def MPI_send_at_edges(self, psi):
        """Start an asynchronous MPI send data from the edges of A to all
        adjacent MPI processes"""
        if psi.dtype == np.float64:
            self.MPI_left_send_buffer_real[:] = psi[LEFT_EDGE]
            self.MPI_right_send_buffer_real[:] = psi[RIGHT_EDGE]
            self.MPI_bottom_send_buffer_real[:] = psi[BOTTOM_EDGE]
            self.MPI_top_send_buffer_real[:] = psi[TOP_EDGE]
            MPI.Prequest.Startall(self.MPI_all_requests_real)
            self.pending_requests = self.MPI_all_requests_real
        elif psi.dtype == np.complex128:
            self.MPI_left_send_buffer_complex[:] = psi[LEFT_EDGE]
            self.MPI_right_send_buffer_complex[:] = psi[RIGHT_EDGE]
            self.MPI_bottom_send_buffer_complex[:] = psi[BOTTOM_EDGE]
            self.MPI_top_send_buffer_complex[:] = psi[TOP_EDGE]
            MPI.Prequest.Startall(self.MPI_all_requests_complex)
            self.pending_requests = self.MPI_all_requests_complex

    def MPI_receive_at_edges(self):
        """Finalise an asynchronous MPI transfer from all adjacent MPI
        processes. Data remains in the receive buffers and can be accessed by
        the caller after this method returns."""
        MPI.Prequest.Waitall(self.pending_requests)
        self.pending_requests = None

    def par_vdot(self, psi1, psi2):
        """"Dots two vectors (with complex comjucation of the first) and sums
        result over MPI processes"""
        local_dot = np.vdot(psi1, psi2)
        local_dot = np.asarray(local_dot, dtype=np.complex128).reshape(1)
        result = np.zeros(1, dtype=np.complex128)
        self.MPI_comm.Allreduce(local_dot, result, MPI.SUM)
        return result[0]

    def par_laplacian_init(self, psi):
        # if not self.pending_requests:
        self.MPI_send_at_edges(psi)
        # Compute laplacian on internal elements:
        result = np.zeros(psi.shape, dtype=psi.dtype)
        result = laplacian(psi, self.dx, self.dy)
        return result

    def par_laplacian_finalise(self, result):
        # if self.pending_requests:
        self.MPI_receive_at_edges()
        # Add contribution on edges from adjacent MPI processes:
        if result.dtype == np.float64:
            result[0, :] += self.MPI_left_receive_buffer_real/self.dx**2
            result[-1, :] += self.MPI_right_receive_buffer_real/self.dx**2
            result[:, 0] += self.MPI_bottom_receive_buffer_real/self.dy**2
            result[:, -1] += self.MPI_top_receive_buffer_real/self.dy**2
        elif result.dtype == np.complex128:
            result[0, :] += self.MPI_left_receive_buffer_complex/self.dx**2
            result[-1, :] += self.MPI_right_receive_buffer_complex/self.dx**2
            result[:, 0] += self.MPI_bottom_receive_buffer_complex/self.dy**2
            result[:, -1] += self.MPI_top_receive_buffer_complex/self.dy**2
        return result

    def par_laplacian(self, psi):
        result = self.par_laplacian_init(psi)
        return self.par_laplacian_finalise(result)

    def fft_laplacian(self, psi):
        if self.MPI_size > 1 or not (self.periodic_x and self.periodic_y):
            msg = "FFT laplacian can only be done in a single node with periodic x and y directions"
            raise RuntimeError(msg)
        return ifft2(self.f_laplacian*fft2(psi))

    def fft_gradx(self, psi):
        if self.MPI_size > 1 or not self.periodic_x:
            msg = "FFT laplacian can only be done in a single node with periodic x direction"
            raise RuntimeError(msg)
        return ifft2(self.f_gradx*fft2(psi))

    def fft_grady(self, psi):
        if self.MPI_size > 1 or not self.periodic_y:
            msg = "FFT laplacian can only be done in a single node with periodic y direction"
            raise RuntimeError(msg)
        return ifft2(self.f_grady*fft2(psi))


class HDFOutput(object):
    def __init__(self, simulator, output_dir):
        self.simulator = simulator
        self.output_dir = output_dir
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

    def save(self, psi, output_log_data):
        if not 'psi' in self.file:
            self.file.create_dataset('psi', (0,) + self.simulator.shape,
                                     maxshape=(None,) + self.simulator.shape,
                                     dtype=psi.dtype)
        if not 'output_log' in self.file:
            self.file.create_dataset('output_log', (0,), maxshape=(None,), dtype=output_log_data.dtype)

        output_log_dataset = self.file['output_log']
        output_log_dataset.resize((len(output_log_dataset) + 1,))
        output_log_dataset[-1] = output_log_data
        psi_dataset = self.file['psi']
        psi_dataset.resize((len(psi_dataset) + 1,) + psi_dataset.shape[1:])
        psi_dataset[-1] = psi

    @staticmethod
    def iterframes(directory, start_frame=0, n_frames=None):
        with h5py.File(os.path.join(directory, '0.h5')) as master_file:
            MPI_size = master_file['MPI_geometry'].attrs['MPI_size']
            shape = master_file.attrs['global_shape']
            dtype = master_file['psi'].dtype
            if n_frames is None:
                n_frames = len(master_file['psi'])
        files = []
        for rank in range(MPI_size):
            basename = str(rank).zfill(len(str(MPI_size))) + '.h5'
            f = h5py.File(os.path.join(directory, basename))
            psi_dataset = f['psi']
            start_x = f['MPI_geometry']['first_x_index'][0]
            start_y = f['MPI_geometry']['first_y_index'][0]
            nx = f['MPI_geometry']['nx'][0]
            ny = f['MPI_geometry']['ny'][0]
            files.append((f, psi_dataset, start_x, start_y, nx, ny))

        for i in range(start_frame, n_frames):
            psi = np.zeros(shape, dtype=dtype)
            for f, psi_dataset, start_x, start_y, nx, ny in files:
                psi[start_x:start_x + nx, start_y:start_y + ny] = psi_dataset[i]
            yield psi


def _pre_step_checks(i, t, psi, output_interval, output_callback, post_step_callback, final_call=False):
    if np.isnan(psi).any() or np.isinf(psi).any():
        raise RuntimeError('It exploded :(')
    if post_step_callback is not None:
        post_step_callback(i, t, psi)
    output_callback_called = False
    if output_callback is not None:
        if np.iterable(output_interval):
            if i in output_interval:
                output_callback(i, t, psi)
                output_callback_called = True
        elif not i % output_interval:
            output_callback(i, t, psi)
            output_callback_called = True
    if final_call and not output_callback_called:
        output_callback(i, t, psi)


def rk4(dt, t_final, dpsi_dt, psi, output_interval=100, output_callback=None, post_step_callback=None):
    """Fourth order Runge-Kutta. dpsi_dt should return an array for the time derivatives of psi."""
    t = 0
    i = 0
    while not t > t_final:
        _pre_step_checks(i, t, psi, output_interval, output_callback, post_step_callback)
        k1 = dpsi_dt(t, psi)
        k2 = dpsi_dt(t + 0.5*dt, psi + 0.5*k1*dt)
        k3 = dpsi_dt(t + 0.5*dt, psi + 0.5*k2*dt)
        k4 = dpsi_dt(t + dt, psi + k3*dt)
        psi[:] += dt/6*(k1 + 2*k2 + 2*k3 + k4)
        t += dt
        i += 1
    _pre_step_checks(i, t, psi, output_interval, output_callback, post_step_callback, final_call=True)


def rk4ilip(dt, t_final, dpsi_dt, psi, omega_imag_provided=False,
            output_interval=100, output_callback=None, post_step_callback=None):
    """Fourth order Runge-Kutta in an instantaneous local interaction picture.
    dpsi_dt should return both the derivative of psi, and an array omega,
    which is H_local/hbar at each point in space (Note that H_local should not
    be excluded from the calculation of dpsi_dt). If omega is purely
    imaginary, you can instead return the omega_imag comprising its imaginary
    part, in which case you should set omega_imag_provided to True. This means
    real arrays can be used for arithmetic instead of complex ones, which is
    faster."""
    t = 0
    i = 0
    while not t > t_final:
        _pre_step_checks(i, t, psi, output_interval, output_callback, post_step_callback)
        if omega_imag_provided:
            # Omega is purely imaginary, and so omega_imag has been provided
            # instead so real arithmetic can be used:
            f1, omega_imag = dpsi_dt(t, psi)
            i_omega = -omega_imag
            i_omega_clipped = i_omega.clip(-400/dt, 400/dt)
            U_half = np.exp(i_omega_clipped*0.5*dt)
        else:
            f1, omega = dpsi_dt(t, psi)
            i_omega = 1j*omega
            if omega.dtype == np.float64:
                theta = omega*0.5*dt
                U_half = np.cos(theta) + 1j*np.sin(theta) # faster than np.exp(1j*theta) when theta is real
            else:
                i_omega_clipped = i_omega.real.clip(-400/dt, 400/dt) + 1j*i_omega.imag
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
        t += dt
        i += 1
    _pre_step_checks(i, t, psi, output_interval, output_callback, post_step_callback, final_call=True)


def successive_overrelaxation(simulator, system, psi, relaxation_parameter=1.7, convergence=1e-13,
                               output_interval=100, output_callback=None, post_step_callback=None,
                               convergence_check_interval=10):
    i = 0
    while True:
        _pre_step_checks(i, 0, psi, output_interval, output_callback, post_step_callback)
        simulator.MPI_send_at_edges(psi)
        A_diag, A_nondiag, b = system(psi)
        if not i % convergence_check_interval:
            # Only compute the error every convergence_check_interval steps to save time
            compute_error=True
            integral_b = simulator.par_vdot(b, b).real
        else:
            compute_error = False
        squared_error = SOR_step_interior(psi, A_diag, A_nondiag, b, simulator.dx, simulator.dy,
                                          relaxation_parameter, compute_error=compute_error)
        simulator.MPI_receive_at_edges()
        if psi.dtype == np.complex128:
            left_buffer = simulator.MPI_left_receive_buffer_complex
            right_buffer = simulator.MPI_right_receive_buffer_complex
            bottom_buffer = simulator.MPI_bottom_receive_buffer_complex
            top_buffer = simulator.MPI_top_receive_buffer_complex
        elif psi.dtype == np.float64:
            left_buffer = simulator.MPI_left_receive_buffer_real
            right_buffer = simulator.MPI_right_receive_buffer_real
            bottom_buffer = simulator.MPI_bottom_receive_buffer_real
            top_buffer = simulator.MPI_top_receive_buffer_real
        squared_error = SOR_step_edges(psi, A_diag, A_nondiag, b, simulator.dx, simulator.dy, relaxation_parameter,
                                       left_buffer, right_buffer, bottom_buffer, top_buffer, squared_error,
                                       compute_error=compute_error)
        if compute_error:
            squared_error = np.asarray(squared_error).reshape(1)
            total_squared_error = np.zeros(1)
            simulator.MPI_comm.Allreduce(squared_error, total_squared_error, MPI.SUM)
            convergence_calc = np.sqrt(total_squared_error[0]/integral_b)
            if convergence_calc < convergence:
                break
        i += 1
    _pre_step_checks(i, 0, psi, output_interval, output_callback, post_step_callback, final_call=True)


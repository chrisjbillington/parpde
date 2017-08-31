# To be run as a single process, not with MPI:
#     python plot.py

from __future__ import division, print_function
import sys
sys.path.insert(0, '../..') # The location of the modules we need to import
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.image
pi = np.pi

from parPDE import HDFOutput

def plot(name, i, psi):
    psi = psi.transpose()
    rho = np.abs(psi)**2
    phase = np.angle(psi)
    hsl = np.zeros(psi.shape + (3,))
    hsl[:, :, 2] = rho/rho.max()
    rgb = matplotlib.colors.hsv_to_rgb(hsl)
    hsl[:, :, 0] = np.array((phase + pi)/(2*pi))
    hsl[:, :, 1] = 0.33333
    rgb = matplotlib.colors.hsv_to_rgb(hsl)
    matplotlib.image.imsave('%s/%04d.png' % (name, i),  rgb, origin='lower')


def plot_sim(name):
    output_dir = name + '_images'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for i, psi in HDFOutput.iterframes(name, step=1):
        print(name, i)
        plot(output_dir, i, psi)

if __name__ == '__main__':
    plot_sim('groundstate')
    plot_sim('smoothing')
    plot_sim('evolution')

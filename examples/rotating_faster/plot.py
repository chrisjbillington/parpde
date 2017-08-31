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
    hsl[:, :, 1] = 0.2
    rgb = matplotlib.colors.hsv_to_rgb(hsl)
    matplotlib.image.imsave('%s/%04d.png' % (name, i),  rgb, origin='lower')


def plot_sim(name):
    output_dir = name + '_images'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    listdir = os.listdir(output_dir)
    if listdir:
        up_to = max(int(name.strip('.png')) for name in listdir)
    else:
        up_to=0
    for i, psi in HDFOutput.iterframes(name, start=up_to, step=5):
        import IPython
        IPython.embed()
        print(name, i)
        plot(output_dir, i, psi)

if __name__ == '__main__':
    while True:
        plot_sim('groundstate')
        import time
        time.sleep(10)
    # plot_sim('smoothing')

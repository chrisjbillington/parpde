# To be run as a single process, not with MPI:
#     python plot.py
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.image
pi = np.pi

from parPDE import HDFOutput

def plot(imgdir, i, psi):
    # Transpose because our arrays have the x dimension first. Index 0 since we only
    # want to plot the field, component 1 is its time derivative.
    u = psi[0].transpose()

    # psi = psi.transpose()
    # rho = np.abs(psi)**2
    # phase = np.angle(psi)
    # hsl = np.zeros(psi.shape + (3,))
    # hsl[:, :, 2] = rho/rho.max()
    # rgb = matplotlib.colors.hsv_to_rgb(hsl)
    # hsl[:, :, 0] = np.array((phase + pi)/(2*pi))
    # hsl[:, :, 1] = 0.33333
    # rgb = matplotlib.colors.hsv_to_rgb(hsl)
    # matplotlib.image.imsave(f'{imgdir}/{i:04d}.png',  rgb, origin='lower')

    # Pretty colourful output for phase info not relevant for a real-valued field,
    # just plot with cmap=seismic instead.

    matplotlib.image.imsave(
        f'{imgdir}/{i:04d}.png', u, origin='lower', vmin=-0.5, vmax=0.5, cmap='seismic'
    )

def make_movie(name, imgdir):
    import subprocess
    subprocess.call(
        ['ffmpeg', '-i', os.path.join(imgdir, '%04d.png'), f'{name}.avi']
    )

def plot_sim(name):
    imgdir = name + '_images'
    if not os.path.isdir(imgdir):
        os.mkdir(imgdir)
    for i, psi in HDFOutput.iterframes(name, step=1):
        print(name, i)
        plot(imgdir, i, psi)
    make_movie(name, imgdir)

if __name__ == '__main__':
    plot_sim('evolution')

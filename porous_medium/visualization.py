#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this file implements visualizations functions for visualizations
# of the state of the porous medium equation
#
#

import jax.numpy as jnp

import matplotlib.pyplot as plt
from matplotlib import cm

from helpers.other import mpl_settings
from damped_wave.basisfunctions import cache_basis_functions
from helpers.ph import PortHamiltonian

# apply settings for plots
# mpl_settings(backend='macosx')

def plot_3d_state(
        tt: jnp.ndarray,
        cc: jnp.ndarray,
        ph_sys: PortHamiltonian,
        title: str = 'functions',
        savename: str = None,
        subfigure_titles: tuple = None,
        ):
    """
    cc is an array of the form

    cc = [c(0), ..., c(T)],

    where c(0), ... are from R^D and correspond to coefficients
    for a FEM discretization of a pde.

    the corresponding function is plotted over time in a 3d plot.

    :param tt: array of timepoints, shape (nt,)
    :param cc: array of coefficient values at timepoints, shape (nt,D)
    :param ph_sys: pH system to which the coefficient values in cc belong
    :param title: title for the figure
    :param savename: savename of the figure, None if no saving
    :param subfigure_titles: (optional) titles for the subfigure as tuple of strings
    :return:
    """
    # step 0: run matplotlib settings
    # mpl_settings(figsize=(11,4))

    # step 1: obtain information
    nt = tt.shape[0]
    D = cc.shape[1]
    assert cc.shape == (nt, D)
    l = ph_sys.info['constants']['l']

    n_inner = ph_sys.info['inner_grid_points']
    d_z = n_inner + 2

    # step 2: evaluate real functions using the basis functions
    nx = 100
    xx = jnp.linspace(0,l,nx)
    info = {
        'N': n_inner+1,
        'T': tt[-1],
        'D': 1,
        'nt_cache': nx,
        }
    z = jnp.zeros((nt,nx))

    cache = cache_basis_functions(info)
    phi = cache['phi']
    psi = cache['psi']

    # for each timepoint
    for t in range(nt):

        # take coefficients
        cc_t = cc[t,:]

        # get function values for these coefficients
        ff_t = jnp.zeros((nx,))

        for j, coeff in enumerate(cc_t):
            ff_t += coeff*psi[j][0].reshape((-1,))

        # now ff_rho_t and ff_m_t store the function values to
        # these coefficients

        # put into storage
        z = z.at[t,:].set(ff_t)


    # plot z
    # setup
    fig = plt.figure()
    ax_z = fig.add_subplot(111, projection='3d')

    # data
    T, X = jnp.meshgrid(tt, xx, indexing='ij')

    ax_z.plot_surface(T, X, z, cmap=cm.coolwarm)

    ax_z.set_title('$z$')

    # fix weird autoscale issue
    z_avg = jnp.average(z)
    ax_z.set_zlim(jnp.min(z)-0.2*z_avg,jnp.max(z)+0.2*z_avg)

    plt.subplots_adjust(
        left=0.08,
        right=0.88,
        wspace=0.55,
        )

    ax_z.set_xlabel('time')
    ax_z.set_ylabel('space')
    ax_z.set_zlabel('function value')

    if savename is not None:
        plt.savefig(savename)
        print(f'figure saved under savename {savename}')

    fig.suptitle(title)

    # plt.subplot_tool()
    plt.show(block=False)

    return






if __name__ == '__main__':

    pass
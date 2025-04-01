#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this file implements visualizations functions for visualizations
# of pressure / massflow for the nonlinearly damped wave model
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

def plot_3d_wave(
        tt: jnp.ndarray,
        cc: jnp.ndarray,
        ph_sys: PortHamiltonian,
        title: str = 'functions',
        savename: str = None,
        subfigure_titles: tuple = None,
        ):
    """
    if cc is of shape (nt,D), then it is assumed that
    the first (D-3)//2 + 1 components belong to the first
    function stored in cc (e.g. density rho), and that
    the other (D-3)//2 + 3 components belong to the second
    function stored in cc (e.g. massflow m).

    in other words, cc is an array of the form

    cc = [c(0), ..., c(T)],

    where c(0), ... are from R^D and correspond to coefficients
    for a FEM discretization of a pde.

    for each c in cc, this method assembles two functions
    that correspond to the coefficients in c.

    these functions are then plotted over time in a 3d plot.

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
    d_rho = n_inner + 1
    d_m = n_inner + 2

    c_rho, c_m = cc[:,:d_rho], cc[:,d_rho:]
    # now c_rho is of shape (nt, d_rho)
    #     c_m   is of shape (nt, d_m)

    # step 2: evaluate real functions using the basis functions
    nx = 100
    xx = jnp.linspace(0,l,nx)
    info = {
        'N': n_inner+1,
        'T': tt[-1],
        'D': 1,
        'nt_cache': nx,
        }
    rho = jnp.zeros((nt,nx))
    m = jnp.zeros((nt,nx))

    cache = cache_basis_functions(info)
    phi = cache['phi']
    psi = cache['psi']

    # for each timepoint
    for t in range(nt):

        # take coefficients
        c_rho_t, c_m_t = c_rho[t,:], c_m[t,:]

        # get function values for these coefficients
        ff_rho_t = jnp.zeros((nx,))
        ff_m_t = jnp.zeros((nx,))

        for i, coeff in enumerate(c_rho_t):
            ff_rho_t += coeff*phi[i][0].reshape((-1,))
        for j, coeff in enumerate(c_m_t):
            ff_m_t += coeff*psi[j][0].reshape((-1,))

        # now ff_rho_t and ff_m_t store the function values to
        # these coefficients

        # put into storage
        rho = rho.at[t,:].set(ff_rho_t)
        m = m.at[t,:].set(ff_m_t)


    # plot rho and m
    # setup
    fig = plt.figure()
    ax_rho = fig.add_subplot(121, projection='3d')
    ax_m = fig.add_subplot(122, projection='3d')

    # data
    T, X = jnp.meshgrid(tt, xx, indexing='ij')

    ax_rho.plot_surface(T, X, rho, cmap=cm.coolwarm)
    ax_m.plot_surface(T, X, m, cmap=cm.coolwarm)

    if subfigure_titles is None:
        subfigure_titles = ('$\\rho$', '$v$')
    ax_rho.set_title(subfigure_titles[0])
    ax_m.set_title(subfigure_titles[1])

    # fix weird autoscale issue
    rho_avg = jnp.average(rho)
    ax_rho.set_zlim(jnp.min(rho)-0.2*rho_avg,jnp.max(rho)+0.2*rho_avg)
    m_avg = jnp.average(m)
    ax_m.set_zlim(jnp.min(m)-0.2*m_avg,jnp.max(m)+0.2*m_avg)

    plt.subplots_adjust(
        left=0.08,
        right=0.88,
        wspace=0.55,
        )

    for ax in [ax_rho,ax_m]:
        ax.set_xlabel('time')
        ax.set_ylabel('space')
        ax.set_zlabel('function value')
        # ax.legend()

    if savename is not None:
        plt.savefig(savename)
        print(f'figure saved under savename {savename}')

    fig.suptitle(title)

    # plt.subplot_tool()
    plt.show(block=False)

    return



def plot_3d_comparison(
        tt_1: jnp.ndarray,
        cc_1: jnp.ndarray,
        tt_2: jnp.ndarray,
        cc_2: jnp.ndarray,
        ph_sys: PortHamiltonian,
        quantity_to_plot: str = 'rho',
        title: str = 'functions',
        savename: str | None = None,
        ):
    """
    this method is very similar to plot_3d_wave, but accepts two
    coefficient vectors (and time point arrays) instead of one.
    the method plots the chosen quantity (rho or v) for both coefficient
    vector arrays next to each other

    :param tt_1: array of timepoints for first coeffiecent array, shape (nt_1,)
    :param cc_1: first array of coefficient values at timepoints, shape (nt_1,D)
    :param tt_2: array of timepoints for second coeffiecent array, shape (nt_2,)
    :param cc_2: second array of coefficient values at timepoints, shape (nt_2,D)
    :param ph_sys: pH system to which the coefficient values in cc belong
    :param quantity_to_plot: (optional) quantity to plot next to each other, default 'rho'
    :param title: (optional) title for the figure
    :param savename: (optional) savename of the figure, None if no saving
    :return:
    """

    # step 0: run matplotlib settings
    # mpl_settings(figsize=(11,4))

    # step 1: obtain information
    nt_1 = tt_1.shape[0]
    nt_2 = tt_2.shape[0]
    D = cc_1.shape[1]
    assert cc_1.shape == (nt_1, D)
    assert cc_2.shape == (nt_2, D)
    l = ph_sys.info['constants']['l']

    n_inner = ph_sys.info['inner_grid_points']
    d_rho = n_inner + 1
    d_m = n_inner + 2

    c_rho_1, c_m_1 = cc_1[:,:d_rho], cc_1[:,d_rho:]
    c_rho_2, c_m_2 = cc_2[:,:d_rho], cc_2[:,d_rho:]
    # now c_rho_i is of shape (nt_i, d_rho)
    #     c_m_i   is of shape (nt_i, d_m)

    # step 2: evaluate real functions using the basis functions
    nx = 100
    xx = jnp.linspace(0,l,nx)
    info = {
        'N': n_inner+1,
        'T': tt_1[-1],
        'D': 1,
        'nt_cache': nx,
        }
    rho_1 = jnp.zeros((nt_1,nx))
    m_1 = jnp.zeros((nt_1,nx))
    rho_2 = jnp.zeros((nt_2,nx))
    m_2 = jnp.zeros((nt_2,nx))

    cache = cache_basis_functions(info)
    phi = cache['phi']
    psi = cache['psi']

    # for each timepoint in nt_1
    for t in range(nt_1):

        # take coefficients
        c_rho_t_1, c_m_t_1 = c_rho_1[t,:], c_m_1[t,:]

        # get function values for these coefficients
        ff_rho_t_1 = jnp.zeros((nx,))
        ff_m_t_1 = jnp.zeros((nx,))

        for i, coeff in enumerate(c_rho_t_1):
            ff_rho_t_1 += coeff*phi[i][0].reshape((-1,))
        for j, coeff in enumerate(c_m_t_1):
            ff_m_t_1 += coeff*psi[j][0].reshape((-1,))

        # now ff_rho_t and ff_m_t store the function values to
        # these coefficients

        # put into storage
        rho_1 = rho_1.at[t,:].set(ff_rho_t_1)
        m_1 = m_1.at[t,:].set(ff_m_t_1)

    # for each timepoint in nt_2
    for t in range(nt_2):

        # take coefficients
        c_rho_t_2, c_m_t_2 = c_rho_2[t,:], c_m_2[t,:]

        # get function values for these coefficients
        ff_rho_t_2 = jnp.zeros((nx,))
        ff_m_t_2 = jnp.zeros((nx,))

        for i, coeff in enumerate(c_rho_t_2):
            ff_rho_t_2 += coeff*phi[i][0].reshape((-1,))
        for j, coeff in enumerate(c_m_t_2):
            ff_m_t_2 += coeff*psi[j][0].reshape((-1,))

        # now ff_rho_t and ff_m_t store the function values to
        # these coefficients

        # put into storage
        rho_2 = rho_2.at[t,:].set(ff_rho_t_2)
        m_2 = m_2.at[t,:].set(ff_m_t_2)


    # plot rho and m
    # setup
    fig = plt.figure()
    ax_1 = fig.add_subplot(121, projection='3d')
    ax_2 = fig.add_subplot(122, projection='3d')

    # data
    T_1, X_1 = jnp.meshgrid(tt_1, xx, indexing='ij')
    T_2, X_2 = jnp.meshgrid(tt_2, xx, indexing='ij')

    if quantity_to_plot == 'rho':
        ax_1.set_title(r'$\rho$')
        ax_2.set_title(r'$\rho$')
        ax_1.plot_surface(T_1, X_1, rho_1, cmap=cm.coolwarm)
        ax_2.plot_surface(T_2, X_2, rho_2, cmap=cm.coolwarm)

        # fix weird autoscale issue
        rho_avg_1 = jnp.average(rho_1)
        ax_1.set_zlim(jnp.min(rho_1)-0.2*rho_avg_1,jnp.max(rho_1)+0.2*rho_avg_1)
        rho_avg_2 = jnp.average(rho_2)
        ax_2.set_zlim(jnp.min(rho_2)-0.2*rho_avg_2,jnp.max(rho_2)+0.2*rho_avg_2)

    else:
        ax_1.set_title(r'$v$')
        ax_2.set_title(r'$v$')
        ax_1.plot_surface(T_1, X_1, m_1, cmap=cm.coolwarm)
        ax_2.plot_surface(T_2, X_2, m_2, cmap=cm.coolwarm)

        # fix weird autoscale issue
        m_avg_1 = jnp.average(m_1)
        ax_1.set_zlim(jnp.min(m_1)-0.2*m_avg_1,jnp.max(m_1)+0.2*m_avg_1)
        m_avg_2 = jnp.average(m_2)
        ax_2.set_zlim(jnp.min(m_2)-0.2*m_avg_2,jnp.max(m_2)+0.2*m_avg_2)

    plt.subplots_adjust(
        left=0.08,
        right=0.88,
        wspace=0.55,
        )

    for ax in [ax_1,ax_2]:
        ax.set_xlabel('time')
        ax.set_ylabel('space')
        ax.set_zlabel('function value')
        # ax.legend()

    if savename is not None:
        plt.savefig(savename)
        print(f'figure saved under savename {savename}')

    fig.suptitle(title)

    # plt.subplot_tool()
    plt.show(block=False)

    return






if __name__ == '__main__':

    pass
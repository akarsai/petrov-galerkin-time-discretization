#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this file implements functions to obtain the matrices for the
# port-hamiltonian semidiscretization of a porous medium equation.
# please see the corresponding pdf/latex file (porous-medium-details.pdf).
#
# important variables:
#     n : number of inner grid points
#
# physical constants:
#     l : length of space interval
#
# other parameters:
#   eps : regularization parameter
#     q : parameter
#
#

import jax
import jax.numpy as jnp
from scipy.special import roots_legendre

from helpers.gauss import gauss_quadrature_with_values
from damped_wave.basisfunctions import get_psi_slim

from helpers.ph import PortHamiltonian, PortHamiltonian_LinearEQB, PortHamiltonian_LinearEJRB


# default_constants = {
#     'l':     1,  # length of space domain
#     }


# ----------------------------------------
# ------------- for E matrix -------------
# ----------------------------------------


def get_E(
        n: int,
        hgrid: float,
        ) -> jnp.ndarray:

    # assemble mass matrix for piecewise linear hat functions

    diag = (2/3*jnp.ones((n+2,))).at[0].set(1/3).at[-1].set(1/3)
    offdiag = 1/6*jnp.ones((n+1,))

    E = jnp.diag(diag,k=0) \
        + jnp.diag(offdiag,k=1) \
        + jnp.diag(offdiag,k=-1)

    return hgrid*E


# ----------------------------------------
# ------------- for S matrix -------------
# ----------------------------------------

def get_S(
        n: int,
        hgrid: float,
        ) -> jnp.ndarray:
    """
    calculates stiffness matrix
    """

    ### for stiffness matrix S
    diag = (2*jnp.ones((n+2,))).at[0].set(1).at[-1].set(1)
    offdiag = -jnp.ones((n+1,))

    S = (
            jnp.diag(diag,k=0)
            + jnp.diag(offdiag,k=1)
            + jnp.diag(offdiag,k=-1)
        )

    return 1/hgrid * S



# ----------------------------------------
# --------------- for eta ----------------
# ----------------------------------------

def get_eta(
        n: int,
        hgrid: float,
        wh: jnp.ndarray,
        psi_values_at_gauss_points: jnp.ndarray,
        eta_nondiscrete: callable,
        gauss_weights: jnp.ndarray
        ) -> jnp.ndarray:

    ### get mass matrix using get_E
    E = get_E(n, hgrid)

    ### for Pi_h eta = e_h

    # compute zh at gauss_points
    zh_at_gauss_points = jnp.einsum('i,gIi->gI', wh, psi_values_at_gauss_points)

    # eta_nondiscrete(zh) * psi_k for all k and all intervals at once
    eta_psi_at_gauss_points = jnp.einsum('gI, gIJ -> gIJ', eta_nondiscrete(zh_at_gauss_points), psi_values_at_gauss_points) # is shape (ngp, n+1, n+2)

    # this contains all integrals over the intervals for all k and subsequently has shape (n+1, n+2)
    eta_psi_unsummed = gauss_quadrature_with_values(gauss_weights, fvalues=eta_psi_at_gauss_points, length=hgrid) # has shape (n+1, n+2)

    # summing along first axis gives eta_F
    eta_psi = jnp.sum(eta_psi_unsummed, axis=0)

    # return E^-1 * eta_F
    return jnp.linalg.solve(E, eta_psi)


# ----------------------------------------
# ----------- for hamiltonian ------------
# ----------------------------------------

def get_hamiltonian(
        hgrid: float,
        eps: float,
        q: float,
        wh: jnp.ndarray,
        psi_values_at_gauss_points: jnp.ndarray,
        gauss_weights: jnp.ndarray,
        ) -> jnp.ndarray:
    """
    computes the hamiltonian of the space discretized system, which reads as

    H_h (w_h) = H(z_h),

    where z_h is the piecewise linear interpolation of w_h and H is the hamiltonian
    of the space-continuous system which reads as

    H(z) = int_{0}^{\ell} [ 1/(q+1) * |z|**(q+1) + eps/2 * |z|**2 ] dx.
    """

    # compute zh at gauss_points
    zh_at_gauss_points = jnp.einsum('i,gIi->gI', wh, psi_values_at_gauss_points) # switch from wh to zh formulation

    # calculate function values for integral terms
    fvalues = 1/(q+1)*jnp.abs(zh_at_gauss_points)**(q+1) + eps/2 * jnp.abs(zh_at_gauss_points)**2

    # this contains all integrals over the intervals for all k and subsequently has shape (n+1, n+2)
    integral_unsummed = gauss_quadrature_with_values(gauss_weights, fvalues=fvalues, length=hgrid) # has shape (n+1, n+2)

    # summing along first axis gives integral over the whole domain
    return jnp.sum(integral_unsummed, axis=0)




# ----------------------------------------
# --------------- combined ---------------
# ----------------------------------------


# helper function
def plot_matrix(A,title=''):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(A,cmap='nipy_spectral')
    fig.colorbar(cax)
    plt.title(title)
    plt.show()

    return


def get_example_system(
        options: dict = None,
        q=2,
        eps=1e-10,
        ) -> PortHamiltonian_LinearEJRB |  PortHamiltonian_LinearEQB:

    # default constants
    constants = {'l': 15, 'q': q, 'eps': eps}
    inner_grid_points = 14
    num_gauss_nodes = 10

    # retrieve from options
    if options is not None:
        constants = options.get('constants',constants)
        inner_grid_points = options.get('inner_grid_points',inner_grid_points)
        num_gauss_nodes = options.get('num_gauss_nodes',num_gauss_nodes)

    # space-continuous eta
    eta_nondiscrete = lambda z: jnp.abs(z)**(q-1) * z + eps*z

    # prepare gauss quadrature
    l = constants['l']
    xi = jnp.linspace(0,l,inner_grid_points+2)
    gauss_points, gauss_weights = roots_legendre(num_gauss_nodes)
    gauss_points, gauss_weights = jnp.array(gauss_points), jnp.array(gauss_weights)

    # build cache for S -- naive and really slow
    psi_values_at_gauss_points = jnp.zeros( (num_gauss_nodes, inner_grid_points+1, inner_grid_points+2) )

    onehat = get_psi_slim(j=2, info={'T': l, 'N': inner_grid_points+1}) # j = 1, ..., n+2

    increasing_front_at_gauss_nodes = onehat((xi[1]-xi[0])/2 * gauss_points + (xi[0]+xi[1])/2)
    decreasing_front_at_gauss_nodes = increasing_front_at_gauss_nodes[::-1]

    for I in range(inner_grid_points+1):
        psi_values_at_gauss_points = psi_values_at_gauss_points.at[:,I,I].set(decreasing_front_at_gauss_nodes)
        psi_values_at_gauss_points = psi_values_at_gauss_points.at[:,I,I+1].set(increasing_front_at_gauss_nodes)
    dim_sys = inner_grid_points + 2
    hgrid = constants.get('l')/(inner_grid_points+1)

    E = get_E(inner_grid_points, hgrid)
    J_matrix = jnp.zeros((inner_grid_points+2,inner_grid_points+2))
    R_matrix = get_S(inner_grid_points, hgrid)
    B = jnp.zeros((inner_grid_points+2,1))

    info = {
        'type': 'porous_medium',
        'dim_sys': dim_sys,
        'constants': constants,
        'inner_grid_points': inner_grid_points,
        'hgrid': hgrid,
        }

    eta = lambda wh: \
        get_eta(inner_grid_points, hgrid, wh, psi_values_at_gauss_points, eta_nondiscrete, gauss_weights)

    eta_vmap = jax.vmap(
        eta,
        in_axes = 0, # 0 -> index where time array is
        out_axes = 0, # 0 -> index where time array is
        )

    hamiltonian = lambda wh: \
        get_hamiltonian(
            hgrid,
            eps,
            q,
            wh,
            psi_values_at_gauss_points,
            gauss_weights,
            )

    hamiltonian_vmap = jax.vmap(
        hamiltonian,
        in_axes = 0, # 0 -> index where time array is
        out_axes = 0, # 0 -> index where time array is
        )

    # for eta_inv
    # -> eta_inv does not appear in the time discretization since J and R are constant
    # -> as a consequence, we can set it arbitrarily
    eta_inv = lambda wh: wh

    ph_sys = PortHamiltonian_LinearEJRB(
        E=E,
        J=J_matrix,
        R=R_matrix,
        eta=eta_vmap,
        eta_inv=eta_inv,
        B=B,
        hamiltonian=hamiltonian_vmap,
        info=info,
        )

    # add Hprime for test purposes
    ph_sys.hamiltonian_prime = jax.vmap(jax.grad(hamiltonian), in_axes=0, out_axes=0)

    return ph_sys

def get_example_initial_state_and_control(
        ph_sys: PortHamiltonian_LinearEJRB |  PortHamiltonian_LinearEQB
        ) -> (jnp.ndarray, callable):

    dim_sys = ph_sys.info.get('dim_sys')
    nsys = ph_sys.info.get('inner_grid_points')
    l = ph_sys.info['constants']['l']
    inner_grid_points = ph_sys.info['inner_grid_points']

    # initial value of barenblatt solution taken from eq. (6.2) from
    # Pop, I. S. and Yong, W.-A., A numerical approach to degenerate parabolic equations, DOI 10.1007/s002110100330
    d = 1 # dimension of spacial domain
    q = ph_sys.info['constants']['q'] # parameter in porous medium equation \partial_t z = \Delta( z^q )
    m = q
    def z(t,x):
        xshift = x-l/2 # shift x so that solution corresponds to space domain [-l/2, l/2]
        fac = m*d + 2 - d
        val = jax.nn.relu( 1 - ( (1/2 - 1/(2*m) ) * xshift**2 )/( fac * (t+1)**(2/fac) ) )**(1/(m-1))
        return 1/((t+1)**(d/fac)) * val

    # true solution evaluated at discretized space points
    xi = jnp.linspace(0,l,inner_grid_points+2)
    # print(xi)

    # only scalar t allowed
    def z_discret(t):
        return z(t, xi)

    z0 = z_discret(0.)
    # print(z0)

    # control values
    def control(t):
        return jnp.zeros((1,))
    control = jax.vmap(control, in_axes=0, out_axes=0,) # 0 = index where time is

    return z0, control

def get_example_manufactured_solution(
        ph_sys: PortHamiltonian_LinearEJRB | PortHamiltonian_LinearEQB,
        kind: str = 'barenblatt',
        ) -> (jnp.ndarray, callable, callable, callable):

    inner_grid_points = ph_sys.info['inner_grid_points']
    l = ph_sys.info['constants']['l']
    q = ph_sys.info['constants']['q']

    # control does not matter
    def control(t):
        return jnp.zeros((1,))
    control = jax.vmap(control, in_axes=0, out_axes=0,) # 0 = index where time is

    # what we want as the true solution z

    if kind == 'barenblatt':
        # here: barenblatt solution taken from eq. (6.2) from
        # Pop, I. S. and Yong, W.-A., A numerical approach to degenerate parabolic equations, DOI 10.1007/s002110100330
        d = 1 # dimension of spacial domain
        m = q # parameter in porous medium equation \partial_t z = \Delta( z^m )
        def z(t,x):
            xshift = x-l/2 # shift x so that solution corresponds to space domain [-l/2, l/2]
            fac = m*d + 2 - d
            val = jax.nn.relu( 1 - ( (1/2 - 1/(2*m) ) * xshift**2 )/( fac * (t+1)**(2/fac) ) )**(1/(m-1))
            return 1/((t+1)**(d/fac)) * val

    else: # kind == 'smooth':
        # alternative test case
        def z(t,x):
            return jnp.cos(t) * jnp.sin(x)

    # partial derivative of z with respect to x
    # dx_z = jax.jit(jax.jacobian(z, argnums=1))

    # true solution evaluated at discretized space points
    xi = jnp.linspace(0,l,inner_grid_points+2)

    # only scalar t allowed
    def z_discret(t):
        return z(t, xi)

    z0 = z_discret(0.)

    # only scalar t allowed
    dt_z_discret = jax.jacobian(z_discret, argnums=0)

    # space discretized and vmapped in time
    z_discret = jax.vmap(z_discret, in_axes=0, out_axes=0,)
    dt_z_discret = jax.vmap(dt_z_discret, in_axes=0, out_axes=0,)

    # rho_discret = jax.vmap(rho_discret, in_axes=0, out_axes=0,)
    # v_discret = jax.vmap(v_discret, in_axes=0, out_axes=0,)

    # define rhs for manufactured solution
    E = ph_sys.E_constant # constant
    J = ph_sys.J # callable, always zero
    R = ph_sys.R # callable, possibly non-constant, depending on where the nonlinearity is
    eta = ph_sys.eta # callable, possibly non-linear, depending on where the nonlinearity is
    B = ph_sys.B_constant # constant

    def g(t):
        E_dtz = jnp.einsum('xn, tn -> tx', E, dt_z_discret(t))
        JmR_eta_z = jnp.einsum('txn, tn -> tx', J(z_discret(t)) - R(z_discret(t)), eta(z_discret(t)))
        B_u = jnp.einsum('nm, tm -> tn', B, control(t))

        return E_dtz - (JmR_eta_z + B_u)
        # return E_dtz - JmR_eta_z

    return z0, control, g, z_discret


if __name__ == '__main__':

    jax.config.update("jax_enable_x64", True)

    from helpers.other import mpl_settings
    mpl_settings()

    import matplotlib.pyplot as plt
    from timeit import default_timer as timer
    from spp import spp

    from porous_medium.visualization import plot_3d_state

    # get example system
    ph_sys = get_example_system(
        q=2,
        eps=0.5,
        options={'inner_grid_points': 25} # 9 -> hgrid = 1
        ) # nonlinearity_in is an optional argument
    igp = ph_sys.info['inner_grid_points']

    # get manufactured solution
    z0, control, g, z_discret = get_example_manufactured_solution(ph_sys, kind='smooth')
    zero_control = jax.vmap(lambda t: jnp.zeros((igp+2,)), in_axes=0, out_axes=0)

    # test hamiltonian and eta
    # hgrid = ph_sys.info['constants']['l']/(igp+1)
    # Hprime = ph_sys.hamiltonian_prime
    # # testvector = 3*jnp.linspace(1,5,igp+2).reshape((1,-1))
    # # testvector = 3*jnp.ones((igp+2,)).reshape((1,-1))
    # testvector = jnp.sin(jnp.linspace(0,2*jnp.pi,igp+2)).reshape((1,-1))
    # Minv = jnp.linalg.inv(ph_sys.E_constant)
    # h = Hprime(testvector).reshape((-1,))
    # Minvh = Minv @ h
    # e = ph_sys.eta(testvector).reshape((-1,))
    # Me = ph_sys.E_constant.T @ e
    # Minve = Minv.T @ e
    # norm_e = jnp.linalg.norm(e)
    # diff_Minvh = jnp.linalg.norm(Minvh-e)/norm_e
    # diff_Me = jnp.linalg.norm(h-Me)/norm_e
    # print(f'\n\nsystem dimension = {igp+2} ({igp} spatial grid points, hgrid = {hgrid})\n')
    # print(f'testvector =\n{testvector.reshape((-1,))}\n')
    # print(f'||M^-1 * Hprime - eta||/||eta|| = {diff_Minvh:.4e}\n')
    # print(f'||Hprime - M^T * eta||/||eta|| = {diff_Me:.4e}\n')

    # solve
    T = 5
    nt_spp = 501
    tt_spp = jnp.linspace(0,T,nt_spp)
    spp_degree = 4
    num_proj_nodes = 2*spp_degree # default choice for porous_medium system
    # num_quad_nodes = 2*spp_degree + 1 # this does not seem to matter

    s_spp = timer()
    spp_solution = spp(ph_sys=ph_sys, tt=tt_spp, z0=z0, u=control, degree=spp_degree, num_proj_nodes=num_proj_nodes,
                       # g=g
                       )
    e_spp = timer()
    print(f'\nspp done (igp={igp}, nt_spp={nt_spp}, degree={spp_degree}), took {e_spp-s_spp:.2f} seconds')

    # plot
    # tt_fine, cc_fine = spp_solution['superfine']
    # plot_3d_state(
    #     tt = tt_fine,
    #     cc = cc_fine,
    #     ph_sys = ph_sys,
    #     title = f'\\noindent solution to the porous medium equation\\\\with $k={spp_degree}$',
    #     )

    # energybalance
    from helpers.visualization import calculate_error_in_energybalance
    error_in_energybalance = calculate_error_in_energybalance(spp_solution, ph_sys, u=control, relative=True,)

    plt.semilogy(
        tt_spp[1:], error_in_energybalance,
        label=f'$k = {spp_degree},~ s_{{\Pi}} = {num_proj_nodes}$',
        color=plt.cm.tab20(0)
        )
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('relative error in energy balance')
    plt.ylim(1.5e-18, 1.5e-3)
    plt.show()


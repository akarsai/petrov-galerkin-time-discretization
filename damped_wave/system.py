#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this file implements functions to obtain the matrices for the
# port-hamiltonian semidiscretization of an damped wave equation model.
# please see the corresponding pdf/latex file (damped-wave-details.pdf).
#
# important variables:
#     n : number of inner grid points
#
# physical variables:
#   rho : density
#     v : velocity
#
# physical constants:
#     l : length of space interval
# gamma : friction coefficient
#
# other parameters:
# p(rho): nonlinear pressure function
#  f(v) : nonlinear damping function
#
#

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.special import roots_legendre

from helpers.gauss import gauss_quadrature_with_values
from damped_wave.basisfunctions import get_psi_slim

from helpers.ph import PortHamiltonian_LinearEJB


default_constants = {
    'l':     1,  # length of space domain
    'gamma': 1,  # friction coefficient
    }


# ----------------------------------------
# ------------- for E matrix -------------
# ----------------------------------------


def get_M22(n: int,
            hgrid: float) -> jnp.ndarray:

    # assemble mass matrix for piecewise linear hat functions

    diag = 2/3*jnp.ones((n+2,))
    diag = diag.at[0].set(1/3).at[-1].set(1/3)
    offdiag = 1/6*jnp.ones((n+1,))

    M = jnp.diag(diag,k=0) \
        + jnp.diag(offdiag,k=1) \
        + jnp.diag(offdiag,k=-1)

    return hgrid*M


def get_E(n: int,
          hgrid: float) -> jnp.ndarray:

    Z = jnp.zeros((n+1,n+2))

    M11 = hgrid * jnp.eye(n+1)
    M22 = get_M22(n, hgrid)

    E = jnp.block( [ [M11, Z], [Z.T, M22] ] )

    return E



# ----------------------------------------
# ------------- for J matrix -------------
# ----------------------------------------

def get_D(n: int) -> jnp.ndarray:

    D = (jnp.diag(-jnp.ones((n+2,)),k=0) + jnp.diag(jnp.ones((n+1,)),k=1))[:-1,:]

    return D

def get_J(n: int) -> jnp.ndarray:

    # make matrices
    Z1 = jnp.zeros((n+1,n+1))
    Z2 = jnp.zeros((n+2,n+2))
    D = get_D(n)

    # make first row of J
    J1 = jnp.hstack((Z1,-D))

    # make second row of J
    J2 = jnp.hstack((D.T,Z2))

    # stack them
    J = jnp.vstack((J1,J2))

    # print(J)

    return J




# ----------------------------------------
# ------------- for R matrix -------------
# ----------------------------------------

def get_RF(
        n: int,
        hgrid: float,
        beta: jnp.ndarray,
        psi_values_at_gauss_points: jnp.ndarray,
        g: callable,
        gauss_weights: jnp.ndarray,
        ) -> jnp.ndarray:
    """
    calculates R_F(beta) for coefficients in beta
    """

    # psi_values_at_gauss_points has shape (ngp, n+1, n+2 )
    #  = (num gauss points, num intervals, num basis functions)
    #  = (g, I, i)  <- indices

    # g = gauss points, I = interval, i = index of basis function
    vh = jnp.einsum('i,gIi->gI', beta, psi_values_at_gauss_points)
    g_vh = g(vh) # has shape (ngp, n+1)

    # general scheme:
    # g_vh_on_interval_I = g_vh[:,I] # shape (ngp,)

    diag = jnp.zeros((n+2,))

    first_diagonal_entry = \
        gauss_quadrature_with_values(gauss_weights, fvalues=g_vh[:,0] * psi_values_at_gauss_points[:,0,0]**2, length=hgrid)
    last_diagonal_entry = \
        gauss_quadrature_with_values(gauss_weights, fvalues=g_vh[:,-1] * psi_values_at_gauss_points[:,-1,-1]**2, length=hgrid)

    # this array stores values of g(v_h) psi_i**2 on the gauss points in the interval [xi_{I}, xi_{I+1}]
    # g = gauss points, I = interval, i = index of basis function
    # we only need values for the case I = i-2 and I = i-1, and i = [1:-1]
    values_for_diagonal_integrals_1 = jnp.einsum('gI, gII -> gI', g_vh[:,:-1], psi_values_at_gauss_points[:,:-1,1:-1]**2)
    values_for_diagonal_integrals_2 = jnp.einsum('gI, gII -> gI', g_vh[:,1:], psi_values_at_gauss_points[:,1:,1:-1]**2)

    # needs to be shape (n,)
    other_diagonal_entries = (
        gauss_quadrature_with_values(gauss_weights, fvalues=values_for_diagonal_integrals_1, length=hgrid) # has shape (n,)
        +
        gauss_quadrature_with_values(gauss_weights, fvalues=values_for_diagonal_integrals_2, length=hgrid) # has shape (n,)
        )

    diag = diag.at[0].set(first_diagonal_entry)
    diag = diag.at[-1].set(last_diagonal_entry)
    diag = diag.at[1:-1].set(other_diagonal_entries)

    values_for_offdiagonal_integrals = jnp.einsum('gI, gII, gII -> gI', g_vh, psi_values_at_gauss_points[:,:,:-1], psi_values_at_gauss_points[:,:,1:])
    offdiagonal_integrals = gauss_quadrature_with_values(gauss_weights, fvalues=values_for_offdiagonal_integrals, length=hgrid) # is shape (n+1,)

    RF = (
            jnp.diag(diag,k=0)
            + jnp.diag(offdiagonal_integrals,k=1)
            + jnp.diag(offdiagonal_integrals,k=-1)
        )

    return RF

def get_Rnu(
        n: int,
        hgrid: float,
        ) -> jnp.ndarray:
    """
    calculates R_nu matrix
    """

    diag = (2*jnp.ones((n+2,))).at[0].set(1).at[-1].set(1)
    offdiag = -jnp.ones((n+1,))

    Rnu = (
            jnp.diag(diag,k=0)
            + jnp.diag(offdiag,k=1)
            + jnp.diag(offdiag,k=-1)
        )

    return 1/hgrid * Rnu



def get_R(
        n: int,
        hgrid: float,
        wh: jnp.ndarray,
        constants: dict,
        psi_values_at_gauss_points: jnp.ndarray,
        g: callable,
        gauss_weights: jnp.ndarray,
        ) -> jnp.ndarray:

    # get constants
    l, gamma, nu = constants['l'], constants['gamma'], constants['nu']

    # get beta from wh
    beta = wh[n+1:]

    Z1 = jnp.zeros((n+1,n+1))
    Z2 = jnp.zeros((n+1,n+2))
    Z3 = Z2.T
    RF = get_RF(
        n=n,
        hgrid=hgrid,
        beta=beta,
        psi_values_at_gauss_points=psi_values_at_gauss_points,
        g=g,
        gauss_weights=gauss_weights,
        )
    Rnu = get_Rnu(n=n, hgrid=hgrid)

    R = jnp.block([[Z1, Z2], [Z3, gamma * RF + nu * Rnu]])

    return R


# ----------------------------------------
# ----------- for hamiltonian ------------
# ----------------------------------------
def get_eta(
        n: int,
        wh: jnp.ndarray,
        p: callable,
        ) -> jnp.ndarray:
    # p is the pressure law
    # if the hamiltonian is needed, p(rho) = rho + rho^3 needs to be used!

    # extract coefficient vector
    alpha = wh[:n+1]

    # since these are piecewise constant functions, the projection is trivial.
    e1 = p(alpha)

    return wh.at[:n+1].set(e1)

# this is not needed, as eta is the identity on the second component
# def get_eta_inv(
#         n: int,
#         e: jnp.ndarray,
#         pinv: callable,
#         ) -> jnp.ndarray:
#     # pinv is the inverse map of the pressure law
#     # if the hamiltonian is needed, p(rho) = rho + rho^3 needs to be used!
#
#     e1 = e[:n+1] # eta is the identity on the second component, only first component needs to be considered
#     pinv_e1 = pinv(e1)
#
#     return e.at[:n+1].set(pinv_e1)


def get_hamiltonian(
        n: int,
        hgrid: float,
        wh: jnp.ndarray,
        P: callable,
        ) -> jnp.ndarray:
    """
    computes the hamiltonian of the space discretized system, which reads as

    H_h(w_h) = H(z_h) = H(rho_h, v_h),

    where z_h = (rho_h, v_h) are the piecewise constant/linear interpolations of w_h = (alpha, beta) and H is the hamiltonian
    of the space-continuous system which reads as

    H(z) = int_{0}^{\ell} [ P(rho) + 1/2 * v**2 ] dx.
    """

    # extract state
    alpha = wh[:n+1]
    beta = wh[n+1:]

    # P(rho) term is easy since P(rho) is constant
    int_P = hgrid * jnp.sum(P(alpha))

    # get mass matrix M22 for v**2 term
    M22 = get_M22(n, hgrid)

    # compute integral over v**2 term
    int_v = 1/2 * beta.T @ M22 @ beta

    return int_P + int_v



# ----------------------------------------
# ------------- for B matrix -------------
# ----------------------------------------

def get_Btilde(n: int,
               constants: dict) -> jnp.ndarray:

    Btilde = jnp.zeros((n+2,2))

    Btilde = Btilde.at[0,0].set(1).at[n+1,1].set(-1)

    return Btilde


def get_B(n: int,
          constants: dict) -> jnp.ndarray:

    Z = jnp.zeros((n+1,2))
    Btilde = get_Btilde(n,constants)

    B = jnp.vstack((Z,Btilde))

    return B




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


def get_example_system(options: dict = None, nu: float = 0.0, friction_kind: str = 'regular') -> PortHamiltonian_LinearEJB:

    # default constants
    constants = {'l': 10.0, 'gamma': 0.1, 'nu': nu}
    inner_grid_points = 9
    num_gauss_nodes = 10

    # default pressure law p(rho)
    P = jax.jit( lambda rho: 1/2 * rho**2 + 1/4 * rho**4 )
    p = jax.jit( lambda rho: rho + rho**3 ) # satisfies P'(rho) = p(rho)

    # default friction law divided by v.
    # friction is assumed to decompose into f(v) = v * g(v)
    # g = lambda v: v * jnp.tanh(v)
    if friction_kind == 'regular':
        g = jax.jit( lambda v: (1+v**2)/(jnp.sqrt(1+v**2)) )
    else: # friction_kind == 'nonregular':
        g = jax.jit( lambda v: jnp.sign(v)*jnp.sqrt(jnp.abs(v)) / v )


    # retrieve from options
    if options is not None:
        constants = options.get('constants',constants)
        inner_grid_points = options.get('inner_grid_points',inner_grid_points)
        num_gauss_nodes = options.get('num_gauss_nodes',num_gauss_nodes)
        p = options.get('pressure_law',p)
        g = options.get('friction_law_divided_by_v',g)

    l = constants['l']
    xi = jnp.linspace(0,l,inner_grid_points+2)

    # prepare gauss quadrature
    gauss_points, gauss_weights = roots_legendre(num_gauss_nodes)
    gauss_points, gauss_weights = jnp.array(gauss_points), jnp.array(gauss_weights)

    # build cache for R22 and eta -- naive and really slow
    psi_values_at_gauss_points = jnp.zeros( (num_gauss_nodes, inner_grid_points+1, inner_grid_points+2) )

    onehat = get_psi_slim(j=2, info={'T': l, 'N': inner_grid_points+1}) # j = 1, ..., n+2

    increasing_front_at_gauss_nodes = onehat((xi[1]-xi[0])/2 * gauss_points + (xi[0]+xi[1])/2)
    decreasing_front_at_gauss_nodes = increasing_front_at_gauss_nodes[::-1]

    for I in range(inner_grid_points+1):
        # here, first index is the gauss point, second index is the interval, third index is the basis function
        psi_values_at_gauss_points = psi_values_at_gauss_points.at[:,I,I].set(decreasing_front_at_gauss_nodes)
        psi_values_at_gauss_points = psi_values_at_gauss_points.at[:,I,I+1].set(increasing_front_at_gauss_nodes)

    dim_sys = 2*inner_grid_points + 3
    hgrid = constants.get('l')/(inner_grid_points+1)

    E = get_E(inner_grid_points, hgrid)
    J = get_J(inner_grid_points)
    B = get_B(inner_grid_points, constants)

    def R(wh: jnp.ndarray):
        return get_R(
            inner_grid_points,
            hgrid,
            wh,
            constants,
            psi_values_at_gauss_points,
            g,
            gauss_weights,
        )
        # return 0*wh

    R_vmap = jax.vmap(
        R,
        in_axes=0, # 0 -> index where time array is
        out_axes=0, # 0 -> index where time array is
        )

    eta = lambda wh: \
        get_eta(
            inner_grid_points,
            wh,
            p,
            )

    eta_vmap = jax.vmap(
        eta,
        in_axes = 0, # 0 -> index where time array is
        out_axes = 0, # 0 -> index where time array is
        )

    hamiltonian = lambda wh: \
        get_hamiltonian(
            inner_grid_points,
            hgrid,
            wh,
            P,
            )

    hamiltonian_vmap = jax.vmap(
        hamiltonian,
        in_axes = 0, # 0 -> index where time array is
        out_axes = 0, # 0 -> index where time array is
        )

    # for eta_inv
    # -> taking eta_inv as the identity is sufficient since R is only nonlinear in the v and eta is the identity in that component
    eta_inv = lambda wh: wh

    info = {
        'type': 'damped wave',
        'dim_sys': dim_sys,
        'constants': constants,
        'inner_grid_points': inner_grid_points,
        'hgrid': hgrid,
        'pressure_law': p,
        }

    ph_sys = PortHamiltonian_LinearEJB(
        E=E,
        J=J,
        R=R_vmap,
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
        ph_sys: PortHamiltonian_LinearEJB
        ) -> (jnp.ndarray, callable):

    dim_sys = ph_sys.info.get('dim_sys')
    nsys = ph_sys.info.get('inner_grid_points')

    # initial condition
    z0 = jnp.zeros((dim_sys,))
    p_boundary = 1.0

    # initial condition for rho
    rho_0 = jnp.zeros((nsys+1,))
    for k in range(nsys+1):
        # for a nice sin curve
        value = 0.5*jnp.sin(k*jnp.pi/nsys) + p_boundary
        rho_0 = rho_0.at[k].set(value)
    # print(f'rho_0 = {rho_0}')
    z0 = z0.at[0:nsys+1].set(rho_0)

    # initial condition for velocity
    m_0 = jnp.zeros((nsys+2,))
    for k in range(nsys+2):
        # for a nice curve
        # value = jnp.log(5*k/nsys + 1) + 1
        value_in_m1_p1 = (2*k/(nsys+2) - 1) # result is in [-1,1]
        value = (2*value_in_m1_p1)**3
        m_0 = m_0.at[k].set(value)
    # print(f'm_0 = {m_0}')
    z0 = z0.at[nsys+1:dim_sys].set(m_0)

    # control values
    def control(t):
        # value = p_boundary # constant control
        # value = p_boundary - jnp.sin(t)/((t/2)**2+1) - 0.1*t # slowly decreasing sine wave
        value = p_boundary - jnp.sin(t) # sine wave
        return value*jnp.ones((2,))

    control = jax.vmap(control, in_axes=0, out_axes=0,) # 0 = index where time is

    return z0, control

def get_example_manufactured_solution(
        ph_sys: PortHamiltonian_LinearEJB,
        ) -> (jnp.ndarray, callable, callable, callable):

    inner_grid_points = ph_sys.info['inner_grid_points']
    l = ph_sys.info['constants']['l']
    nu = ph_sys.info['constants']['nu']
    p = ph_sys.info['pressure_law']

    # what we want for rho and v
    def rho(t,x):
        return jnp.sin(t) * jnp.sin(x)

    def v(t,x):
        return jnp.sin(t) * jnp.sin(x)

    # partial derivative of v with respect to x
    # dx_v = jax.jit(jax.jacobian(v, argnums=1))
    def dx_v(t,x):
        return jnp.sin(t) * jnp.cos(x)

    # control is just p( rho(t, 0) ) and p( rho(t, ell) )
    def control(t):
        u0 = p(rho(t,0.0)) - nu * dx_v(t,0.0)
        ul = p(rho(t,l)) - nu * dx_v(t,l)
        return jnp.vstack((u0, ul)).T

    # true solution evaluated at discretized space points
    xi = jnp.linspace(0,l,inner_grid_points+2)

    # only scalar t allowed
    def rho_discret(t):
        return rho(t, (xi[:-1] + xi[1:])/2)

    # only scalar t allowed
    def v_discret(t):
        return v(t, xi)

    # only scalar t allowed
    def z_discret(t):
        return jnp.hstack(
            ( rho_discret(t), v_discret(t) )
            )

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
    J = ph_sys.J # constant map, but still callable
    R = ph_sys.R # nonlinear
    eta = ph_sys.eta # nonlinear
    B = ph_sys.B_constant # constant

    def g(t):
        E_dtz = jnp.einsum('xn, tn -> tx', E, dt_z_discret(t))
        JmR_eta_z = jnp.einsum('txn, tn -> tx', J(z_discret(t)) - R(z_discret(t)), eta(z_discret(t)))
        B_u = jnp.einsum('nm, tm -> tn', B, control(t))

        return E_dtz - (JmR_eta_z + B_u)

    return z0, control, g, z_discret


if __name__ == '__main__':

    jax.config.update("jax_enable_x64", True)

    from helpers.other import mpl_settings
    mpl_settings()

    import matplotlib.pyplot as plt
    from timeit import default_timer as timer
    from spp import spp
    from damped_wave.visualization import plot_3d_wave


    # get example system
    ph_sys = get_example_system(
        nu = 1.0, # viscosity parameter
        options={'inner_grid_points': 25}, # 9 -> hgrid = 1
        friction_kind = 'nonregular',
        ) # nonlinearity_in is an optional argument
    igp = ph_sys.info['inner_grid_points']

    # get manufactured solution
    z0, control, g, true_solution = get_example_manufactured_solution(ph_sys)
    zero_control = jax.vmap(lambda t: jnp.zeros((igp+2,)), in_axes=0, out_axes=0)


    # # test hamiltonian and eta
    # hgrid = ph_sys.info['constants']['l']/(igp+1)
    # Hprime = ph_sys.hamiltonian_prime
    # # testvector = 3*jnp.linspace(1,5,igp+2).reshape((1,-1))
    # # testvector = 3*jnp.ones((igp+2,)).reshape((1,-1))
    # testvector = jnp.sin(jnp.linspace(0,2*jnp.pi,2*igp+3)).reshape((1,-1))
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

    # # solve
    T = 5
    nt_spp = 51
    tt_spp = jnp.linspace(0,T,nt_spp)
    # spp_degree = 4
    # num_proj_nodes = 2*spp_degree # correct choice for this system
    #
    # s_spp = timer()
    # spp_solution = spp(ph_sys=ph_sys, tt=tt_spp, z0=z0, u=control, degree=spp_degree, num_proj_nodes=num_proj_nodes,
    #                    g=g
    #                    )
    # e_spp = timer()
    # print(f'\nspp done (igp={igp}, nt_spp={nt_spp}, degree={spp_degree}), took {e_spp-s_spp:.2f} seconds')
    #
    # # plot
    # tt_fine, cc_fine = spp_solution['boundaries']
    # plot_3d_wave(
    #     tt = tt_fine,
    #     cc = cc_fine,
    #     ph_sys = ph_sys,
    #     title = f'\\noindent solution to the damped wave equation\\\\with $k={spp_degree}, ~s_{{\Pi}}={num_proj_nodes}$ and {friction_kind} friction',
    #     )

    # plot manufactured solution
    plot_3d_wave(
        tt = tt_spp,
        cc = true_solution(tt_spp),
        ph_sys = ph_sys,
        title = f'\\noindent manufactured solution to the damped wave equation',
        )

    # energybalance
    # from helpers.visualization import calculate_error_in_energybalance
    # error_in_energybalance = calculate_error_in_energybalance(spp_solution, ph_sys, u=control, relative=True,)
    #
    # plt.semilogy(
    #     tt_spp[1:], error_in_energybalance,
    #     label=f'$k = {spp_degree},~ s_{{\Pi}} = {num_proj_nodes}$',
    #     color=plt.cm.tab20(0)
    #     )
    # plt.legend()
    # plt.xlabel('time')
    # plt.ylabel('relative error in energy balance')
    # plt.ylim(1.5e-18, 1.5e-3)
    # plt.show()


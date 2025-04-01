#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this file implements the structure preserving projection scheme
# discussed in the paper for arbitrary polynomial degrees. see the
# accompanied pdf notes (spp-details.pdf) for details on the
# implementation.
#
#

import jax
# jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)
import jax.numpy as jnp

from scipy.special import roots_legendre

from helpers.newton import newton
from helpers.ph import PortHamiltonian_LinearE
from helpers.legendre import scaled_legendre, scaled_legendre_on_boundaries
from helpers.gauss import gauss_quadrature_with_values, project_with_gauss

from timeit import default_timer as timer


def spp(
        ph_sys: PortHamiltonian_LinearE,
        tt: jnp.ndarray,
        z0: jnp.ndarray,
        u: callable,
        degree: int,
        num_quad_nodes: int | None = None,
        num_proj_nodes: int | None = None,
        debug: bool = False,
        g: callable = None,
    ) -> jnp.array:
    """
    computes an approximate solution of

    E z' = (J(z) - R(z)) eta(z) + B(z) u + g,    z(0) = z_0

    on given timesteps in the time horizon [0,T].

    spp stands for "structure preserving projection"

    the method finds an approximate solution zh such that
    the function zh satisfies the projection equation.
    here, zh is a piecewise polynomial with a specified
    maximal degree.

    :param ph_sys: port hamiltonian system for which dynamics should be approximated
    :param tt: array of timepoints to be used
    :param z0: initial condition
    :param u: control function, callable
    :param degree: degree of piecewise polynomial approximation
    :param num_quad_nodes: number of quadrature nodes used for the quadrature rule for (J-R) eta + B u (optional, default = degree)
    :param num_proj_nodes: number of quadrature nodes used in the projection of eta (optional, default = degree)
    :param debug: debug flag (default False)
    :param g: optional function for using a manufactured solution
    :return: solution as a dictionary containing:
                - value of solution at time points [t_0, t_1, ... ]
                - value of solution at 25 points in each interval [t_0, t_1], [t_1, t_2], ...
                - value of solution at gauss points in each interval [t_0, t_1], [t_1, t_2], ...
                - array of coefficients for solution of shape (N, (n+1), D)
                    the coefficient for the k-th time interval [t_k, t_{k+1}] is
                    stored in coefflist[k,:,:].
                    evaluation of these coefficients is possible using, e.g.,
                        jnp.einsum('MD,Mt->tD',coefflist[k,:,:],scaled_legendre(n,wanted_points)[0])
                - the degree
                - the number of quadrature nodes for the quadrature rules
                - the number of quadrature nodes for the projection
    """

    n = degree

    # eta^inv is not needed if eta is linear or J and R are state-independent.
    E,J,R,eta,eta_inv,B = ph_sys.E_constant,\
                  ph_sys.J,         \
                  ph_sys.R,         \
                  ph_sys.eta,       \
                  ph_sys.eta_inv,   \
                  ph_sys.B

    D = ph_sys.info.get('dim_sys')
    M = tt.shape[0]-1 # number of timepoints

    if num_quad_nodes is None:
        num_quad_nodes = degree

    if num_proj_nodes is None:
        num_proj_nodes = degree

    if g is None:
        def g(t):
            return jnp.zeros((t.shape[0],D))


    # setup gauss quadrature for E term -> always n nodes
    n_gauss_points, n_gauss_weights = roots_legendre(n)
    n_gauss_points, n_gauss_weights = jnp.array(n_gauss_points), jnp.array(n_gauss_weights)

    # setup variable gauss quadrature for (J-R) term -> nqn nodes (nqn = `num quadrature nodes`)
    nqn_gauss_points, nqn_gauss_weights = roots_legendre(num_quad_nodes)
    nqn_gauss_points, nqn_gauss_weights = jnp.array(nqn_gauss_points), jnp.array(nqn_gauss_weights)

    # setup gauss quadrature for projection of eta -> proj_nodes nodes
    proj_gauss_points, proj_gauss_weights = roots_legendre(num_proj_nodes)
    proj_gauss_points, proj_gauss_weights = jnp.array(proj_gauss_points), jnp.array(proj_gauss_weights)

    # get legendre values on n_gauss_points (for integrals with E)
    pk_at_n_gauss_points, pkprime_at_n_gauss_points = scaled_legendre(n, n_gauss_points)
    phi_at_n_gauss_points = pk_at_n_gauss_points[:-1,:] # phi_at_gauss_points[i,j] = phi_i(tj)

    # get legendre values on nqn_gauss_points (for quadratures Q_i)
    pk_at_nqn_gauss_points, pkprime_at_nqn_gauss_points = scaled_legendre(n, nqn_gauss_points) # n in first argument is correct here
    phi_at_nqn_gauss_points = pk_at_nqn_gauss_points[:-1,:] # phi_at_gauss_points[i,j] = phi_i(tj)

    # get legendre values on proj_gauss_points (for scalar product in projection of eta)
    pk_at_proj_gauss_points, pkprime_at_proj_gauss_points = scaled_legendre(n, proj_gauss_points) # n in first argument is correct here

    # get scaled legendre values on -1 and 1
    minus1, plus1 = scaled_legendre_on_boundaries(n)

    def F(
        coeffs: jnp.ndarray,
        left_boundary_value: jnp.ndarray,
        k: int,
        ) -> jnp.ndarray:
        """
        this function calculates

        F(c0,...,cn)

        where zeros of F correspond to coefficients of solutions of the
        scheme for the k-th time interval.

        note: most shifts from [-1,1] to [tk,tkp1] are missing here,
        since they do not change the result of the computation. only
        in the calculation of the derivative partial_t z_h, a scaling
        factor is incorporated.

        :param coeffs: coefficient vectors c0,...,cn
        :param left_boundary_value: value zh(tk)
        :param k: timestep number
        :return: F(coeffs)
        """

        tk, tkp1 = tt[k], tt[k+1]

        # reshape coefficients for easier handling
        coeffs = coeffs.reshape((n+1,D))
        # coeffs[k,d] is the d-th entry of the coefficient belong to p_k

        # find E*zh_dot at n_gauss_points
        zh_dot = jnp.einsum('ND,Nt->tD', coeffs, pkprime_at_n_gauss_points) * 2/(tkp1-tk) # factor for compensating for chain rule in shift: f on [-1,1] -> sqrt(2/(b-a)) f((2t - a - b)/(b-a)) on [a,b]
        E_zh_dot_n_gauss = jnp.einsum('xD,tD->tx', E, zh_dot)

        # find zh at nqn_gauss_points and at proj_gauss_points
        # zh_nqn = jnp.einsum('ND,Nt->tD', coeffs, pk_at_nqn_gauss_points)
        zh_proj = jnp.einsum('ND,Nt->tD', coeffs, pk_at_proj_gauss_points)

        # find projection of eta at nqn_gauss_points
        eta_proj_nqn = project_with_gauss(proj_gauss_weights, pk_at_proj_gauss_points, eta(zh_proj), evaluate_at=nqn_gauss_points)

        eta_inv_eta_proj_nqn = eta_inv(eta_proj_nqn)

        # find values of (J(z) - R(z)) P_{n-1}[eta(z)] at nqn_gauss_points
        JmR_eta_nqn = jnp.einsum('txD,tD->tx', J(eta_inv_eta_proj_nqn) - R(eta_inv_eta_proj_nqn), eta_proj_nqn)

        # find values of B(z) u at nqn_gauss_points
        shifted_nqn_gauss_points = (tkp1-tk)/2 * nqn_gauss_points + (tk+tkp1)/2
        Bu_nqn = jnp.einsum('tDu,tu->tD', B(eta_inv_eta_proj_nqn), u(shifted_nqn_gauss_points))

        # for manufactured solution
        g_nqn = g(shifted_nqn_gauss_points)

        # piece them together
        JmR_nqn = JmR_eta_nqn + Bu_nqn + g_nqn # shape (t,D)

        # integrals with < E ... , phi >
        E_phi = jnp.einsum('tD,nt->tnD', E_zh_dot_n_gauss, phi_at_n_gauss_points)
        ints_with_E = gauss_quadrature_with_values(n_gauss_weights, E_phi, interval=(tk,tkp1)) # now shape (n,D)

        # integrals with < (J-R) ... , phi >
        JmR_phi = jnp.einsum('tD,nt->tnD', JmR_nqn, phi_at_nqn_gauss_points)
        ints_with_R = gauss_quadrature_with_values(nqn_gauss_weights, JmR_phi, interval=(tk,tkp1)) # now shape (n,D)

        # ints... contains the integral
        #   ints[k,d] = int < ... , phi_{kd} >
        # where phi_{kd} is phi_k in the d-th row

        # write to freturn array
        freturn = ints_with_E - ints_with_R # shape (n,D)

        # calculate g(coeffs)
        greturn = jnp.zeros((1,D))
        left_boundary_value_with_coeffs = jnp.einsum('ND,N->D', coeffs, minus1)
        greturn = greturn.at[0,:].set(left_boundary_value - left_boundary_value_with_coeffs)

        # stack and reshape, this is zero if the coefficients are correct
        return jnp.vstack((freturn, greturn)).reshape((-1,))


    coefflist = jnp.zeros((M,n+1,D))
    # coefflist = coefflist.at[0,:,:].set(jnp.ones((n+1,D)).at[1:,:].set(0)) # constant function as initial guess for first time interval

    DF = jax.jacobian(F, argnums = 0)
    rootfinder = newton(f=F, Df=DF, maxIter=20, tol=1e-17, debug=debug)

    # return array
    zz = jnp.zeros((M+1,D)).at[0,:].set(z0)

    nt_superfine = 25
    t_superfine = jnp.linspace(-1,1,nt_superfine) # in one interval
    zz_superfine = jnp.zeros((M*nt_superfine,D)) # will store z values for all intervals
    tt_superfine = jnp.zeros(M*nt_superfine) # will store t values for all intervals
    v_superfine, v_dsuperfine = scaled_legendre(n,t_superfine)

    # for values at gauss points
    # t_gauss = gauss_points
    zz_quad_nodes = jnp.zeros((M*num_quad_nodes,D))
    tt_quad_nodes = jnp.zeros((M*num_quad_nodes,))

    # initialize left_boundary_value
    left_boundary_value = z0

    # initialize coefficient guess
    coeff_guess = jnp.ones((n+1,D)).at[1:,:].set(0).reshape((-1,))
    # coeff_guess = jnp.vstack( ( jnp.sqrt(2)*z0.reshape(1,D), jnp.zeros((n,D)) ) ).reshape((-1,)) # constant polynomial with initial condition as first guess

    # print(f'setup complete, starting up now')

    # solve the ODE
    for k in range(M):

        # find root with newton
        root = rootfinder(coeff_guess, left_boundary_value=left_boundary_value, k=k)
        coeffs = root.reshape((n+1,D))
        # print(f'DF(root, k={k}) =\n{DF(root, left_boundary_value=left_boundary_value, k=k)}')
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # cax = ax.matshow(DF(root, left_boundary_value=left_boundary_value, k=k),
        #                  cmap='nipy_spectral')
        # fig.colorbar(cax)
        # plt.title(f'DF(root, k={k})')
        # plt.show()

        # for values at tt
        right_boundary_value = jnp.einsum('ND,N->D', coeffs, plus1)
        zz = zz.at[k+1,:].set(right_boundary_value)

        # update coefflist and coeff_guess
        coefflist = coefflist.at[k,:,:].set(coeffs)
        coeff_guess = coeffs.reshape((-1,))

        # for superfine points
        tk, tkp1 = tt[k], tt[k+1]
        zz_superfine_in_tk_tkp1 = jnp.einsum('ND,Nt->tD', coeffs, v_superfine)
        zz_superfine = zz_superfine.at[k*nt_superfine:(k+1)*nt_superfine,:].set(zz_superfine_in_tk_tkp1)
        tt_superfine = tt_superfine.at[k*nt_superfine:(k+1)*nt_superfine].set((tkp1-tk)/2 * t_superfine + (tk+tkp1)/2)

        # for gauss points
        zz_quad_nodes_in_tk_tkp1 = jnp.einsum('ND,Nt->tD', coeffs, pk_at_nqn_gauss_points)
        zz_quad_nodes = zz_quad_nodes.at[k*num_quad_nodes:(k+1)*num_quad_nodes,:].set(zz_quad_nodes_in_tk_tkp1)
        tt_quad_nodes = tt_quad_nodes.at[k*num_quad_nodes:(k+1)*num_quad_nodes].set((tkp1-tk)/2 * nqn_gauss_points + (tk+tkp1)/2)

        # update left boundary value for next iteration
        left_boundary_value = right_boundary_value

        # print(f'iteration k={k} done')

    return {
        'boundaries': (tt, zz), # values of solution at time interval boundaries
        'superfine': (tt_superfine, zz_superfine), # values at "superfine" sample points in between boundaries
        'gauss': (tt_quad_nodes, zz_quad_nodes), # values at gauss nodes
        'coefflist': coefflist, # shape (N,n+1,D)
        # #
        # 'ph_sys': ph_sys,
        'degree': degree,
        'num_quad_nodes': num_quad_nodes,
        'num_proj_nodes': num_proj_nodes,
        # #
        # 'control': u,
        # 'g': g,
        }


def eval_spp_solution(
        ph_sys: PortHamiltonian_LinearE,
        tt_ref: jnp.ndarray,
        spp_solution: dict,
        resample_step: float,
        ):
    """
    this method evaluates the spp solution on a given reference
    timeframe. the method assumes that

    tt_ref[::k] == tt_spp

    for some number k, here called "resample_step". another assumption
    is that both tt_ref and tt_spp are equally spaced.

    :param ph_sys: port hamiltonian system for which dynamics should be approximated
    :param tt_ref: reference array of timepoints to sample to
    :param spp_solution: output of spp method
    :param resample_step: number k such that tt_ref[::k] = tt_spp
    :return: values of spp solution at timepoints specified in tt_ref
    """

    dim_sys = ph_sys.info['dim_sys']
    tt_spp, zz_spp = spp_solution['boundaries']
    coefflist = spp_solution['coefflist']
    degree = spp_solution['degree']

    assert jnp.allclose(tt_ref[::resample_step]-tt_spp, 0)

    tt_j = tt_ref[:resample_step]
    t_j, t_jp1 = tt_spp[0], tt_spp[1]
    tt_j_shift = (2*tt_j - t_j - t_jp1)/(t_jp1 - t_j) # shift to [-1,1]

    # since tt_j_shift will be the same for every interval, we can precompute scaled_legendre once
    scaled_legendre_values, _ = scaled_legendre(degree, tt_j_shift)

    # calculate values in subintervals with einsum -> for every subinterval at once
    values_j_all = jnp.einsum('jMD,Mt->jtD', coefflist, scaled_legendre_values)

    # the values we need are just a reshape away
    values = values_j_all.reshape((-1,dim_sys))

    # the final value is missed, set it manually
    values = jnp.concatenate((values, zz_spp[-1,:][None,:]),axis=0)

    return values


if __name__ == '__main__':

    pass
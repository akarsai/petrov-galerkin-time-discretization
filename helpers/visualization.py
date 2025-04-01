#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this file implements visualizations for the power balance
#
#

import jax.numpy as jnp
import jax


from helpers.gauss import gauss_quadrature_with_values, project_with_gauss
from helpers.ph import PortHamiltonian_LinearE
from helpers.legendre import scaled_legendre
from scipy.special import roots_legendre


def calculate_error_in_energybalance(
        spp_solution: dict,
        ph_sys: PortHamiltonian_LinearE,
        u: callable,
        relative: bool = True,
        ):
    """
    """

    tt_spp = spp_solution['boundaries'][0]
    spp_degree = spp_solution['degree']
    num_quad_nodes = spp_solution['num_quad_nodes']
    num_proj_nodes = spp_solution['num_proj_nodes']

    R = ph_sys.R
    B = ph_sys.B
    eta = ph_sys.eta
    eta_inv = ph_sys.eta_inv
    hamiltonian = ph_sys.hamiltonian

    coefflist = spp_solution['coefflist']

    n = spp_degree
    M = coefflist.shape[0]

    # setup gauss quadrature
    # setup variable gauss quadrature for (J-R) term -> nqn nodes (nqn = `num quadrature nodes`)
    nqn_gauss_points, nqn_gauss_weights = roots_legendre(num_quad_nodes)
    nqn_gauss_points, nqn_gauss_weights = jnp.array(nqn_gauss_points), jnp.array(nqn_gauss_weights)

    # setup gauss quadrature for projection of eta -> proj_nodes nodes
    proj_gauss_points, proj_gauss_weights = roots_legendre(num_proj_nodes)
    proj_gauss_points, proj_gauss_weights = jnp.array(proj_gauss_points), jnp.array(proj_gauss_weights)

    # get legendre values on proj_gauss_points (for scalar product in projection of eta)
    pk_at_proj_gauss_points, pkprime_at_proj_gauss_points = scaled_legendre(n, proj_gauss_points) # n in first argument is correct here

    # prepare output arrays
    int_yu = jnp.zeros((M,))
    int_etaReta = jnp.zeros((M,))

    # loop over time steps
    def body(k, tup):

        int_yu, int_etaReta = tup

        tk, tkp1 = tt_spp[k], tt_spp[k+1]

        coeffs = coefflist[k,:,:]

        # find zh at nqn_gauss_points and at proj_gauss_points
        # zh_nqn = jnp.einsum('MD,Mt->tD', coeffs, pk_at_nqn_gauss_points)
        zh_proj = jnp.einsum('MD,Mt->tD', coeffs, pk_at_proj_gauss_points)

        # find projection of eta at nqn_gauss_points
        eta_proj_nqn = project_with_gauss(proj_gauss_weights, pk_at_proj_gauss_points, eta(zh_proj), evaluate_at=nqn_gauss_points)

        # jax.debug.print('eta_proj_nqn = {x}', x=eta_proj_nqn)

        # find eta_inv of projection of eta
        eta_inv_eta_proj_nqn = eta_inv(eta_proj_nqn)

        # find values of (J - R(z)) P_{n-1}[eta(z)] at variable gauss points
        eta_R_eta_nqn = jnp.einsum('tx,txD,tD->t', eta_proj_nqn, R(eta_inv_eta_proj_nqn), eta_proj_nqn)

        # find values of y(z)^T u = eta(z)^T B(z) u
        shifted_nqn_gauss_points = (tkp1-tk)/2 * nqn_gauss_points + (tk+tkp1)/2
        Bu_nqn = jnp.einsum('tDu,tu->tD', B(eta_inv_eta_proj_nqn), u(shifted_nqn_gauss_points))
        yu_nqn = jnp.einsum('tD,tD->t', eta_proj_nqn, Bu_nqn)

        # calculate integrals
        int_yu_k = gauss_quadrature_with_values(nqn_gauss_weights, yu_nqn, interval=(tk,tkp1))
        int_eta_R_eta_k = gauss_quadrature_with_values(nqn_gauss_weights, eta_R_eta_nqn, interval=(tk,tkp1))

        # store
        int_yu = int_yu.at[k].set(int_yu_k)
        int_etaReta = int_etaReta.at[k].set(int_eta_R_eta_k)

        return int_yu, int_etaReta

    int_yu, int_etaReta = jax.lax.fori_loop( 0, M, body, (int_yu, int_etaReta) ) # this is way faster than a python loop

    # calculate energy balance
    HH = hamiltonian(spp_solution['boundaries'][1])
    Hdiff = HH[1:] - HH[:-1]
    integral = int_yu - int_etaReta
    error_in_energybalance = jnp.abs(Hdiff - integral)

    if relative:
        error_in_energybalance = error_in_energybalance/ jnp.max(jnp.abs(Hdiff))

    return error_in_energybalance


if __name__ == '__main__':
    pass
#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this file implements gauss-quadratures used throughout this
# project
#
#


import jax
# jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)
import jax.numpy as jnp

from helpers.legendre import scaled_legendre, legendre

@jax.jit
def gauss_quadrature_with_values(
        gauss_weights: jnp.ndarray,
        fvalues: jnp.ndarray,
        interval: tuple | None = None,
        length: float | None = None,
        ) -> float | jnp.ndarray:
    """
    uses gauss quadrature to calculate the integral of
    f: R -> R^D (or R^{D,D} or something else)
    on the interval [t0,t1], where

    t0, t1 = interval

    the formula is

    int_t0^t1 f(x) dx ~=~ (t1-t0)/2 sum_i=1^n w_i f( (t1-t0)/2 x_i + (t0+t1)/2 )

    where w_i and x_i are the weights and points for the gauss quadrature
    on [-1,1]

    :param fvalues: values of function to project on the transformed gauss points (t1-t0)/2 x_i + (t0+t1)/2
    :param interval: interval to calculate integral on (optional)
    :param length: length of the interval (optional, needed if interval is not provided)
    :return: projection calculated with gauss quadrature
    """

    if interval is not None: # length is overwritten
        t0, t1 = interval
        length = (t1-t0)
    # else length has to be provided

    integral = length/2 * jnp.einsum('a,a...->...',gauss_weights,fvalues)

    return integral


def gauss_quadrature_4(
        f: callable,
        interval: tuple,
        ) -> float | jnp.ndarray:
    """
    uses gauss quadrature to calculate the integral of
    f: R -> R^D (or R^{D,D} or something else)
    on the interval [t0,t1], where

    t0, t1 = interval

    the formula is

    int_t0^t1 f(x) dx ~=~ (t1-t0)/2 sum_i=1^n w_i f( (t1-t0)/2 x_i + (t0 + t1)/2 )

    where w_i and x_i are the weights and points for the gauss quadrature
    on [-1,1]

    currently, only a 4 point quadrature is implemented

    :param f: function to project
    :param interval: time interval to project in
    :return: projection calculated with gauss quadrature
    """

    t0, t1 = interval
    # assert t0 < t1, 't0 must be smaller than t1'

    d = (t1 - t0)/2
    m = (t0 + t1)/2

    # points for 4 point gauss quadrature
    x = jnp.array([
        -jnp.sqrt(3/7 + 2/7*jnp.sqrt(6/5)),
        -jnp.sqrt(3/7 - 2/7*jnp.sqrt(6/5)),
         jnp.sqrt(3/7 - 2/7*jnp.sqrt(6/5)),
         jnp.sqrt(3/7 + 2/7*jnp.sqrt(6/5))
        ])

    # weights for 4 point guass quadrature
    w = jnp.array([
        (18-jnp.sqrt(30))/36,
        (18+jnp.sqrt(30))/36,
        (18+jnp.sqrt(30))/36,
        (18-jnp.sqrt(30))/36,
        ])

    fvals = f(d*x + m)
    integral = d*jnp.einsum('a,a...->...',w,fvals)

    # equivalent code, non vectorized:
    # apply quadrature rule
    # integral = 0
    # for i in range(4):
    #     xi, wi = x[i], w[i]
    #     integral += wi * f( d*xi + m )
    #
    # integral = d * integral

    return integral

def project_with_gauss(
        gauss_weights: jnp.ndarray,
        pk_values: jnp.ndarray,
        f_values: jnp.ndarray,
        evaluate_at: jnp.ndarray | None = None,
        only_coeffs: bool = False,
        ) -> jnp.ndarray:
    """
    this function computes the projection onto polynomials
    of a given function f, that is defined via function values in f_values

    the projection reads as

    f = sum_{k=0}^{n-1} < f, p_k >_{L^2} p_k

    where:
    - p_k is the k-th orthonormally scaled legendre polynomial
    - n is the number of supplied gauss weights
    - the scalar product in L^2 is computed using the n gauss weights

    note: this function does not really make sense on its own. it is primarily used
    to find the projection of eta in the file spp.py.

    :param gauss_weights: n gauss weights coming from quadrature rule
    :param pk_values: values of p_0, ... p_n at the n gauss points corresponding to the weights
    :param f_values: function values of the function to project at the n gauss points
    :param evaluate_at: points at which f_proj should be evaluated (optional)
    :param only_coeffs: boolean parameter to return only the calculated coefficients (optional)
    :return: value of the projected function at the n gauss points (or desired evaluation points)
    """

    ### note that pk_values is cut here! pk_values[:-1,:] is used, i.e. values of the highest order polynomial is cut away

    # calculate coefficients
    f_pk_values = jnp.einsum('t...,mt->tm...', f_values, pk_values[:-1,:]) # values of < f(t), p_k(t) > at gauss points for k = 0, ..., n-1
    f_coeffs = gauss_quadrature_with_values(gauss_weights, f_pk_values, interval=(-1,1)) # integrals int_{-1}^{1} < f(t), p_k(t) > dt, shape (m,D) (these are the coefficients in the legendre basis)

    # return only coefficients, if this is requested
    if only_coeffs:
        return f_coeffs

    # get function values
    if evaluate_at is None:
        f_proj = jnp.einsum('m...,mt->t...', f_coeffs, pk_values[:-1,:])  # P_{n-1}[f(t)] at gauss points
    else:
        n = pk_values.shape[0] - 1
        f_proj = jnp.einsum('m...,mt->t...', f_coeffs, scaled_legendre(n,evaluate_at)[0][:-1,:])  # P_{n-1}[f(t)] at desired points

    return f_proj
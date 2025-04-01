#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this file implements basis functions used for the semidiscretization
# of a porous medium equation. see porous-medium-details.pdf.
#
#


import jax.numpy as jnp
import jax
# jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt

from timeit import default_timer as timer


def get_phi(i: int, d: int, info: dict):
    """
    computes the (i,d)-th piecewise constant basis function
    as defined in the implementation details

    here, N is determined by the time discretization:
    0 = t_0 < t_1 < ... < t_N = T
    i.e., there are:
    N+1 total grid points and
    N-1 inner grid points

    :param i: index, ranging from 1 to N.
    :param d: dimension index, ranging from 1 to D=dim_sys.
    :param info: dict containing information such as N and T
    :return: function phi_i,d, that given an array
    tt of shape (nt,) returns an array of shape (nt,D)
    """

    N = info.get('N')
    T = info.get('T')
    D = info.get('D')
    tt = jnp.linspace(0,T,N+1)

    assert 1 <= i <= N
    assert 1 <= d <= D

    tt_im1, tt_i = tt[i-1], tt[i]

    def phi_i(x):
        return jnp.where(
            x < tt_im1,
            0.,
            jnp.where(
                x > tt_i,
                0.,
                1.
            )
        )

    column = jnp.eye(D)[:,d-1]

    # @jax.jit
    def phi_i_d(x):
        return jnp.einsum('a,b->ba', column, phi_i(x))

    # phi_i_d(jnp.array([0])) # trigger jit compilation

    return phi_i_d, phi_i


def get_Lambda(info: dict) -> (callable, callable):
    """
    computes the mother of all hatfunctions,
    called Lambda in the implementation details

    :param info: dict containing information such as N and T
    :return: function Lambda and its derivative
    """

    N = info.get('N')
    T = info.get('T')
    Delta_t = T/N

    @jax.jit
    def Lambda(z):

        return jnp.where(
                z < -Delta_t,
                0., # if z < -Delta_t return 0
                jnp.where(
                    z > Delta_t,
                    0., # if z > Delta_t return 0
                    jnp.where(
                        z < 0, # z in [-Delta_t,0]
                        1 + z/Delta_t, # true
                        1 - z/Delta_t, # false
                    )
                )
               )

        # readable version:
        # if z < -Delta_t or z > Delta_t:
        #     return 0.
        # elif z < 0: # z in [-Delta_t,0]
        #     return 1 + z/Delta_t
        # else: # z in [0,Delta_t]
        #     return 1 - z/Delta_t

    @jax.jit
    def Lambda_dot(z):

        return jnp.where(
                z < -Delta_t,
                0., # if z < -Delta_t return 0
                jnp.where(
                    z > Delta_t,
                    0., # if z > Delta_t return 0
                    jnp.where(
                        z < 0, # z in [-Delta_t,0]
                        1/Delta_t, # true
                        -1/Delta_t, # false
                    )
                )
               )

    return Lambda, Lambda_dot

def get_psi_slim(j: int, info: dict) -> callable:
    """
    computes the j-th piecewise linear basis function
    as defined in the implementation details

    here, N is determined by the time discretization:
    0 = t_0 < t_1 < ... < t_N = T
    i.e., there are:
    N+1 total grid points and
    N-1 inner grid points

    :param j: index, ranging from 1 to N+1.
    :param info: dict containing information such as N and T
    :return: function psi_j that, given an array
    tt of shape (nt,) returns an array of shape (nt,)
    """

    N = info.get('N')
    T = info.get('T')
    tt = jnp.linspace(0,T,N+1)

    # assert 1 <= j <= N+2
    # assert 1 <= d <= D

    tt_jm1 = tt[j-1]

    Lambda, _ = get_Lambda(info)

    @jax.jit
    def psi_j(x):
        return Lambda(x-tt_jm1)

    return psi_j


def get_psi(j: int, d: int, info: dict) -> callable:
    """
    computes the (i,d)-th piecewise linear basis function
    as defined in the implementation details

    here, N is determined by the time discretization:
    0 = t_0 < t_1 < ... < t_N = T
    i.e., there are:
    N+1 total grid points and
    N-1 inner grid points

    :param j: index, ranging from 1 to N+1.
    :param d: dimension index, ranging from 1 to D=dim_sys.
    :param info: dict containing information such as N and T
    :return: function psi_j,d that, given an array
    tt of shape (nt,) returns an array of shape (nt,D)
    alongside its derivative (psi_j,d)'
    """

    N = info.get('N')
    T = info.get('T')
    D = info.get('D')
    tt = jnp.linspace(0,T,N+1)

    # assert 1 <= j <= N+1
    # assert 1 <= d <= D

    tt_jm1 = tt[j-1]

    Lambda, Lambda_dot = get_Lambda(info)

    if j == 1:
        def psi_j(x):
            return jnp.where(
                x < 0,
                0.,
                Lambda(x)
            )
        def psi_j_dot(x):
            return jnp.where(
                x < 0,
                0.,
                Lambda_dot(x)
            )

    elif j == N+1:
        def psi_j(x):
            return jnp.where(
                x > T,
                0.,
                Lambda(x-T)
            )
        def psi_j_dot(x):
            return jnp.where(
                x > T,
                0.,
                Lambda_dot(x-T)
            )

    else: # j in {2,...,N}
        def psi_j(x):
            return Lambda(x-tt_jm1)
        def psi_j_dot(x):
            return Lambda_dot(x-tt_jm1)

    column = jnp.eye(D)[:,d-1]

    # @jax.jit
    def psi_j_d(x):
        return jnp.einsum('a,b->ba', column, psi_j(x))

    # @jax.jit
    def psi_j_d_dot(x):
        return jnp.einsum('a,b->ba', column, psi_j_dot(x))

    # trigger jit compilation
    # psi_j_d(jnp.array([0]))
    # psi_j_d_dot(jnp.array([0]))

    return psi_j_d, psi_j_d_dot, psi_j, psi_j_dot


def cache_basis_functions(info: dict) -> dict:
    """
    this method is used to cache all basis functions
    for piecewise constant, piecewise linear (+ derivatives)

    :param info: dict containing information such as N, T and D
    :return: dict containing all basis functions, possibly with projections
    """

    N, T, D, nt_cache = info['N'], info['T'], info['D'], info['nt_cache']
    tt_cache = jnp.linspace(0,T,nt_cache)

    phi_list = []
    psi_list = []
    psi_dot_list = []
    # psi_proj_list = []

    # piecewise constant
    for i in range(1,N+1):

        _, phi_i = get_phi(i, 1, info)
        # psi_j_proj = projection(psi_j, info)
        phi_i_tt = phi_i(tt_cache)
        local_list = []

        for d in range(1,D+1):
            column = jnp.eye(D)[:,d-1]
            local_list.append(jnp.einsum('a,b->ba', column, phi_i_tt))

        phi_list.append(local_list)

    # piecewise linear
    for j in range(1,N+2):

        _, _, psi_j, psi_j_dot = get_psi(j, 1, info)
        # psi_j_proj = projection(psi_j, info)
        psi_j_tt = psi_j(tt_cache)
        psi_j_dot_tt = psi_j_dot(tt_cache)
        local_list = []
        local_dot_list = []
        # local_proj_list = []

        for d in range(1,D+1):
            column = jnp.eye(D)[:,d-1]
            local_list.append(jnp.einsum('a,b->ba', column, psi_j_tt))
            local_dot_list.append(jnp.einsum('a,b->ba', column, psi_j_dot_tt))
            # local_proj_list.append(psi_j_proj * column)

        psi_list.append(local_list)
        psi_dot_list.append(local_dot_list)
        # psi_proj_list.append(local_proj_list)

    cache = {
        'phi': phi_list,
        'psi': psi_list,
        'psi_dot': psi_dot_list,
        # 'psi_proj': psi_proj_list,
        }

    return cache



def eval_function_from_coeffs(
        coeffs: list,
        tt: jnp.ndarray,
        info: dict,
        ) -> callable:
    """
    this method is used to evaluate a function u
    given by the linear combination of piecewise linear basis functions

    u(x) = sum_{j=1}^{N+1} psi_j(x) * coeffs[j-1]

    at the points provided in the array tt.

    :param coeffs: list of coefficient vectors.
        if the list is shorter than N+1, the remaining coefficients are interpreted as zero
    :param tt: array of shape (nt,) containing the points where to evaluate u
    :param info: dict containing information such as N, T and D
    :return: u(tt) as an array of shape (nt,D)
    """

    N, T = info['N'], info['T']
    nt = tt.shape[0]
    D = coeffs[0].shape[0]

    assert len(coeffs) <= N+1, f'number of coefficients must be smaller than N+1={N+1}. got {len(coeffs)}'

    xx = jnp.zeros((nt,D))

    psi_values = jax.jit(lambda j: get_psi_slim(j,info)(tt))

    for j,coeff in enumerate(coeffs):
        # assert len(coeff) == D, f'coefficient vectors must be of length D={D}'
        xx += jnp.einsum('a,b->ba',coeff,psi_values(j+1))

    return xx


def get_zh(
    alpha_k: float | jnp.ndarray,
    alpha_km1: float | jnp.ndarray,
    k: int,
    info: dict,
    ) -> callable:
    """
    for the time interval [t_{k-1}, t_k], this function
    returns a callable function

    zh(t) = alpha_{k-1} psi_{k-1}(t) + alpha_{k} psi_{k}(t)

    this is the expression of the approximate solution zh
    in that time interval, since all other basis functions
    from the sum vanish.

    :param alpha_k: coefficient
    :param alpha_km1: coefficient
    :param k: time interval index
    :param info: dictionary containing information about the discretization
    :return: callable function zh
    """

    psi_km1 = get_psi_slim(k, info)
    psi_k = get_psi_slim(k+1, info)

    def zh(t):

        psi_km1_t = psi_km1(t)
        psi_k_t = psi_k(t)

        if t.shape == ():
            return alpha_km1*psi_km1_t + alpha_k*psi_k_t
        else:
            return jnp.einsum('a,b->ba',alpha_km1,psi_km1_t) + jnp.einsum('a,b->ba',alpha_k,psi_k_t)

    return zh











if __name__ == '__main__':


    D = 5
    N = 10
    T = 1
    nt_cache = 1000
    info = {'N': N,
            'T': T,
            'D': D,
            'nt_cache': nt_cache}

    # for plotting
    tt_fine = jnp.linspace(0,T,nt_cache)
    tt_local = tt_fine[0:2*nt_cache//N]
    # print(tt_local)
    # tt_local = jnp.linspace(0,2*T/N,100)


    # cache basis functions
    s = timer()
    cache = cache_basis_functions(info)
    e = timer()
    print(f'caching the basis functions took {e-s:.2f} seconds')


    # coeffs = [jnp.array([1,2,3,4,5]), jnp.array([-1,-1,2.3,2,1]), jnp.array([4,-1,2.3,2,1])]
    coeffs = [jnp.array([-2,-2,-2,-2,-2]), jnp.array([2,2,2,2,2]), jnp.array([1,1,1,1,1])]
    uu = eval_function_from_coeffs(coeffs, tt_local, info, cache=cache)
    print(uu.shape)
    uudot = eval_function_from_coeffs(coeffs, tt_local, info, cache=cache, type='psi_dot')
    plt.plot(tt_local,uu[:,0],label='uu[:,0]')
    plt.plot(tt_local,uudot[:,0],label='uu dot[:,0]')
    plt.legend()
    plt.show()


    # ### piecewise constant functions
    # phi_1_1 = cache.get('phi')[0][0]
    # phi_N_2 = cache.get('phi')[-1][1]
    # phi_2_3 = cache.get('phi')[1][2]
    #
    # # plot
    # fig = plt.figure()
    # ax1 = fig.add_subplot(311)
    # ax2 = fig.add_subplot(312)
    # ax3 = fig.add_subplot(313)
    #
    # ax1.plot(tt_fine, phi_1_1(tt_fine), label=r'$\phi_{3,1}$')
    # ax2.plot(tt_fine, phi_N_2(tt_fine), label=r'$\phi_{N,2}$')
    # ax3.plot(tt_fine, phi_2_3(tt_fine), label=r'$\phi_{2,3}$')
    # ax1.legend()
    # ax2.legend()
    # ax3.legend()
    #
    # fig.suptitle(r'piecewise constant basis functions $\phi_{i,d}$')
    #
    # plt.show()

    ### piecewise linear functions
    # psi_1_1, psi_1_1_dot = cache.get('psi')[0][0],\
    #                        cache.get('psi_dot')[0][0]
    #
    # psi_Np1_2, psi_Np1_2_dot = cache.get('psi')[-1][1],\
    #                        cache.get('psi_dot')[-1][1]
    #
    # psi_2_3, psi_2_3_dot = cache.get('psi')[1][2],\
    #                        cache.get('psi_dot')[1][2]
    #
    # # print(psi_2_3_dot(jnp.array([0.,.3])))
    #
    # # plot
    # fig = plt.figure()
    # ax1 = fig.add_subplot(311)
    # ax2 = fig.add_subplot(312)
    # ax3 = fig.add_subplot(313)
    #
    # ax1.plot(tt_fine, psi_1_1, label=r'$\psi_{3,1}$')
    # ax2.plot(tt_fine, psi_Np1_2, label=r'$\psi_{N+1,2}$')
    # ax3.plot(tt_fine, psi_2_3, label=r'$\psi_{2,3}$')
    # ax1.plot(tt_fine, psi_1_1_dot, label=r'$\dot{\psi}_{3,1}$')
    # ax2.plot(tt_fine, psi_Np1_2_dot, label=r'$\dot{\psi}_{N+1,2}$')
    # ax3.plot(tt_fine, psi_2_3_dot, label=r'$\dot{\psi}_{2,3}$')
    # # ax1.legend()
    # # ax2.legend()
    # # ax3.legend()
    #
    # fig.suptitle(r'piecewise linear basis functions $\psi_{j,d}$')
    #
    # plt.show()






    pass
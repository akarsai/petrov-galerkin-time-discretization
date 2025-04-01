#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this file implements functions to obtain the matrices
# describing a toda lattice as considered in
#
# STRUCTURE-PRESERVING MODEL REDUCTION FOR NONLINEAR PORT-HAMILTONIAN SYSTEMS
# - S. CHATURANTABUT, C. BEATTIE, AND S. GUGERCIN
#
# important variables:
#       N : number of particles in toda lattice
#
# physical variables:
#       q : displacement of particle
#       p : momentum of particle
#
# physical constants:
#   gamma : friction term in lattice


import jax
# jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from helpers.ph import PortHamiltonian_LinearEJRB

def get_J(number_of_particles):

    N = number_of_particles

    # make matrices
    Z = jnp.zeros((N,N))
    I = jnp.eye(N)

    # make first row of J
    J1 = jnp.hstack((Z,I))

    # make second row of J
    J2 = jnp.hstack((-I,Z))

    # stack them
    J = jnp.vstack((J1,J2))

    # print(J)

    return J

def get_R(number_of_particles, constants):

    N = number_of_particles
    gamma = constants.get('gamma')

    # make matrices
    Z = jnp.zeros((N,N))
    Gamma = gamma * jnp.eye(N)

    # make first row of R
    R1 = jnp.hstack((Z,Z))

    # make second row of R
    R2 = jnp.hstack((Z,Gamma))

    # stack them
    R = jnp.vstack((R1,R2))

    return R

def get_B(number_of_particles):

    N = number_of_particles

    Z = jnp.zeros((N,1))
    e = Z.at[0,:].set(1)

    B = jnp.vstack((Z,e))

    return B

def hamiltonian(number_of_particles, state):

    N = number_of_particles

    # get q and p
    q = state[:N]
    p = state[N:]

    # calculate quadratic part of hamiltonian
    value = 1/2 * p.T @ p \
            + jnp.sum(jnp.exp(q[:-1] - q[1:])) \
            + jnp.exp(q[-1] - q[0]) - N

    return value

def get_example_system(options: dict = None) -> PortHamiltonian_LinearEJRB:

    # default constants
    constants = {'gamma': 0.1}
    number_of_particles = 5

    # retrieve from options
    if options is not None:
        constants = options.get('constants',constants)
        number_of_particles = options.get('number_of_particles',number_of_particles)

    dim_sys = 2*number_of_particles
    dim_input = 1

    E = jnp.eye(dim_sys)
    J = get_J(number_of_particles)
    B = get_B(number_of_particles)
    R = get_R(number_of_particles, constants)
    hamiltonian_fixed = lambda state: hamiltonian(number_of_particles, state)

    hamiltonian_vmap = jax.vmap(
        hamiltonian_fixed,
        in_axes=0, # 0 -> index where time array is
        out_axes=0, # 0 -> index where time array is
        )

    eta_vmap = jax.vmap(
        jax.grad(hamiltonian_fixed),
        in_axes=0, # 0 -> index where time array is
        out_axes=0, # 0 -> index where time array is
        )

    # here, eta_inv is not important, since J and R do not depend on the state.
    # therefore, J(eta^inv(eta(z))) = J anyways, so we can set eta_inv to the identity.
    eta_inv = lambda e: e

    info = {
        'type': 'toda lattice',
        'dim_sys': dim_sys,
        'dim_input': dim_input,
        'constants': constants,
        'number_of_particles': number_of_particles,
        }

    ph_sys = PortHamiltonian_LinearEJRB(E,J,R,eta_vmap,eta_inv,B,hamiltonian_vmap,info)

    return ph_sys

def get_example_initial_state_and_control(
        ph_sys: PortHamiltonian_LinearEJRB
        ) -> (jnp.ndarray, callable):

    dim_sys = ph_sys.info.get('dim_sys')
    dim_input = ph_sys.info.get('dim_input')
    z0 = jnp.zeros((dim_sys,))

    # control
    def control(t):
        return jnp.sin(2*t).reshape((-1,dim_input))
        # return 0.1 * jnp.ones(t.shape).reshape((-1,dim_input))

    return z0, control

def get_example_manufactured_solution(
        ph_sys: PortHamiltonian_LinearEJRB
        ) -> (jnp.ndarray, callable, callable, callable):

    _, control = get_example_initial_state_and_control(ph_sys)
    N = ph_sys.info['number_of_particles']

    # only scalar t allowed
    def q(t):
        return jnp.sin(t) * jnp.ones((N,))

    # only scalar t allowed
    def p(t):
        return jnp.cos(t) * jnp.ones((N,))

    # only scalar t allowed
    def z(t):
        return jnp.hstack(
            ( q(t), p(t) )
            )

    z0 = z(0.)

    # only scalar t allowed
    dt_z = jax.jacobian(z, argnums=0)

    # space discretized and vmapped in time
    z = jax.vmap(z, in_axes=0, out_axes=0,)
    dt_z = jax.vmap(dt_z, in_axes=0, out_axes=0,)

    # define rhs for manufactured solution
    E = ph_sys.E_constant # constant
    J = ph_sys.J_constant # constant
    R = ph_sys.R_constant # constant
    eta = ph_sys.eta # nonlinear
    B = ph_sys.B_constant # constant

    def g(t):
        E_dtz = jnp.einsum('xn, tn -> tx', E, dt_z(t))
        JmR_eta_z = jnp.einsum('xn, tn -> tx', J - R, eta(z(t)))
        B_u = jnp.einsum('nm, tm -> tn', B, control(t))

        return E_dtz - (JmR_eta_z + B_u)

    return z0, control, g, z



if __name__ == '__main__':

    from timeit import default_timer as timer
    import matplotlib.pyplot as plt

    from spp import spp
    from helpers.other import mpl_settings

    # apply matplotlib settings
    mpl_settings()

    # ph_sys = get_example_system(options={'number_of_particles': 3})
    ph_sys = get_example_system()
    z0, control = get_example_initial_state_and_control(ph_sys)
    dim_sys = ph_sys.info['dim_sys']

    J = ph_sys.J_constant # constant
    R = ph_sys.R_constant # constant
    B = ph_sys.B_constant # constant
    eta = ph_sys.eta
    H = ph_sys.hamiltonian

    T = 5

    # ### bdf4
    # nt_bdf = 100
    # tt_bdf = jnp.linspace(0,T,nt_bdf)
    # uu_bdf = control(tt_bdf)
    #
    # # bdf needs right hand side that takes z[:], not z[t_i,:]
    # def rhs_bdf(z,u):
    #     eta = ph_sys.eta # nonlinear
    #     B = ph_sys.B_constant # constant
    #
    #     z_reshaped = z.reshape((1,-1)) # puts z=[1,2,3] -> z=[[1,2,3]]
    #
    #     JmR_eta = jnp.einsum('mn,tn->tm',J-R,eta(z_reshaped)).reshape((-1,))
    #
    #     # rhs = B@u
    #     rhs = JmR_eta + B@u
    #     # jax.debug.print('rhs = {x}', x = rhs)
    #
    #     return rhs
    #
    # s_bdf = timer()
    #
    # zz_bdf = bdf4(
    #     rhs_bdf,
    #     tt_bdf,
    #     z0,
    #     uu_bdf
    #     )
    #
    # e_bdf = timer()
    # print(f'\nbdf done (dim_sys={dim_sys}, nt_bdf={nt_bdf}), took {e_bdf-s_bdf:.2f} seconds')

    ### spp
    nt_spp = 10
    spp_degree = 1
    num_quad_nodes = 100
    # num_proj_nodes = 2*spp_degree + 1 # 2*spp_degree + 1 gives best results?
    num_proj_nodes = 100 # 2*spp_degree + 1 gives best results?
    tt_spp = jnp.linspace(0,T,nt_spp)

    s_spp = timer()
    spp_solution = spp(ph_sys, tt=tt_spp, z0=z0, u=control, degree=spp_degree, num_quad_nodes=num_quad_nodes, num_proj_nodes=num_proj_nodes)
    e_spp = timer()

    print(f'\nspp done (dim_sys={dim_sys}, nt_spp={nt_spp}), took {e_spp-s_spp:.2f} seconds')

    tt_spp, zz_spp = spp_solution['boundaries']


    # discrete gradient method
    nt_dis = nt_spp
    tt_dis = jnp.linspace(0,T,nt_dis)

    s_dis = timer()
    zz_dis = discrete_gradient(ph_sys, tt=tt_dis, z0=z0, u=control,
                               # g=g
                               )
    e_dis = timer()

    print(f'\ndiscrete_gradient done (dim_sys={dim_sys}, nt_dis={nt_dis}), took {e_dis-s_dis:.2f} seconds')


    ### plot

    # yy_bdf = jnp.einsum('mn,tn->tm',B.T,eta(zz_bdf))
    # yy_spp = jnp.einsum('mn,tn->tm',B.T,eta(zz_spp))

    # index = int(dim_sys/2)
    index = 2

    plt.plot(tt_spp, zz_spp[:,index], label=rf'$z(t)_{index}$ (spp, degree {spp_degree}, {num_quad_nodes} quad nodes, {num_proj_nodes} proj nodes)')
    plt.plot(tt_dis, zz_dis[:,index], label=rf'$z(t)_{index}$ (discrete gradients)')

    plt.title(label=rf'toda lattice (dim_sys = {dim_sys})')

    plt.legend()
    plt.show()

    plt.plot(tt_spp, jnp.linalg.norm(zz_spp - zz_dis, axis=1), label=rf'$\|e(t)\|$ (spp minus discrete gradient)')

    plt.title(label=rf'toda lattice (dim_sys = {dim_sys}, spp degree {spp_degree}, {num_quad_nodes} quad nodes, {num_proj_nodes} proj nodes)')

    plt.legend()
    plt.show()


    plt.plot(tt_spp, H(zz_spp) - H(zz_dis), label=r'difference in hamiltonians (H(spp) - H(discrete gradient))')
    plt.legend()
    plt.show()

    max_diff = jnp.max(jnp.linalg.norm(zz_spp-zz_dis,axis=0))
    print(f'max_diff = {max_diff}')

    # plot_energybalance(
    #     spp_solution,
    #     u=control,
    #     ph_sys=ph_sys,
    #     title=f'projection method for toda lattice (dim_sys={dim_sys}, nt_spp = {nt_spp}, polynomial degree = {spp_degree})',
    #     kind='projected',
    #     show_energybalance=True,
    #     relative_energybalance=True,
    #     spp_degree=spp_degree,
    #     num_quad_nodes=num_quad_nodes,
    #     num_proj_nodes=num_proj_nodes,
    #     )

    # plot_error_in_energybalance_chain(
    #     spp_solution=spp_solution, # dictionary of tuples of tt, zz
    #     ph_sys=ph_sys, # only for mostly linear systems for now
    #     u=control,
    #     spp_degree=spp_degree,
    #     num_quad_nodes=num_quad_nodes, # quadrature nodes used for determining the error
    #     num_proj_nodes=num_proj_nodes,
    #     title = f'error in energy balance chain\n({nt_spp} timesteps, degree {spp_degree}, {num_proj_nodes} projection nodes, {num_quad_nodes} quadrature nodes)',
    #     savename = None,
    #     relative_error = True,
    #     show_plot = True,
    #     )
#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this file implements functions to obtain the matrices
# describing a spinning ridig body in 3 dimensions, taken from
#
# L^2-Gain and Passivity Techniques in Nonlinear Control
# - A. Van der Schaft
#
# physical variables:
#  p_x, p_y, p_z : momentum of body in space directions
#
# physical constants:
#  I_x, I_y, I_z : principle moments of inertia
#  b_x, b_y, b_z : torque axis for control input
#


import jax
# jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from helpers.ph import PortHamiltonian_LinearERQB

def get_J(state):

    p_x, p_y, p_z = state
    J = jnp.zeros((3,3))

    J = J.at[0,:].set([0, -p_z, p_y])
    J = J.at[1,:].set([p_z, 0, -p_x])
    J = J.at[2,:].set([-p_y, p_x, 0])

    return J

def get_B(b):

    B = jnp.zeros((3,1))
    B = B.at[:,0].set(b)

    return B

def get_Q(I):

    return jnp.diag(1/I)

def get_example_system(options: dict = None) -> PortHamiltonian_LinearERQB:

    # default constants
    constants = {
        'I': jnp.array([1, 1, 1]),
        'b': jnp.array([1, 1, 1]),
        }

    # retrieve from options
    if options is not None:
        constants = options.get('constants',constants)

    dim_sys = 3
    dim_input = 1

    E = jnp.eye(dim_sys)
    R = 0*E
    Q = get_Q(constants['I'])
    B = get_B(constants['b'])

    J = jax.vmap(
        get_J,
        in_axes=0, # 0 -> index where time array is
        out_axes=0, # 0 -> index where time array is
        )

    info = {
        'type': 'rigid_body',
        'dim_sys': dim_sys,
        'dim_input': dim_input,
        'constants': constants,
        }

    ph_sys = PortHamiltonian_LinearERQB(E,J,R,Q,B,info)

    return ph_sys

def get_example_initial_state_and_control(
        ph_sys: PortHamiltonian_LinearERQB
        ) -> (jnp.ndarray, callable):

    dim_sys = ph_sys.info.get('dim_sys')
    dim_input = ph_sys.info.get('dim_input')
    z0 = jnp.array([0,0.5,1])

    # control
    def control(t):
        return jnp.sin(2*t).reshape((-1,dim_input))
        # return 0.1 * jnp.ones(t.shape).reshape((-1,dim_input))

    return z0, control

def get_example_manufactured_solution(
        ph_sys: PortHamiltonian_LinearERQB
        ) -> (jnp.ndarray, callable, callable, callable):

    _, control = get_example_initial_state_and_control(ph_sys)

    # only scalar t allowed
    def z(t):
        return jnp.hstack(
            ( jnp.sin(t), jnp.sin(2*t)*jnp.cos(t)**2 + 0.5, jnp.cos(t) )
            )

    z0 = z(0.)

    # only scalar t allowed
    dt_z = jax.jacobian(z, argnums=0)

    # space discretized and vmapped in time
    z = jax.vmap(z, in_axes=0, out_axes=0,)
    dt_z = jax.vmap(dt_z, in_axes=0, out_axes=0,)

    # define rhs for manufactured solution
    E = ph_sys.E_constant # constant
    J = ph_sys.J # nonlinear
    R = ph_sys.R_constant # constant
    Q = ph_sys.Q_constant # constant
    B = ph_sys.B_constant # constant

    def g(t):
        E_dtz = jnp.einsum('xn, tn -> tx', E, dt_z(t))
        R_Q_z = jnp.einsum('xn, tn -> tx', R@Q, z(t))
        J_Q_z = jnp.einsum('txn, nm ,tm -> tx', J(z(t)), Q, z(t))
        B_u = jnp.einsum('nm, tm -> tn', B, control(t))

        return E_dtz - (J_Q_z - R_Q_z + B_u)

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
    # z0, control = get_example_initial_state_and_control(ph_sys)
    z0, control, g, _ = get_example_manufactured_solution(ph_sys)
    dim_sys = ph_sys.info['dim_sys']

    T = 5

    ### spp
    nt_spp = 100
    spp_degree = 4
    tt_spp = jnp.linspace(0,T,nt_spp)

    s_spp = timer()
    spp_solution = spp(ph_sys, tt=tt_spp, z0=z0, u=control, degree=spp_degree, g=g)
    e_spp = timer()

    print(f'\nspp done (dim_sys={dim_sys}, nt_bdf={nt_spp}), took {e_spp-s_spp:.2f} seconds')

    tt_spp, zz_spp = spp_solution['superfine']

    ### plot
    # index = int(dim_sys/2)

    plt.plot(tt_spp, zz_spp[:,0], label='$z(t)_{0}$')
    plt.plot(tt_spp, zz_spp[:,1], label='$z(t)_{1}$')
    plt.plot(tt_spp, zz_spp[:,2], label='$z(t)_{2}$')

    plt.title(label=f'spinning rigid body (spp solution, degree={spp_degree}, nt_spp={nt_spp})')

    plt.legend()
    plt.show()


    # # only for g = 0!
    # plot_energybalance(
    #     spp_solution,
    #     u=control,
    #     ph_sys=ph_sys,
    #     title=f'projection method for ridig body (dim_sys={dim_sys}, nt_spp = {nt_spp}, polynomial degree = {spp_degree})',
    #     kind='projected',
    #     show_energybalance=True,
    #     relative_energybalance=True,
    #     spp_degree=spp_degree,
    #     )









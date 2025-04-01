#
#                        author:
#                     attila karsai
#                 karsai@math.tu-berlin.de
#
# this file implements a class to store port-hamiltonian systems
#
#


import jax
# jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

class PortHamiltonian:
    """
    provides a class for fully nonlinear autonomous
    finite dimensional port-hamiltonian systems
    of the form

    E(z) z' = (J(z) - R(z)) eta(z) + B(z) u
         y  = B(z)^T eta(z)

    together with the hamiltonian H: Z -> R that satisfies

    d/dz H(z) = E(z)^T eta(z).
    """

    def __init__(self,
                 E: callable,
                 J: callable,
                 R: callable,
                 eta: callable,
                 eta_inv: callable, # inverse map of eta
                 B: callable,
                 hamiltonian: callable,
                 info: dict | None = None):
        """
        :param E: map E
        :param J: map J
        :param R: map R
        :param eta: map eta
        :param eta_inv: inverse of the map eta
        :param B: input map B
        :param hamiltonian: hamiltonian of the system
        :param info: information about the ph system
        """

        self.E = E
        self.J = J
        self.R = R
        self.eta = eta
        self.eta_inv = eta_inv
        self.B = B
        self.hamiltonian = hamiltonian
        self.info = info

class PortHamiltonian_LinearE(PortHamiltonian):
    """
    provides a class for autonomous finite dimensional
    port-hamiltonian systems of the form

    E z' = (J(z) - R(z)) eta(z) + B(z) u
      y  = B(z)^T eta(z).

    for these systems, the hamiltonian H satisfies

    d/dz H(z) = E^T eta(z)

    """

    def __init__(self,
                 E: jnp.ndarray,
                 J: callable,
                 R: callable,
                 eta: callable,
                 eta_inv: callable, # inverse map of eta
                 B: callable,
                 hamiltonian: callable,
                 info: dict | None = None):
        """
        :param E: matrix E
        :param J: matrix J
        :param R: matrix R
        :param eta: map eta
        :param eta_inv: inverse of the map eta
        :param B: matrix B
        :param hamiltonian: hamiltonian of the system
        :param info: information about the ph system
        """

        # create constant E map
        # -> can be evaluated for arrays of the form z[t_i,:]
        Evmap = jax.vmap(lambda z: E, in_axes=0, out_axes=0)

        self.E_constant = E

        # set system dimension
        if info is None:
            info = {'dim_sys': E.shape[1],
                    'dim_input': B(jnp.zeros((1,E.shape[1]))).shape[2], # try to probe by plugging zero in B
                    }

        # rest is handled by parent class
        super().__init__(Evmap, J, R, eta, eta_inv, B, hamiltonian, info)


class PortHamiltonian_LinearEJB(PortHamiltonian):
    """
    provides a class for autonomous finite dimensional
    port-hamiltonian systems of the form

    E z' = (J - R(z)) eta(z) + B u
      y  = B^T eta(z).

    for these systems, the hamiltonian H satisfies

    d/dz H(z) = E^T eta(z)

    """

    def __init__(self,
                 E: jnp.ndarray,
                 J: jnp.ndarray,
                 R: callable,
                 eta: callable,
                 eta_inv: callable, # inverse map of eta
                 B: jnp.ndarray,
                 hamiltonian: callable,
                 info: dict | None = None):
        """
        :param E: matrix E
        :param J: matrix J
        :param R: map R
        :param eta: map eta
        :param eta_inv: inverse of the map eta
        :param B: matrix B
        :param hamiltonian: hamiltonian of the system
        :param info: information about the ph system
        """

        # create constant vmaps
        # -> can be evaluated for arrays of the form z[t_i,:]
        Evmap = jax.vmap(lambda z: E, in_axes=0, out_axes=0)
        Jvmap = jax.vmap(lambda z: J, in_axes=0, out_axes=0)
        Bvmap = jax.vmap(lambda z: B, in_axes=0, out_axes=0)

        self.E_constant = E
        self.J_constant = J
        self.B_constant = B

        # set system dimension
        if info is None:
            info = {'dim_sys': E.shape[1],
                    'dim_input': B.shape[1],
                    }

        # rest is handled by parent class
        super().__init__(Evmap, Jvmap, R, eta, eta_inv, Bvmap, hamiltonian, info)

class PortHamiltonian_LinearEJRB(PortHamiltonian):
    """
    provides a class for autonomous finite dimensional
    port-hamiltonian systems of the form

    E z' = (J - R) eta(z) + B u
      y  = B^T eta(z).

    for these systems, the hamiltonian H satisfies

    d/dz H(z) = E^T eta(z)

    """

    def __init__(self,
                 E: jnp.ndarray,
                 J: jnp.ndarray,
                 R: jnp.ndarray,
                 eta: callable,
                 eta_inv: callable, # inverse map of eta
                 B: jnp.ndarray,
                 hamiltonian: callable,
                 info: dict | None = None):
        """
        :param E: matrix E
        :param J: matrix J
        :param R: matrix R
        :param eta: map eta
        :param eta_inv: inverse of the map eta
        :param B: matrix B
        :param hamiltonian: hamiltonian of the system
        :param info: information about the ph system
        """

        # store constant matrices
        self.E_constant = E
        self.J_constant = J
        self.R_constant = R
        self.B_constant = B

        # create constant E map
        # -> can be evaluated for arrays of the form z[t_i,:]
        Evmap = jax.vmap(lambda z: E, in_axes=0, out_axes=0)

        # create constant J map
        # -> can be evaluated for arrays of the form z[t_i,:]
        Jvmap = jax.vmap(lambda z: J, in_axes=0, out_axes=0)

        # create constant R map
        # -> can be evaluated for arrays of the form z[t_i,:]
        Rvmap = jax.vmap(lambda z: R, in_axes=0, out_axes=0)

        # create constant B map
        # -> can be evaluated for arrays of the form z[t_i,:]
        Bvmap = jax.vmap(lambda z: B, in_axes=0, out_axes=0)

        # set system info
        if info is None:
            info = {
                'dim_sys': J.shape[0],
                'dim_input': B.shape[1],
                }

        # rest is handled by parent class
        super().__init__(Evmap, Jvmap, Rvmap, eta, eta_inv, Bvmap, hamiltonian, info)

class PortHamiltonian_LinearEJQB(PortHamiltonian):
    """
    provides a class for autonomous finite dimensional
    port-hamiltonian systems of the form

    E z' = (J - R(z)) Qz + B u
      y  = B^T Q z.

    for these systems, the hamiltonian reads as

    H(z) = 1/2 z^T E^T Q z

    """

    def __init__(self,
                 E: jnp.ndarray,
                 J: jnp.ndarray,
                 R: callable,
                 Q: jnp.ndarray,
                 B: jnp.ndarray,
                 info: dict | None = None):
        """
        :param E: matrix E
        :param J: matrix J
        :param R: map R
        :param Q: matrix Q
        :param B: matrix B
        :param info: information about the ph system
        """

        self.E_constant = E
        self.J_constant = J
        self.Q_constant = Q
        self.B_constant = B

        # create hamiltonian map
        # -> can be evaluated for arrays of the form z[t_i,:]
        hamiltonian = jax.vmap(lambda z: 1/2 * z.T @ E.T @ Q @ z, in_axes=0, out_axes=0)

        # create eta map
        # -> can be evaluated for arrays of the form z[t_i,:]
        eta = jax.vmap(lambda z: Q @ z, in_axes=0, out_axes=0)

        # create eta_inv map
        # -> can be evaluated for arrays of the form z[t_i,:]
        Qinv = jnp.linalg.inv(Q)
        eta_inv = jax.vmap(lambda z: Qinv @ z, in_axes=0, out_axes=0)

        # create constant E map
        # -> can be evaluated for arrays of the form z[t_i,:]
        Evmap = jax.vmap(lambda z: E, in_axes=0, out_axes=0)

        # create constant J map
        # -> can be evaluated for arrays of the form z[t_i,:]
        Jvmap = jax.vmap(lambda z: J, in_axes=0, out_axes=0)

        # create constant B map
        # -> can be evaluated for arrays of the form z[t_i,:]
        Bvmap = jax.vmap(lambda z: B, in_axes=0, out_axes=0)

        # set system info
        if info is None:
            info = {
                'dim_sys': J.shape[0],
                'dim_input': B.shape[1],
                }

        # rest is handled by parent class
        super().__init__(Evmap, Jvmap, R, eta, eta_inv, Bvmap, hamiltonian, info)

class PortHamiltonian_LinearEQB(PortHamiltonian):
    """
    provides a class for autonomous finite dimensional
    port-hamiltonian systems of the form

    E z' = (J(z) - R(z)) Qz + B u
      y  = B^T Q z.

    for these systems, the hamiltonian reads as

    H(z) = 1/2 z^T E^T Q z

    """

    def __init__(self,
                 E: jnp.ndarray,
                 J: callable,
                 R: callable,
                 Q: jnp.ndarray,
                 B: jnp.ndarray,
                 info: dict | None = None):
        """
        :param E: matrix E
        :param J: map J
        :param R: map R
        :param Q: matrix Q
        :param B: matrix B
        :param info: information about the ph system
        """

        self.E_constant = E
        self.Q_constant = Q
        self.B_constant = B

        # create hamiltonian map
        # -> can be evaluated for arrays of the form z[t_i,:]
        hamiltonian = jax.vmap(lambda z: 1/2 * z.T @ E.T @ Q @ z, in_axes=0, out_axes=0)

        # create eta map
        # -> can be evaluated for arrays of the form z[t_i,:]
        eta = jax.vmap(lambda z: Q @ z, in_axes=0, out_axes=0)

        # create eta_inv map
        # -> can be evaluated for arrays of the form z[t_i,:]
        Qinv = jnp.linalg.inv(Q)
        eta_inv = jax.vmap(lambda z: Qinv @ z, in_axes=0, out_axes=0)

        # create constant E map
        # -> can be evaluated for arrays of the form z[t_i,:]
        Evmap = jax.vmap(lambda z: E, in_axes=0, out_axes=0)

        # create constant B map
        # -> can be evaluated for arrays of the form z[t_i,:]
        Bvmap = jax.vmap(lambda z: B, in_axes=0, out_axes=0)

        # set system info
        if info is None:
            info = {
                'dim_sys': E.shape[0],
                'dim_input': B.shape[1],
                }

        # rest is handled by parent class
        super().__init__(Evmap, J, R, eta, eta_inv, Bvmap, hamiltonian, info)

class PortHamiltonian_LinearERQB(PortHamiltonian):
    """
    provides a class for autonomous finite dimensional
    port-hamiltonian systems of the form

    E z' = (J(z) - R) Qz + B u
      y  = B^T Q z.

    for these systems, the hamiltonian reads as

    H(z) = 1/2 z^T E^T Q z

    """

    def __init__(self,
                 E: jnp.ndarray,
                 J: callable,
                 R: jnp.ndarray,
                 Q: jnp.ndarray,
                 B: jnp.ndarray,
                 info: dict | None = None):
        """
        :param E: matrix E
        :param J: map J
        :param R: matrix R
        :param Q: matrix Q
        :param B: matrix B
        :param info: information about the ph system
        """

        self.E_constant = E
        self.R_constant = R
        self.Q_constant = Q
        self.B_constant = B

        # create hamiltonian map
        # -> can be evaluated for arrays of the form z[t_i,:]
        hamiltonian = jax.vmap(lambda z: 1/2 * z.T @ E.T @ Q @ z, in_axes=0, out_axes=0)

        # create eta map
        # -> can be evaluated for arrays of the form z[t_i,:]
        eta = jax.vmap(lambda z: Q @ z, in_axes=0, out_axes=0)

        # create eta_inv map
        # -> can be evaluated for arrays of the form z[t_i,:]
        Qinv = jnp.linalg.inv(Q)
        eta_inv = jax.vmap(lambda z: Qinv @ z, in_axes=0, out_axes=0)

        # create constant E map
        # -> can be evaluated for arrays of the form z[t_i,:]
        Evmap = jax.vmap(lambda z: E, in_axes=0, out_axes=0)

        # create constant R map
        # -> can be evaluated for arrays of the form z[t_i,:]
        Rvmap = jax.vmap(lambda z: R, in_axes=0, out_axes=0)

        # create constant B map
        # -> can be evaluated for arrays of the form z[t_i,:]
        Bvmap = jax.vmap(lambda z: B, in_axes=0, out_axes=0)

        # set system info
        if info is None:
            info = {
                'dim_sys': E.shape[0],
                'dim_input': B.shape[1],
                }

        # rest is handled by parent class
        super().__init__(Evmap, J, Rvmap, eta, eta_inv, Bvmap, hamiltonian, info)

class PortHamiltonian_AllLinear(PortHamiltonian):
    """
    provides a class for autonomous finite dimensional
    port-hamiltonian systems of the form

    E z' = (J - R) Qz + B u
      y  = B^T Q z.

    for these systems, the hamiltonian reads as

    H(z) = 1/2 z^T E^T Q z.
    """

    def __init__(self,
                 E: jnp.ndarray,
                 J: jnp.ndarray,
                 R: jnp.ndarray,
                 Q: jnp.ndarray,
                 B: jnp.ndarray,
                 info: dict | None = None):
        """
        :param E: matrix E
        :param J: matrix J
        :param R: matrix R
        :param Q: matrix Q
        :param B: matrix B
        :param info: information about the ph system
        """

        # store constant matrices
        self.E_constant = E
        self.J_constant = J
        self.R_constant = R
        self.Q_constant = Q
        self.B_constant = B

        # create hamiltonian map
        # -> can be evaluated for arrays of the form z[t_i,:]
        hamiltonian = jax.vmap(lambda z: 1/2 * z.T @ E.T @ Q @ z, in_axes=0, out_axes=0)

        # create eta map
        # -> can be evaluated for arrays of the form z[t_i,:]
        eta = jax.vmap(lambda z: Q @ z, in_axes=0, out_axes=0)

        # create eta_inv map
        # -> can be evaluated for arrays of the form z[t_i,:]
        Qinv = jnp.linalg.inv(Q)
        eta_inv = jax.vmap(lambda z: Qinv @ z, in_axes=0, out_axes=0)

        # create constant E map
        # -> can be evaluated for arrays of the form z[t_i,:]
        Evmap = jax.vmap(lambda z: E, in_axes=0, out_axes=0)

        # create constant J map
        # -> can be evaluated for arrays of the form z[t_i,:]
        Jvmap = jax.vmap(lambda z: J, in_axes=0, out_axes=0)

        # create constant R map
        # -> can be evaluated for arrays of the form z[t_i,:]
        Rvmap = jax.vmap(lambda z: R, in_axes=0, out_axes=0)

        # create constant B map
        # -> can be evaluated for arrays of the form z[t_i,:]
        Bvmap = jax.vmap(lambda z: B, in_axes=0, out_axes=0)


        # set system info
        if info is None:
            info = {
                'dim_sys': J.shape[0],
                'dim_input': B.shape[1],
                }

        # rest is handled by parent class
        super().__init__(Evmap, Jvmap, Rvmap, eta, eta_inv, Bvmap, hamiltonian, info)




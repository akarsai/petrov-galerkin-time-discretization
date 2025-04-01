#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this file implements a fast newton method using jax.
#
#


# jax
import jax.numpy as jnp
from jax import jit, jacobian
import jax.lax

@jax.profiler.annotate_function
def newton( f , Df = None, maxIter = 10, tol = 1e-14, debug=False):
    """
    calculates the derivative of f with jax.jacobian
    and returns a jitted newton solver for solving

        f(x) = 0.

    this newton solver can then be called with arbitrary
    initial guesses (and other arguments passed to f)

    the method assumes that the argument of f for which
    we want to find the root is the first one. in other
    words, if

        f = f(a,b,c)

    then for fixed b and c, the method finds a such that

        f(a,b,c) = 0.

    :param f: function to find root of (w.r.t. first argument)
    :param Df: jacobian of f (w.r.t. first argument)
    :param maxIter: maximum number of iterations for newton solver
    :param tol: tolerance for newton solver
    :param debug: debug flag
    :return: callable newton solver
    """

    if Df is None:
        Df =  jacobian( f, argnums = 0 )

    @jax.profiler.annotate_function
    @jit
    def solver( x0, * args, ** kwargs  ):
        """
        this function is a newton solver and can be used
        to find x such that

        f(x, *args, **kwargs) = 0.

        :param x0: initial guess for x0
        :param args: arguments passed to f
        :param kwargs: keyword arguments passed to f
        :return: approximate solution x
        """

        # with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):

        @jax.profiler.annotate_function
        def body( tup ):
            i, x = tup
            update = jnp.linalg.solve( Df(x, * args, ** kwargs ), f(x, * args, ** kwargs ) )
            return  i+1, x - update

        def cond( tup ):
            i, x = tup

            # return jnp.less( i, maxIter )  # only check for maxIter

            return  jnp.logical_and( # check maxIter and tol
                jnp.less( i , maxIter ),  # i < maxIter
                jnp.greater( jnp.linalg.norm( f(x, * args, ** kwargs ) ), tol )  # norm( f(x) ) > tol
            )

        i, x = jax.lax.while_loop(cond, body, (0, x0) )

        if debug:
            jax.debug.print( '||f(x)|| = {x}, cond Df = {y}', x = jnp.linalg.norm(f(x, * args, ** kwargs )), y = jnp.linalg.cond(Df(x, * args, ** kwargs )))
            # jax.debug.print( 'iter = {x}', x = i)

        return x

    return solver



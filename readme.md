# Energy-consistent Petrov–Galerkin time discretization of port-Hamiltonian systems

This repository contains the code to the paper

[J. Giesselmann, A. Karsai, T. Tscherpel, Energy-consistent Petrov-Galerkin time discretization of port-Hamiltonian systems](https://doi.org/10.5802/smai-jcm.127)

## Reproducing our results

### Installing Python
**Note:** This step can be skipped if you already have a Python installation.

The first step is to install Python with version `>=3.13.0`.
We recommend using a virtual environment for this.
Using [pyenv](https://github.com/pyenv/pyenv) with [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv), the steps are as follows:

<details>
<summary><b>How to install pyenv and pyenv-virtualenv</b></summary>
<br>

```bash
## install pyenv
# automatic installer
curl -fsSL https://pyenv.run | bash
# or macos or linux with homebrew:
#     brew install pyenv
# now make pyenv available in the shell (this assumes you use zsh. if you use another shell, please consult the pyenv manual)
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init --path)"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
# restart the shell
exec "$SHELL"

## install pyenv-virtualenv
# download and install
git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv
# or macos or linux with homebrew:
#     brew install pyenv-virtualenv
# add to path
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.zshrc
# restart the shell
exec "$SHELL"
```
</details>

```bash
# this assumes pyenv is available in the environment
pyenv install --list | grep " 3\.[1]" # get all available versions, only starting with 3.1x
pyenv install 3.13.0 # choose 3.13.0 for example
pyenv virtualenv 3.13.0 petrov-galerkin # creates environment 'petrov-galerkin' with version 3.13.0
pyenv activate petrov-galerkin # activate virtual environment
```

### Clone repository, install requirements, run script
The next step is to clone this repository, install the necessary requirements located in `requirements.txt`, and set the `PYTHONPATH` variable accordingly.

If the code does not run, this could be due to recent changes in JAX, scipy or matplotlib. In this case, try changing `>=` in `requirements.txt` to `==`.

```bash
cd ~ # switch to home directory
git clone https://github.com/akarsai/petrov-galerkin-time-discretization.git
cd petrov-galerkin-time-discretization
pip install --upgrade pip # update pip
pip install -r requirements.txt # install requirements
export PYTHONPATH="${PYTHONPATH}:~/petrov-galerkin-time-discretization" # add folder to pythonpath
```

Now, we can run the script [`plots/all_plots.py`](plots/all_plots.py) to reproduce the figures in the paper.
The generated plots will be put in the directory [`results/figures`](results/figures) as `.pgf` and `.png` files.
Furthermore, `.pickle` files of the computed solutions will be created and put in the [`results`](results) folder.
```bash
# this command can take a really long time, since the PDE discretizations are high dimensional
python plots/all_plots.py
```
If you are only interested in parts of the results, you can change lines `1287` to `1292` in [`plots/all_plots.py`](plots/all_plots.py) accordingly.


## Structure of the repository

- The parent directory contains various time discretizations methods. here, [`spp.py`](spp.py) is the proposed approach (`spp` = **s**tructure **p**reserving **p**rojection).

- The folder [`helpers`](/helpers) contains helper functions used throughout the project. These include:
  - A newton method.
  - Methods used for quadrature.
  - Methods for the legendre polynomials.
  - A visualization function to create energy balance plots.
  - A class for port-hamiltonian systems in the formulation $$E(z) \dot{z}(t) = (J(z) - R(z)) \eta(z) + B(z) u$$ with $J(z)$ skew-symmetric, $R(z)$ symmetric positive semi-definite and a hamiltonian $\mathcal{H}$ that satisfies $\frac{d}{dt} \mathcal{H}(z) = E(z)^* \eta(z)$. 
  This formulation is equivalent to the formulation using $j, r$ and $b$ in the paper as long as $\eta$ is a bijection. 
  Thus, the code assumes that this is the case, and the user needs to provide a mapping $\eta^{-1}$ for systems with nonlinearity in the hamiltonian. 
  Note that although this map needs to be supplied, in the case that $J$ and $R$ are not state dependent, it can be set arbitrarily since the implementation essentially calls $R(\eta^{-1}(\Pi_{n-1}[\eta(z)]))$.

- The file in [`plots/all_plots.py`](/plots/all_plots.py) contains the code that is used to produce all figures in the paper. 
The script creates pickle files to save the computation results. 
These files are stored in a directory [`results`](/results).

- The files in [`damped_wave`](/damped_wave) contain the implementation of the quasilinear wave equation.

- The files in [`porous_medium`](/porous_medium) contain the implementation of the porous medium equation.

- The files in [`toda_lattice`](/toda_lattice) contain the implementation of a Toda lattice as considered in [[1]](https://epubs.siam.org/doi/10.1137/15M1055085).

- The files in [`ridig_body`](/rigid_body) contain the implementation the spinning rigid body example from [[2, Example 6.2.1]](https://doi.org/10.1007/978-3-319-49992-5).


[[1] S. Gugercin, R.V. Polyuga, C. Beattie, A. van der Schaft - Structure-Preserving Model Reduction for Nonlinear Port-Hamiltonian Systems](https://epubs.siam.org/doi/10.1137/15M1055085)

[[2] A. Van der Schaft, L2-Gain and Passivity Techniques in Nonlinear Control, vol. 2, Springer International Publishing, 2017.](https://doi.org/10.1007/978-3-319-49992-5)



## Example usage of the `spp` method

```python
# imports
import jax
jax.config.update("jax_enable_x64", True) # set jax to double precision
import jax.numpy as jnp
import matplotlib.pyplot as plt
from timeit import default_timer as timer

from spp import spp

from helpers.ph import PortHamiltonian_AllLinear

# set up port-hamiltonian system
D = 3 # system dimension

E = jnp.eye(D)
J = jnp.zeros((D,D))
R = jnp.eye(D)
Q = jnp.eye(D) # eta(z) = Qz 
B = jnp.zeros((D,1))

ph_sys = PortHamiltonian_AllLinear(E,J,R,Q,B)

# initial condition
z0 = jnp.ones((D,)) # z(0) = [1 ... 1]

# time horizon
T = 10
nt = 101    # number of time steps
nt_mid = nt # implicit midpoint method
nt_spp = nt # our method

tt_spp = jnp.linspace(0,T,nt_spp)
tt_mid = jnp.linspace(0,T,nt_mid)

# control values
def control(t):
    return jnp.ones((1,)) # constant control

control = jax.vmap(control, in_axes=0, out_axes=0,) # 0 = index where time is

uu_mid = control(tt_mid)

### implicit midpoint method
E = ph_sys.E_constant # constant
J = ph_sys.J_constant # constant
R = ph_sys.R_constant # constant
Q = ph_sys.Q_constant # constant
B = ph_sys.B_constant # constant

# implicit midpoint needs right hand side that takes x[:], not x[t_i,:]
def rhs(z,u):
    return jnp.linalg.solve(E, (J-R)@Q@z + B@u)

s_mid = timer()

zz_mid = implicit_midpoint(
    rhs,
    tt_mid,
    z0,
    uu_mid
    )

e_mid = timer()
print(f'\nimplicit midpoint done (dim_sys={D}, nt_mid={nt_mid}), took {e_mid-s_mid:.2f} seconds')

plt.plot(tt_mid,zz_mid,label='implicit midpoint solution')

### spp method
spp_degree = 4

s_spp = timer()

spp_solution = spp(ph_sys=ph_sys, tt=tt_spp, z0=z0, u=control, degree=spp_degree)

# tt_spp, zz_spp = spp_solution['boundaries'] # function values at chosen discretization points
tt_spp, zz_spp = spp_solution['superfine'] # function values in between the chosen discretization points
# tt_spp, zz_spp = spp_solution['gauss'] # function values at gauss points used for the quadrature

e_spp = timer()
print(f'spp done (dim_sys={D}, nt_spp={nt_spp}, degree={spp_degree}), took {e_spp-s_spp:.2f} seconds')

plt.plot(tt_spp, zz_spp, label='spp solution')


plt.legend()
plt.xlabel('time')
plt.ylabel(r'$z(t)$')
plt.show()
```


## Further notes
- Throughout the codebase, all functions depending on time are vectorized in time. 
This means that, e.g. `eta(z)` must be well-defined for arrays `z` with the shape `z.shape == (number_of_timepoints, state_dimension)`. 
The time index is always at position `0`.
- Since the implementation uses the algorithmic differentiation capabilities of JAX, the implementations of the functions $E$, $J$, $R$, $\eta$ and $B$ need to be written in a JAX-compatible fashion. 
The provided examples should be a good starting point.
- In case of questions, feel free to reach out.

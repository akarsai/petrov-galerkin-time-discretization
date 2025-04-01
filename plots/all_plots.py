#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this file is used to generate all plots in the publication.
#
#

# add the parent directory to sys.path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from jax.scipy.integrate import trapezoid
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import pickle

from helpers.visualization import calculate_error_in_energybalance
from helpers.other import mpl_settings, generate_eoc_table_tex_code
from helpers.gauss import gauss_quadrature_with_values

from spp import spp, eval_spp_solution

import damped_wave.system as damped_wave
import porous_medium.system as porous_medium
import toda_lattice.system as toda_lattice
import rigid_body.system as rigid_body

def energybalance(
        T: float,
        spp_degrees: list,
        kind_info: dict,
        save: bool = True,
        ):
    """
    creates the energybalance plots

    :param T: specifies time horizon [0,T]
    :param spp_degrees: tested polynomial degrees
    :param kind_info: dictionary with settings for tested system kind
    :param save: flag to save pgf figure
    :return: none
    """

    # prepare input parameters
    kind = kind_info['kind']
    # supported: 'damped_wave', 'porous_medium', 'toda', 'rigid_body'

    options = kind_info.get('options', None)

    print(f'\n\n--- running energybalance for {kind} system ---')

    if options is not None:
        print('\noptions:')
        for key,value in options.items():
            print(f'    {key} : {value}')

    # default time mesh width
    nt_spp = T*100 + 1

    # to compare power balance
    savepath = f'{SAVEPATH}/figures/{kind}'

    # set up default number of quadrature / projection nodes
    num_quad_nodes_list = spp_degrees
    num_proj_nodes_list = spp_degrees

    if kind == 'damped_wave':

        nu = options['nu']
        friction_kind = options['friction_kind']

        savepath += f'_nu{int(nu)}_{friction_kind}_friction'

        ph_sys = damped_wave.get_example_system(nu=nu, friction_kind=friction_kind)
        z0, control = damped_wave.get_example_initial_state_and_control(ph_sys)

        # print(f'\nsetup of pH system complete (damped_wave, inner_grid_points = {inner_grid_points}, nu = {nu})')

        num_proj_nodes_list = [2*k for k in spp_degrees] # correct number of gauß nodes for p(rho) = rho + rho^3

        inner_grid_points = ph_sys.info['inner_grid_points']

        spp_filename = f'{SAVEPATH}/damped_wave_nu{int(nu)}_spp_T{T}_igp{inner_grid_points}_{friction_kind}_friction_energybalance'

    elif kind == 'porous_medium':

        # settings
        q = options['q']
        eps = options['eps']
        solution_kind = options['solution_kind']

        savepath += f'_q{q}_eps{eps}_{solution_kind}'

        ph_sys = porous_medium.get_example_system(q=q, eps=eps)
        z0, control, _, _ = porous_medium.get_example_manufactured_solution(ph_sys, kind=solution_kind)

        # print(f'\nsetup of pH system complete (porous_medium, inner_grid_points = {inner_grid_points}, eps = {eps})')

        num_proj_nodes_list = [2*k for k in spp_degrees] # correct number of gauß nodes for q = 3

        inner_grid_points = ph_sys.info['inner_grid_points']

        spp_filename = f'{SAVEPATH}/porous_medium_q{q}_eps{eps}_{solution_kind}_spp_T{T}_igp{inner_grid_points}_energybalance'

    elif kind == 'toda':

        ph_sys = toda_lattice.get_example_system()
        z0, control = toda_lattice.get_example_initial_state_and_control(ph_sys)

        number_of_particles = ph_sys.info['number_of_particles']

        spp_filename = f'{SAVEPATH}/toda_spp_T{T}_N{number_of_particles}_energybalance'

    elif kind == 'rigid_body':

        ph_sys = rigid_body.get_example_system()
        z0, control = rigid_body.get_example_initial_state_and_control(ph_sys)

        spp_filename = f'{SAVEPATH}/rigid_body_spp_T{T}_energybalance'


    dim_sys = ph_sys.info.get('dim_sys')

    # set up plot environment
    fig, ax = plt.subplots()

    ### projection method
    tt_spp = jnp.linspace(0,T,nt_spp) # t_i = i * T/(nt_spp-1)

    for index, spp_degree in enumerate(spp_degrees):

        num_proj_nodes = num_proj_nodes_list[index]
        num_quad_nodes = num_quad_nodes_list[index]

        try: # try to skip also the evaluation
            with open(f'{spp_filename}_n{spp_degree}_pn{num_proj_nodes}_M{nt_spp}.pickle','rb') as f:
                spp_solution = pickle.load(f)['spp_solution']
            print(f'(nt_spp={nt_spp}, degree={spp_degree}, num_proj_nodes={num_proj_nodes}) spp result was loaded')

        except FileNotFoundError: # evaluation was not done before
            s_spp = timer()

            spp_solution = spp(ph_sys=ph_sys, tt=tt_spp, z0=z0, u=control, degree=spp_degree, num_quad_nodes=num_quad_nodes, num_proj_nodes=num_proj_nodes)

            e_spp = timer()
            print(f'spp done (dim_sys={dim_sys}, nt_spp={nt_spp}, degree={spp_degree}, num_proj_nodes={num_proj_nodes}), took {e_spp-s_spp:.2f} seconds')

            # save file
            with open(f'{spp_filename}_n{spp_degree}_pn{num_proj_nodes}_M{nt_spp}.pickle','wb') as f:
                pickle.dump({'spp_solution':spp_solution},f)
            print(f'    result was written')


        error_in_energybalance = calculate_error_in_energybalance(spp_solution, ph_sys, u=control, relative=True,)

        ax.semilogy(
            tt_spp[1:], error_in_energybalance,
            label=f'$k = {spp_degree},~ s_{{\Pi}} = {num_proj_nodes}$',
            color=plt.cm.tab20(2*index)
            )


        if kind == 'toda' and (spp_degree == 1 or spp_degree == 2):

            while num_proj_nodes < spp_degree+1:

                num_proj_nodes += 1

                try: # try to skip also the evaluation
                    with open(f'{spp_filename}_n{spp_degree}_pn{num_proj_nodes}_M{nt_spp}.pickle','rb') as f:
                        spp_solution = pickle.load(f)['spp_solution']
                    print(f'(nt_spp={nt_spp}, degree={spp_degree}, num_proj_nodes={num_proj_nodes}) spp result was loaded')

                except FileNotFoundError: # evaluation was not done before
                    s_spp = timer()
                    spp_solution = spp(ph_sys=ph_sys, tt=tt_spp, z0=z0, u=control, degree=spp_degree, num_quad_nodes=num_quad_nodes, num_proj_nodes=num_proj_nodes)
                    e_spp = timer()
                    print(f'spp done (dim_sys={dim_sys}, nt_spp={nt_spp}, degree={spp_degree}, num_proj_nodes={num_proj_nodes}), took {e_spp-s_spp:.2f} seconds')

                    # save file
                    with open(f'{spp_filename}_n{spp_degree}_pn{num_proj_nodes}_M{nt_spp}.pickle','wb') as f:
                        pickle.dump({'spp_solution':spp_solution},f)
                    print(f'    result was written')

                error_in_energybalance = calculate_error_in_energybalance(spp_solution, ph_sys, u=control, relative=True,)

                ax.semilogy(
                    tt_spp[1:], error_in_energybalance,
                    label=f'$k = {spp_degree},~ s_{{\Pi}} = {num_proj_nodes}$',
                    color=plt.cm.tab20(2*index),
                    linestyle='dotted',
                    )

    ax.legend()
    ax.set_xlabel('time')
    # ax.set_ylabel(r'$\frac{| \mathcal{H}(z_\tau(t_{i})) - \mathcal{H}(z_\tau(t_{i-1})) - Q_i[ - r(\Pi\eta(z_\tau),\Pi\eta(z_\tau)) + b(\cdot, \Pi\eta(z_\tau), \Pi\eta(z_\tau))] | }{ \max_{i=1,\dots,m} | \mathcal{H}(z_\tau(t_{i})) - \mathcal{H}(z_\tau(t_{i-1})) | }$')
    ax.set_ylabel(r'$\mathcal{E}(z_\tau; t_i)$')

    plt.ylim(1.5e-18, 1.5e-3)

    if kind in ['damped_wave', 'porous_medium']:
        ax.set_ylabel(r'$\mathcal{E}(w_{h\tau}; t_i)$')

    # save + show
    if save:
        savepath = savepath + '_energybalance'
        fig.tight_layout()
        plt.savefig(savepath + '.pgf') # save as pgf
        plt.savefig(savepath + '.png') # save as png
        print(f'figure saved under savepath {savepath} (as pgf and png)')

    fig.title = f'relative error in energy balance for different methods, {kind}'
    fig.tight_layout()

    plt.show()

    return

def varying_degree(
        T: float,
        spp_degrees: list,
        kind_info: dict,
        different_sampling: bool = False,
        eoc_table = False,
        save: bool = True,
        ):
    """
    creates the varying_degree plots

    :param T: specifies time horizon [0,T]
    :param spp_degrees: tested polynomial degrees
    :param kind_info: dictionary with settings for tested system kind
    :param different_sampling: [optional] flag for superconvergence analysis, default False
    :param save: [optional] flag to save pgf figure
    :return: none
    """

    # general settings
    base_Delta_t = 1e-3
    num_Delta_t_steps = 9
    Delta_t_array = jnp.array([2**i * base_Delta_t for i in range(num_Delta_t_steps)])
    ref_order_smaller = 3 # by which order of magnitude should the reference solution be smaller than the smallest tested Delta t?
    ref_Delta_t = base_Delta_t/(2**ref_order_smaller) # Delta t for reference solution

    # fetch input parameters
    kind = kind_info['kind']
    # supported: 'toda', 'damped_wave', 'porous_medium', 'rigid_body'

    options = kind_info.get('options', None)

    print(f'\n\n--- running varying_degree for {kind} system ---')

    if options is not None:
        print('\noptions:')
        for key,value in options.items():
            print(f'    {key} : {value}')

    # savepath
    savepath = f'{SAVEPATH}/figures/{kind}'

    # convert Delta_t values to nt values
    groß_n = (T/Delta_t_array[-1]).astype(int)
    nt_array = jnp.flip(jnp.array([groß_n * 2**(order) + 1 for order in range(num_Delta_t_steps)]))
    nt_ref = 2**(ref_order_smaller+num_Delta_t_steps-1) * groß_n + 1

    # convert back to get "real" Delta_t corresponding to the nt_spp values
    Delta_t_array = T/nt_array
    Delta_t_ref = T/nt_ref

    print(f'\nnt_ref = {nt_ref}\nDelta_t_ref = {Delta_t_ref:e}')
    print(f'\nnt_array = {nt_array}\nDelta_t_array = {Delta_t_array}\n')

    if kind == 'damped_wave':

        nu = options['nu']
        friction_kind = options['friction_kind']

        savepath += f'_nu{int(nu)}_{friction_kind}_friction'

        ph_sys = damped_wave.get_example_system(nu=nu, friction_kind=friction_kind)
        z0, control, g, true_solution = damped_wave.get_example_manufactured_solution(ph_sys)
        inner_grid_points = ph_sys.info['inner_grid_points']

        print('\nsetup of pH system complete')

        spp_filename = f'{SAVEPATH}/damped_wave_nu{int(nu)}_spp_T{T}_igp{inner_grid_points}_{friction_kind}_friction_manufactured'

    elif kind == 'porous_medium':

        # settings
        q = options['q']
        eps = options['eps']
        solution_kind = options['solution_kind']
        savepath += f'_q{q}_eps{eps}_{solution_kind}'

        ph_sys = porous_medium.get_example_system(q=q, eps=eps)
        z0, control, g, true_solution = porous_medium.get_example_manufactured_solution(ph_sys, kind=solution_kind)
        inner_grid_points = ph_sys.info['inner_grid_points']

        print('\nsetup of pH system complete')

        spp_filename = f'{SAVEPATH}/porous_medium_q{q}_eps{eps}_{solution_kind}_spp_T{T}_igp{inner_grid_points}'

    elif kind == 'toda':
        ph_sys = toda_lattice.get_example_system()
        z0, control, g, true_solution = toda_lattice.get_example_manufactured_solution(ph_sys)
        number_of_particles = ph_sys.info['number_of_particles']

        spp_filename = f'{SAVEPATH}/toda_spp_T{T}_N{number_of_particles}_manufactured'

    elif kind == 'rigid_body':
        ph_sys = rigid_body.get_example_system()
        z0, control, g, true_solution = rigid_body.get_example_manufactured_solution(ph_sys)

        spp_filename = f'{SAVEPATH}/rigid_body_spp_T{T}_manufactured'


    # obtain reference solution
    tt_ref = jnp.linspace(0,T,nt_ref)
    zz_ref = true_solution(tt_ref)

    ### spp
    def calculate_spp_errors(spp_degree):

        errors_for_this_degree = []

        for k in range(num_Delta_t_steps):

            nt_spp = int(nt_array[k])

            try: # try to skip also the evaluation at the nt_bdf points
                with open(f'{spp_filename}_n{spp_degree}_M{nt_spp}.pickle','rb') as f:
                    spp_solution = pickle.load(f)['spp_solution']
                print(f'(nt_spp={nt_spp}, degree={spp_degree}) spp result was loaded')

            except FileNotFoundError: # evaluation was not done before

                tt_spp = jnp.linspace(0,T,nt_spp)

                if kind in ['damped_wave', 'porous_medium']:
                    num_proj_nodes = 2*spp_degree # projection is exact with this degree for p(rho) = rho + rho^3, q=3, F'(z) = z^3 - z
                elif kind == 'toda':
                    num_proj_nodes = spp_degree
                else: # kind == 'rigid_body:
                    num_proj_nodes = spp_degree

                s_proj = timer()
                spp_solution = spp(ph_sys=ph_sys, tt=tt_spp, z0=z0, u=control, degree=spp_degree, num_proj_nodes=num_proj_nodes, g=g)
                e_proj = timer()
                print(f'(nt_spp={nt_spp}, degree={spp_degree}) spp done, took {e_proj-s_proj:.2f} seconds')

                # save file
                with open(f'{spp_filename}_n{spp_degree}_M{nt_spp}.pickle','wb') as f:
                    pickle.dump({'spp_solution':spp_solution},f)
                print(f'    result was written')


            if different_sampling: # analyze superconvergence

                # eval reference solution on spp time gitter
                tt_spp, zz_spp = spp_solution['boundaries'] # function values at boundaries
                zz_ref_resampled = zz_ref[::2**(k+ref_order_smaller),:]

                diff = zz_ref_resampled - zz_spp

            else:

                # eval spp solution on reference gitter
                zz_spp_on_tt_ref = eval_spp_solution(
                    ph_sys=ph_sys,
                    tt_ref=tt_ref,
                    spp_solution=spp_solution,
                    resample_step=2**(k+ref_order_smaller)
                    )

                diff = zz_ref - zz_spp_on_tt_ref



            # calculate relative error
            error = jnp.linalg.norm(diff, axis=1)#/jnp.linalg.norm(zz_bdf, axis=1) # norms along axis 1, since axis 0 are the time points
            error = error/jnp.max(jnp.linalg.norm(zz_ref, axis=1))

            # plt.plot(tt_ref, error, label='error')
            # plt.legend()
            # plt.show()

            max_error = float(jnp.max(error)) # error in Linf norm

            errors_for_this_degree.append(max_error)

        return errors_for_this_degree

    ### calculate and plot in one go
    fig, ax = plt.subplots()

    markerlist = [
        ('.', 'o'),
        ('1', 'v'),
        ('2', '^'),
        ('3', '<'),
        ('4', '>'),
        ]

    # l = int(num_Delta_t_steps/2) + 1
    l = -3

    # spp calculation + plot
    all_spp_errors = {}
    all_spp_errors_array = jnp.zeros((len(Delta_t_array),len(spp_degrees)))

    for index, spp_degree in enumerate(spp_degrees):

        spp_errors_for_this_degree = calculate_spp_errors(spp_degree)
        all_spp_errors[spp_degree] = spp_errors_for_this_degree
        all_spp_errors_array = all_spp_errors_array.at[:,index].set(jnp.flip(jnp.array(spp_errors_for_this_degree)))

        print(f'spp_degree = {spp_degree} done\n')

        # marker_data, marker_fit = markerlist[index]
        marker_data, marker_fit = '.', 'o'

        ax.loglog(Delta_t_array, spp_errors_for_this_degree,
                   label=f'$k = {spp_degree}$',
                   marker=marker_data,
                   # markersize=20,
                   color=plt.cm.tab20(2*index))

        # add linear fit
        if different_sampling: # analyze superconvergence
            c = spp_errors_for_this_degree[-2]/Delta_t_array[-2]**(2*spp_degree) # find coefficient to match Delta_t^p to curves
            plt.loglog(Delta_t_array, c * Delta_t_array**(2*spp_degree),
                       label=f'$\\tau^{{{2*spp_degree}}}$',
                       linestyle='--',
                       marker=marker_fit,
                       markersize=7,
                       color=plt.cm.tab20(2*index + 1),
                       zorder=0)

        else:
            c = spp_errors_for_this_degree[l]/Delta_t_array[l]**(spp_degree+1) # find coefficient to match Delta_t^p to curves
            ax.loglog(Delta_t_array, c * Delta_t_array**(spp_degree+1),
                   label=f'$\\tau^{spp_degree+1}$',
                   linestyle='--',
                   marker=marker_fit,
                   markersize=7,
                   color=plt.cm.tab20(2*index + 1),
                   zorder=0)

    # print(all_spp_errors_array)

    # create EOC table code
    if eoc_table:
        E_subscript = ''
        if different_sampling:
            E_subscript = '\\tau'

        eoc_table_tex_code = generate_eoc_table_tex_code(
            jnp.flip(Delta_t_array),
            jnp.array(spp_degrees),
            all_spp_errors_array,
            with_average=True,
            E_subscript=E_subscript
            )

    print('\n----\n')

    if kind == 'porous_medium':
        if solution_kind == 'barenblatt':

            # add linear fit with slope 1/(q-1)
            slope = 1/(q-1)

            c_q = all_spp_errors[spp_degrees[1]][l]/Delta_t_array[l]**(slope) # find coefficient to match Delta_t^p to curves
            ax.loglog(Delta_t_array, c_q * Delta_t_array**(slope),
                   label=f'$\\tau^{{{slope}}}$',
                   linestyle='--',
                   marker=marker_fit,
                   markersize=7,
                   color=plt.cm.tab20(len(2*spp_degrees) + 1),
                   zorder=0)


    # set plot properties
    if kind == 'toda' and different_sampling:
        loc = 'upper left'
    else:
        loc = 'best'
    ax.legend(loc=loc)
    ax.set_xlabel('$\\tau$')
    plt.ylim(1.5e-18, 1.5e-1)

    if different_sampling:
        # ylabeltext = '$\\frac{\\max\\limits_{t_0, \dots, t_m} \| z(t) - z_{\\tau}(t) \|}{\\max\\limits_{t_0, \dots, t_m} \| z(t)\|}$'
        # if kind in ['damped_wave', 'porous_medium']:
        #     ylabeltext = '$\\frac{\\max\\limits_{t_0, \dots, t_m} \| w_h(t) - w_{h\\tau}(t) \|}{\\max\\limits_{t_0, \dots, t_m} \| w_h(t)\|}$'
        ylabeltext = '$E_{\\tau}$'
    else:
        # ylabeltext = '$\\frac{\\max_{t \\in [0,T]} \| z(t) - z_{\\tau}(t) \|}{\\max_{t \\in [0,T]} \| z(t)\|}$'
        # if kind in ['damped_wave', 'porous_medium']:
        #     ylabeltext = '$\\frac{\\max_{t \\in [0,T]} \| w_h(t) - w_{h\\tau}(t) \|}{\\max_{t \\in [0,T]} \| w_h(t)\|}$'
        ylabeltext = '$E$'

    titeltext = f'{kind} system, options = {options}'
    ax.set_ylabel(ylabeltext)

    # add horizontal line at 1e-16
    # ax.loglog(Delta_t_array, 0*Delta_t_array + 1e-16,
    #           color='gray',
    #           linewidth=1,
    #           linestyle='dashed',
    #           zorder=0,
    #           )
    # extraticks = [1e-16]
    # ylim = ax.get_ylim()
    # ax.set_yticks(list(ax.get_yticks()) + extraticks)
    # ax.set_ylim(ylim)

    # saving the figure
    if save:
        savepath = savepath + '_varying_degree'
        if different_sampling: savepath = savepath + '_different_sampling'
        fig.tight_layout() # call tight_layout to be safe
        fig.savefig(savepath + '.pgf') # save as pgf
        fig.savefig(savepath + '.png') # save as png
        print(f'\n\nfigure saved under {savepath} (as pgf and png)')

        if eoc_table:
            with open(savepath + '_eoc.tex', 'w') as f:
                f.write(eoc_table_tex_code)
            print(f'eoc table saved under {savepath}_eoc.tex')

    # showing the figure
    ax.set_title(titeltext)
    fig.tight_layout()
    plt.show()

def varying_discretization(
        T: float,
        spp_degree: int,
        igp_list: list,
        kind_info: dict,
        different_sampling: bool = False,
        save: bool = True,
        ):
    """
    creates the varying_discretization plots

    :param T: specifies time horizon [0,T]
    :param spp_degree: tested polynomial degree
    :param igp_list: tested inner grid points
    :param kind_info: dictionary with settings for tested system kind
    :param different_sampling: [optional] flag for superconvergence analysis, default False
    :param save: [optional] flag to save pgf figure
    :return: none
    """

    ### general settings
    base_Delta_t = 1e-3 # ohne 8*
    num_Delta_t_steps = 9 # 9
    Delta_t_array = jnp.array([2**i * base_Delta_t for i in range(num_Delta_t_steps)])
    ref_order_smaller = 3 # 3 by which order of magnitude should the reference solution be smaller than the smallest tested Delta t?
    ref_Delta_t = base_Delta_t/(2**ref_order_smaller) # Delta t for reference solution


    kind = kind_info['kind']


    options = kind_info.get('options', None)

    print(f'\n\n--- running varying_discretization for {kind} system ---')

    if options is not None:
        print('\noptions:')
        for key,value in options.items():
            print(f'    {key} : {value}')


    savepath = f'{SAVEPATH}/{kind}'

    # relative or absolute error
    relative = True

    # convert Delta_t values to nt values
    groß_n = (T/Delta_t_array[-1]).astype(int)
    nt_array = jnp.flip(jnp.array([groß_n * 2**(order) + 1 for order in range(num_Delta_t_steps)]))
    nt_ref = 2**(ref_order_smaller+num_Delta_t_steps-1) * groß_n + 1

    # convert back to get "real" Delta_t corresponding to the nt_spp values
    Delta_t_array = T/nt_array
    Delta_t_ref = T/nt_ref

    print(f'\nnt_ref = {nt_ref}\nDelta_t_ref = {Delta_t_ref:e}')
    print(f'\nnt_array = {nt_array}\nDelta_t_array = {Delta_t_array}\n')

    tt_ref = jnp.linspace(0,T,nt_ref)
    dt_ref = tt_ref[1]-tt_ref[0]


    if kind == 'damped_wave':

        # viscosity parameter
        nu = options['nu']
        friction_kind = options['friction_kind']

        savepath += f'_nu{int(nu)}_{friction_kind}_friction'

        ### for manufactured solution
        # what we want for rho and v
        def rho(t,x):
            return jnp.sin(t) * jnp.sin(x)

        def v(t,x):
            return jnp.sin(t) * jnp.sin(x)

        # partial derivative of v with respect to x
        # dx_v = jax.jit(jax.jacobian(v, argnums=1))
        def dx_v(t,x):
            return jnp.sin(t) * jnp.cos(x)

    elif kind == 'porous_medium':

        q = options['q'] # parameter in porous medium equation
        eps = options['eps']  # regularization parameter, 1e-10, 1e-0
        solution_kind = options['solution_kind'] # kind of the manufactured solution, 'barenblatt' or 'smooth'

        savepath += f'_q{q}_eps{eps}_{solution_kind}'

        l = 15 # always for porous medium

        if solution_kind == 'barenblatt':
            # for manufactured solution, use barenblatt
            d = 1 # dimension of spacial domain
            m = q # parameter in porous medium equation \partial_t z = \Delta( z^m )
            def z(t,x):
                xshift = x-l/2 # shift x so that solution corresponds to space domain [-l/2, l/2]
                fac = m*d + 2 - d
                val = jax.nn.relu( 1 - ( (1/2 - 1/(2*m) ) * xshift**2 )/( fac * (t+1)**(2/fac) ) )**(1/(m-1))
                return 1/((t+1)**(d/fac)) * val

        else: # solution_kind == 'smooth':
            # for manufactured solution, use smooth function
            def z(t,x):
                return jnp.cos(t) * jnp.sin(x)

    all_spp_errors = {}
    # operator_norms = {}

    ### for every space discretization
    for index, igp in enumerate(igp_list):

        if kind == 'damped_wave':
            ph_sys = damped_wave.get_example_system(options={'inner_grid_points': igp}, nu=nu, friction_kind=friction_kind)
            # z0, control = damped_wave.get_example_initial_state_and_control(ph_sys)

            l = ph_sys.info['constants']['l']
            p = ph_sys.info['pressure_law']

            # control as usual, contains boundary conditions
            def control(t):
                u0 = p(rho(t,0.0)) - nu * dx_v(t,0.0)
                ul = p(rho(t,l)) - nu * dx_v(t,l)
                return jnp.vstack((u0, ul)).T

            # true solution evaluated at discretized space points
            xi = jnp.linspace(0,l,igp+2)

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

        elif kind == 'porous_medium':
            ph_sys = porous_medium.get_example_system(options={'inner_grid_points': igp}, q=q, eps=eps)
            l = ph_sys.info['constants']['l']
            xi = jnp.linspace(0,l,igp+2)

            # only scalar t allowed
            def z_discret(t):
                return z(t, xi)

            # control is irrelevant
            def control(t):
                return jnp.zeros((1,))
            control = jax.vmap(control, in_axes=0, out_axes=0,) # 0 = index where time is




        hgrid = l/(igp+1)

        z0 = z_discret(0.)

        # only scalar t allowed
        dt_z_discret = jax.jacobian(z_discret, argnums=0)

        # space discretized and vmapped in time
        z_discret = jax.vmap(z_discret, in_axes=0, out_axes=0,)
        dt_z_discret = jax.vmap(dt_z_discret, in_axes=0, out_axes=0,)

        # define rhs for manufactured solution
        E = ph_sys.E_constant # constant
        J = ph_sys.J # possibly nonlinear
        R = ph_sys.R # nonlinear
        eta = ph_sys.eta # possibly nonlinear
        B = ph_sys.B_constant # constant

        def g(t):
            E_dtz = jnp.einsum('xn, tn -> tx', E, dt_z_discret(t))
            JmR_eta_z = jnp.einsum('txn, tn -> tx', J(z_discret(t)) - R(z_discret(t)), eta(z_discret(t)))
            B_u = jnp.einsum('nm, tm -> tn', B, control(t))

            return E_dtz - (JmR_eta_z + B_u)

        print(f'\nsetup of pH system complete for igp = {igp}')

        if kind == 'damped_wave':
            # ref_filename = f'{SAVEPATH}/damped_wave_ref_T{T}_igp{igp}_ntref{nt_ref}.pickle'
            spp_filename = f'{SAVEPATH}/damped_wave_nu{int(nu)}_spp_T{T}_igp{igp}_{friction_kind}_friction_manufactured'

        elif kind == 'porous_medium':
            spp_filename = f'{SAVEPATH}/porous_medium_q{q}_eps{eps}_{solution_kind}_spp_T{T}_igp{igp}'

        zz_ref = z_discret(tt_ref)



        errors_spp = []

        for k in reversed(range(num_Delta_t_steps)):

            nt_spp = int(nt_array[k])

            try:
                with open(f'{spp_filename}_n{spp_degree}_M{nt_spp}.pickle','rb') as f:
                    spp_solution = pickle.load(f)['spp_solution']
                print(f'(nt_spp={nt_spp}, degree={spp_degree}) spp result was loaded')

            except FileNotFoundError: # evaluation was not done before

                tt_spp = jnp.linspace(0,T,nt_spp)

                # projection is exact with this degree for p(rho) = rho + rho^3, q=3, F'(z) = z^3 - z
                num_proj_nodes = 2*spp_degree

                s_spp = timer()
                spp_solution = spp(ph_sys=ph_sys, tt=tt_spp, z0=z0, u=control, degree=spp_degree, num_proj_nodes=num_proj_nodes, g=g) # with g -> manufactured solution
                e_spp = timer()
                print(f'(nt_spp={nt_spp}, degree={spp_degree}) spp done, took {e_spp-s_spp:.2f} seconds')

                # save file
                with open(f'{spp_filename}_n{spp_degree}_M{nt_spp}.pickle','wb') as f:
                    pickle.dump({'spp_solution':spp_solution},f)
                print(f'    result was written')

            if different_sampling:

                # eval reference solution on spp time gitter
                tt_spp, zz_spp = spp_solution['boundaries'] # function values at boundaries
                zz_ref_resampled = zz_ref[::2**(k+ref_order_smaller),:]

                diff = zz_ref_resampled - zz_spp

            else:

                # eval spp solution on reference time gitter
                zz_spp_on_tt_ref = eval_spp_solution(
                    ph_sys=ph_sys,
                    tt_ref=tt_ref,
                    spp_solution=spp_solution,
                    resample_step=2**(k+ref_order_smaller),
                    )

                diff = zz_ref - zz_spp_on_tt_ref


            # calculate error
            error_spp = jnp.sqrt(jnp.einsum('tn,nD,tD->t', diff, E, diff)) # the squared L2 norm in space obtained using the mass matrix
            # error_spp = jnp.linalg.norm(diff, axis=1) # norms along axis 1, since axis 0 are the time points
            if relative:
                # error_spp = error_spp/jnp.max(jnp.linalg.norm(zz_ref, axis=1))
                error_spp = error_spp/jnp.max(jnp.sqrt(jnp.einsum('tn,nD,tD->t', zz_ref, E, zz_ref)))

            max_error_spp = float(jnp.max(error_spp)) # error in Linf norm

            # errors_spp.append(max_error_spp)  # for `k in range`
            errors_spp.insert(0, max_error_spp) # for `k in reversed(range)`


        # errors_spp now contains the errors for the spp solution
        print(f'errors for igp = {igp}\n{errors_spp}')
        all_spp_errors[igp] = errors_spp
        # operator_norms[igp] = jnp.linalg.norm(jnp.linalg.inv(ph_sys.E_constant) @ ph_sys.J_constant, ord=2)

        ## plot spp
        plt.loglog(
            Delta_t_array, errors_spp,
            label=f'{igp} spatial grid points',
            marker='.',
            # markersize=20,
            # color=plt.cm.tab20(2*index)
            )



    # plot for comparison
    if different_sampling:

        i = int(num_Delta_t_steps/2) + 2

        c = all_spp_errors[igp][i]/Delta_t_array[i]**(2*spp_degree) # find coefficient to match Delta_t^p to curves

        plt.loglog(
            Delta_t_array, c * Delta_t_array**(2*spp_degree),
            label=f'$\\tau^{2*spp_degree}$',
            linestyle='--',
            marker='o',
            markersize=7,
            # color=plt.cm.tab20(2*index + 1),
            zorder=0,
            alpha=0.2,
            )

    else:
        slope = spp_degree + 1

        if kind == 'porous_medium':
            if solution_kind == 'barenblatt':
                if q == 1.5:
                    slope = 2
                elif q == 2.0:
                    slope = 1
                elif q == 3.0:
                    slope = 1/2
            elif solution_kind == 'smooth':
                if q == 1.5:
                    slope = 2
                elif q == 2.0:
                    slope = 2
                # elif q == 3.0:
                #     slope = 1/2


        i = int(num_Delta_t_steps/2) + 1

        c = all_spp_errors[igp][i]/Delta_t_array[i]**(slope) # find coefficient to match Delta_t^p to curves

        plt.loglog(
            Delta_t_array, c * Delta_t_array**(slope),
            label=f'$\\tau^{{{slope}}}$',
            linestyle='--',
            marker='o',
            markersize=7,
            # color=plt.cm.tab20(2*index + 1),
            zorder=0,
            alpha=0.2,
            )

    # set plot properties
    plt.legend()
    plt.xlabel('$\\tau$')
    plt.ylim(1.5e-18, 1.5e-1)

    if different_sampling:
        # ylabeltext = '$\\frac{\\max\\limits_{t_0, \dots, t_m} \| z_h(t) - z_{h\\tau}(t) \|}{\\max\\limits_{t_0, \dots, t_m} \| z_h(t)\|}$'
        # ylabeltext = '$\\frac{\\max\\limits_{t_0, \dots, t_m} \| w_h(t) - w_{h\\tau}(t) \|}{\\max\\limits_{t_0, \dots, t_m} \| w_h(t)\|}$'
        # titletext = '\\noindent relative $L^{\\infty}$ error compared to true solution \\\\ varying number of spatial grid points'
        ylabeltext = '$E_{\\tau}$'
    else:
        # ylabeltext = '$\\frac{\\max_{t \\in [0,T]} \| z_h(t) - z_{h\\tau}(t) \|}{\\max_{t \\in [0,T]} \| z_h(t)\|}$'
        # ylabeltext = '$\\frac{\\max_{t \\in [0,T]} \| w_h(t) - w_{h\\tau}(t) \|}{\\max_{t \\in [0,T]} \| w_h(t)\|}$'
        # titletext = '\\noindent relative $L^{\\infty}$ error compared to true solution \\\\ varying number of spatial grid points'
        ylabeltext = '$E$'

    titletext = f'{kind} system, options = {options}'

    plt.ylabel(ylabeltext)

    # saving the figure
    if save:
        savepath = savepath + '_varying_discretization'
        if different_sampling: savepath = savepath + '_different_sampling'
        plt.tight_layout() # call tight_layout to be safe
        plt.savefig(savepath + '.pgf') # save as pgf
        plt.savefig(savepath + '.png') # save as png
        print(f'\n\nfigure saved under {savepath} (as pgf and png)')

    # showing the figure
    plt.title(titletext)
    plt.tight_layout()
    plt.show()

    ## print all spp errors and factors
    # print('\n\n')
    #
    # error_0 = all_spp_errors[igp_list[0]]
    #
    # for igp, error in all_spp_errors.items():
    #
    #     print(f'errors for igp = {igp} divided by errors for igp = {igp_list[0]}:\n{jnp.array(error)/jnp.array(error_0)}')
    #
    #     print(f'\n ---- \n')

    return

def varying_quadrature(
        T: float,
        spp_degree: int,
        num_quad_nodes_list: list,
        kind_info: dict,
        save: bool = True,
        ):
    """
    creates the varying_quadrature plots

    :param T: specifies time horizon [0,T]
    :param spp_degree: tested polynomial degree
    :param num_quad_nodes_list: tested quadrature nodes
    :param kind_info: dictionary with settings for tested system kind
    :param save: [optional] flag to save pgf figure
    :return: none
    """

    ### general settings
    base_Delta_t = 1e-3
    num_Delta_t_steps = 9
    Delta_t_array = jnp.array([2**i * base_Delta_t for i in range(num_Delta_t_steps)])
    ref_order_smaller = 3 # by which order of magnitude should the reference solution be smaller than the smallest tested Delta t?
    ref_Delta_t = base_Delta_t/(2**ref_order_smaller) # Delta t for reference solution

    kind = kind_info['kind'] # only 'toda' is supported right now
    print(f'\n\n--- running varying_quadrature for {kind} system ---')

    # convert Delta_t values to nt values
    groß_n = (T/Delta_t_array[-1]).astype(int)
    nt_array = jnp.flip(jnp.array([groß_n * 2**(order) + 1 for order in range(num_Delta_t_steps)]))
    nt_ref = 2**(ref_order_smaller+num_Delta_t_steps-1) * groß_n + 1

    # convert back to get "real" Delta_t corresponding to the nt_spp values
    Delta_t_array = T/nt_array
    Delta_t_ref = T/nt_ref

    print(f'\nnt_ref = {nt_ref}\nDelta_t_ref = {Delta_t_ref:e}')
    print(f'\nnt_array = {nt_array}\nDelta_t_array = {Delta_t_array}\n')


    if kind == 'toda':
        ph_sys = toda_lattice.get_example_system()
        z0, control, g, true_solution = toda_lattice.get_example_manufactured_solution(ph_sys)
        number_of_particles = ph_sys.info['number_of_particles']


        spp_filename = f'{SAVEPATH}/toda_spp_T{T}_N{number_of_particles}_manufactured'


    # obtain reference solution
    tt_ref = jnp.linspace(0,T,nt_ref)
    zz_ref = true_solution(tt_ref)

    ### spp
    def calculate_spp_errors(num_quad_nodes):

        errors_for_theses_quad_nodes = []

        for k in range(num_Delta_t_steps):

            nt_spp = int(nt_array[k])

            try:
                with open(f'{spp_filename}_n{spp_degree}_M{nt_spp}_qn{num_quad_nodes}.pickle','rb') as f:
                    spp_solution = pickle.load(f)['spp_solution']
                print(f'(nt_spp={nt_spp}, degree={spp_degree}, num_quad_nodes={num_quad_nodes}) spp result was loaded')

            except FileNotFoundError: # evaluation was not done before

                tt_spp = jnp.linspace(0,T,nt_spp)

                s_proj = timer()
                spp_solution = spp(ph_sys=ph_sys, tt=tt_spp, z0=z0, u=control, degree=spp_degree, num_quad_nodes=num_quad_nodes, g=g)
                e_proj = timer()
                print(f'(nt_spp={nt_spp}, degree={spp_degree}, num_quad_nodes={num_quad_nodes}) spp done, took {e_proj-s_proj:.2f} seconds')

                # save file
                with open(f'{spp_filename}_n{spp_degree}_M{nt_spp}_qn{num_quad_nodes}.pickle','wb') as f:
                    pickle.dump({'spp_solution':spp_solution},f)
                print(f'    result was written')

            # eval spp solution on reference gitter
            zz_spp_on_tt_ref = eval_spp_solution(
                ph_sys=ph_sys,
                tt_ref=tt_ref,
                spp_solution=spp_solution,
                resample_step=2**(k+ref_order_smaller)
                )

            # eval true solution on spp time gitter
            # zz_ref_resampled = zz_ref[::2**(k+ref_order_smaller),:]
            # zz_ref_resampled = zz_ref_interp(tt_spp)

            diff = zz_ref - zz_spp_on_tt_ref

            # calculate error
            error = jnp.linalg.norm(diff, axis=1)#/jnp.linalg.norm(zz_bdf, axis=1) # norms along axis 1, since axis 0 are the time points
            error = error/jnp.max(jnp.linalg.norm(zz_ref, axis=1))

            max_error = float(jnp.max(error)) # error in Linf norm

            errors_for_theses_quad_nodes.append(max_error)

        return errors_for_theses_quad_nodes

    ### calculate and plot in one go
    markerlist = [
        ('.', 'o'),
        ('1', 'v'),
        ('2', '^'),
        ('3', '<'),
        ('4', '>'),
        ]

    # l = int(num_Delta_t_steps/2)
    l = -1

    # spp calculation + plot
    for index, num_quad_nodes in enumerate(num_quad_nodes_list):

        errors_for_theses_quad_nodes = calculate_spp_errors(num_quad_nodes)

        print(f'num_quad_nodes = {num_quad_nodes} done\n')

        # marker_data, marker_fit = markerlist[index]
        marker_data, marker_fit = '.', 'o'

        c = errors_for_theses_quad_nodes[l]/Delta_t_array[l]**(num_quad_nodes+1) # find coefficient to match Delta_t^p to curves
        # c = 1

        labeltext = f'$s_Q = {num_quad_nodes}$'

        plt.loglog(Delta_t_array, errors_for_theses_quad_nodes,
                   label=labeltext,
                   marker=marker_data,
                   # markersize=20,
                   color=plt.cm.tab20(2*index))
        plt.loglog(Delta_t_array, c * Delta_t_array**(num_quad_nodes+1),
                   label=f'$\\tau^{num_quad_nodes+1}$',
                   linestyle='--',
                   marker=marker_fit,
                   markersize=7,
                   color=plt.cm.tab20(2*index + 1),
                   zorder=0)

    # set plot properties
    plt.legend()
    plt.xlabel('$\\tau$')
    plt.ylim(1.5e-18, 1.5e-1)

    # ylabeltext = '$\\frac{\\max_{t \\in [0,T]} \| z(t) - z_{\\tau}(t) \|}{\\max_{t \\in [0,T]} \| z(t)\|}$'
    ylabeltext = '$E$'
    titeltext = 'relative $L^{\\infty}$ error compared to true solution'
    plt.ylabel(ylabeltext)

    # saving the figure
    if save:
        savepath = f'{SAVEPATH}/figures/{kind}_varying_quadrature'
        plt.tight_layout() # call tight_layout to be safe
        plt.savefig(savepath + '.pgf') # save as pgf
        plt.savefig(savepath + '.png') # save as png
        print(f'\n\nfigure saved under {savepath} (as pgf and png)')

    # showing the figure
    plt.title(titeltext)
    plt.tight_layout()
    plt.show()

    return

def varying_projection(
        T: float,
        spp_degree: int,
        num_proj_nodes_list: list,
        kind_info: dict,
        save: bool = True,
        ):
    """
    creates the varying_projection plots

    :param T: specifies time horizon [0,T]
    :param spp_degree: tested polynomial degree
    :param num_proj_nodes_list: tested quadrature nodes
    :param kind_info: dictionary with settings for tested system kind
    :param save: [optional] flag to save pgf figure
    :return: none
    """

    ### general settings
    base_Delta_t = 1e-3
    num_Delta_t_steps = 9
    Delta_t_array = jnp.array([2**i * base_Delta_t for i in range(num_Delta_t_steps)])
    ref_order_smaller = 3 # by which order of magnitude should the reference solution be smaller than the smallest tested Delta t?
    ref_Delta_t = base_Delta_t/(2**ref_order_smaller) # Delta t for reference solution

    kind = kind_info['kind'] # only 'toda' is supported right now
    print(f'\n\n--- running varying_projection for {kind} system ---')

    # convert Delta_t values to nt values
    groß_n = (T/Delta_t_array[-1]).astype(int)
    nt_array = jnp.flip(jnp.array([groß_n * 2**(order) + 1 for order in range(num_Delta_t_steps)]))
    nt_ref = 2**(ref_order_smaller+num_Delta_t_steps-1) * groß_n + 1

    # convert back to get "real" Delta_t corresponding to the nt_spp values
    Delta_t_array = T/nt_array
    Delta_t_ref = T/nt_ref

    print(f'\nnt_ref = {nt_ref}\nDelta_t_ref = {Delta_t_ref:e}')
    print(f'\nnt_array = {nt_array}\nDelta_t_array = {Delta_t_array}\n')


    if kind == 'toda':
        ph_sys = toda_lattice.get_example_system()
        z0, control, g, true_solution = toda_lattice.get_example_manufactured_solution(ph_sys)
        number_of_particles = ph_sys.info['number_of_particles']


        spp_filename = f'{SAVEPATH}/toda_spp_T{T}_N{number_of_particles}_manufactured'


    # obtain reference solution
    tt_ref = jnp.linspace(0,T,nt_ref)
    zz_ref = true_solution(tt_ref)


    ### spp
    def calculate_spp_errors(num_proj_nodes):

        errors_for_these_proj_nodes = []

        for k in range(num_Delta_t_steps):

            nt_spp = int(nt_array[k])

            try:
                with open(f'{spp_filename}_n{spp_degree}_M{nt_spp}_pn{num_proj_nodes}.pickle','rb') as f:
                    spp_solution = pickle.load(f)['spp_solution']
                print(f'(nt_spp={nt_spp}, degree={spp_degree}, num_proj_nodes={num_proj_nodes}) spp result was loaded')

            except FileNotFoundError: # evaluation was not done before

                tt_spp = jnp.linspace(0,T,nt_spp)

                s_proj = timer()
                spp_solution = spp(ph_sys=ph_sys, tt=tt_spp, z0=z0, u=control, degree=spp_degree, num_proj_nodes=num_proj_nodes, g=g)
                e_proj = timer()
                print(f'(nt_spp={nt_spp}, degree={spp_degree}, num_proj_nodes={num_proj_nodes}) spp done, took {e_proj-s_proj:.2f} seconds')

                # save file
                with open(f'{spp_filename}_n{spp_degree}_M{nt_spp}_pn{num_proj_nodes}.pickle','wb') as f:
                    pickle.dump({'spp_solution':spp_solution},f)
                print(f'    result was written')

            # eval spp solution on reference gitter
            zz_spp_on_tt_ref = eval_spp_solution(
                ph_sys=ph_sys,
                tt_ref=tt_ref,
                spp_solution=spp_solution,
                resample_step=2**(k+ref_order_smaller)
                )


            # eval bdf solution on spp time gitter
            # zz_ref_resampled = zz_ref[::2**(k+ref_order_smaller),:]
            # zz_ref_resampled = zz_ref_interp(tt_spp)

            diff = zz_ref - zz_spp_on_tt_ref

            # calculate error
            error = jnp.linalg.norm(diff, axis=1)#/jnp.linalg.norm(zz_bdf, axis=1) # norms along axis 1, since axis 0 are the time points
            error = error/jnp.max(jnp.linalg.norm(zz_ref, axis=1))

            max_error = float(jnp.max(error)) # error in Linf norm

            errors_for_these_proj_nodes.append(max_error)

        return errors_for_these_proj_nodes


    ### calculate and plot in one go
    markerlist = [
        ('.', 'o'),
        ('1', 'v'),
        ('2', '^'),
        ('3', '<'),
        ('4', '>'),
        ]

    # l = int(num_Delta_t_steps/2)
    l = -1

    # spp calculation + plot
    for index, num_proj_nodes in enumerate(num_proj_nodes_list):

        errors_for_theses_proj_nodes = calculate_spp_errors(num_proj_nodes)

        print(f'num_proj_nodes = {num_proj_nodes} done\n')

        # marker_data, marker_fit = markerlist[index]
        marker_data, marker_fit = '.', 'o'

        c = errors_for_theses_proj_nodes[l]/Delta_t_array[l]**(num_proj_nodes+1) # find coefficient to match Delta_t^p to curves
        # c = 1


        labeltext = f'$s_\Pi = {num_proj_nodes}$'

        plt.loglog(Delta_t_array, errors_for_theses_proj_nodes,
                   label=labeltext,
                   marker=marker_data,
                   # markersize=20,
                   color=plt.cm.tab20(2*index))
        plt.loglog(Delta_t_array, c * Delta_t_array**(num_proj_nodes+1),
                   label=f'$\\tau^{num_proj_nodes+1}$',
                   linestyle='--',
                   marker=marker_fit,
                   markersize=7,
                   color=plt.cm.tab20(2*index + 1),
                   zorder=0)


    # set plot properties
    plt.legend()
    plt.xlabel('$\\tau$')
    plt.ylim(1.5e-18, 1.5e-1)
    # plt.ylim(1e-17,1e-1)

    # ylabeltext = '$\\frac{\\max_{t \\in [0,T]} \| z(t) - z_{\\tau}(t) \|}{\\max_{t \\in [0,T]} \| z(t)\|}$'
    ylabeltext = '$E$'
    titeltext = 'relative $L^{\\infty}$ error compared to true solution'
    plt.ylabel(ylabeltext)

    # saving the figure
    if save:
        savepath = f'{SAVEPATH}/figures/{kind}_varying_projection'
        plt.tight_layout() # call tight_layout to be safe
        plt.savefig(savepath + '.pgf') # save as pgf for latex
        plt.savefig(savepath + '.png') # save as png for preview
        print(f'\n\nfigure saved under {savepath} (as pgf and png)')

    # showing the figure
    plt.title(titeltext)
    plt.tight_layout()
    plt.show()

    return




if __name__ == '__main__':

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_debug_nans", True)

    # plot settings
    mpl_settings(figsize=(5.5,4), latex_font='computer modern')

    # set savepath
    SAVEPATH = './results'

    ### default options for all plots

    # time horizon [0,T]
    T = 5

    # flags
    create_energybalance_plots = True
    create_varying_degree_plots = True
    create_varying_discretization_plots = True # this needs lots of time!
    create_varying_quadrature_plots = True
    create_varying_projection_plots = True


    ######################################
    ##### create energybalance plots #####
    ######################################

    if create_energybalance_plots:

        # toda lattice
        energybalance(
            T = T,
            spp_degrees = [1,2,3,4],
            kind_info = {'kind': 'toda'},
            )

        # spinning rigid body
        energybalance(
            T = T,
            spp_degrees = [1,2,3,4],
            kind_info = {'kind': 'rigid_body'},
            )

        # nonlinear wave equation
        for nu in [1.0]:
            for friction_kind in ['irregular']:
                energybalance(
                    T = T,
                    spp_degrees = [1,2,3,4],
                    kind_info = {'kind': 'damped_wave', 'options': {'nu': nu, 'friction_kind': friction_kind}},
                    )

        # porous medium equation
        for q in [1.5, 2.0, 3.0]:
            for eps in [1e-10]:
                for solution_kind in ['barenblatt', 'smooth']: # only used for initial condition
                    energybalance(
                        T = T,
                        spp_degrees = [1,2,3,4],
                        kind_info = {'kind': 'porous_medium', 'options': {'q': q, 'eps': eps, 'solution_kind': solution_kind}},
                        )


    ######################################
    #### create varying_degree plots #####
    ######################################

    if create_varying_degree_plots:

        # toda
        varying_degree(
            T = T,
            spp_degrees = [1,2,3,4], # tested polynomial degrees
            kind_info = {'kind': 'toda'},
            )

        # toda, nodal superconvergence
        varying_degree(
            T = T,
            spp_degrees = [1,2,3,4], # tested polynomial degrees
            kind_info = {'kind': 'toda'},
            different_sampling = True,
            )

        # rigid_body
        varying_degree(
            T = T,
            spp_degrees = [1,2,3,4],
            kind_info = {'kind': 'rigid_body'},
            )

        # rigid_body, nodal superconvergence
        varying_degree(
            T = T,
            spp_degrees = [1,2,3,4],
            kind_info = {'kind': 'rigid_body'},
            different_sampling = True,
            )

        # nonlinear wave equation
        for nu in [0.0, 1.0]:
            for friction_kind in ['irregular']:

                # convergence
                varying_degree(
                    T = T,
                    spp_degrees = [2,4,6],
                    kind_info = {'kind': 'damped_wave', 'options': {'nu': nu, 'friction_kind': friction_kind}},
                    )

                # nodal superconvergence
                varying_degree(
                    T = T,
                    spp_degrees = [2,4,6],
                    kind_info = {'kind': 'damped_wave', 'options': {'nu': nu, 'friction_kind': friction_kind}},
                    different_sampling = True,
                    )

        # porous medium equation
        for q in [1.5, 2.0, 3.0]:
            if q == 1.5:
                epslist = [1e-10, 1e-8]
            else:
                epslist = [1e-10]
            for eps in epslist:
                for solution_kind in ['barenblatt', 'smooth']:

                    # convergence
                    varying_degree(
                        T = T,
                        spp_degrees = [2,4],
                        kind_info = {'kind': 'porous_medium', 'options': {'q': q, 'eps': eps, 'solution_kind': solution_kind}},
                        eoc_table = True,
                        )

                    # nodal superconvergence
                    varying_degree(
                        T = T,
                        spp_degrees = [2,4],
                        kind_info = {'kind': 'porous_medium', 'options': {'q': q, 'eps': eps, 'solution_kind': solution_kind}},
                        different_sampling = True,
                        eoc_table = True,
                        )


    ##############################################
    #### create varying_discretization plots #####
    ##############################################

    if create_varying_discretization_plots:

        # nonlinear wave equation
        for nu in [1.0]:
            for friction_kind in ['irregular']:

                # convergence
                varying_discretization(
                    T = T,
                    spp_degree = 4,
                    igp_list = [8,16,32,64],
                    kind_info = {'kind': 'damped_wave', 'options': {'nu': nu, 'friction_kind': friction_kind}},
                    )

                # nodal superconvergence
                varying_discretization(
                    T = T,
                    spp_degree = 4,
                    igp_list = [8,16,32,64],
                    kind_info = {'kind': 'damped_wave', 'options': {'nu': nu, 'friction_kind': friction_kind}},
                    different_sampling = True,
                    )


        # porous medium equation
        for q in [1.5, 2.0, 3.0]:
            for eps in [1e-10]:
                for solution_kind in ['barenblatt', 'smooth']:

                    # convergence
                    varying_discretization(
                        T = T,
                        spp_degree = 4,
                        igp_list = [8,16,32,64],
                        kind_info = {'kind': 'porous_medium', 'options': {'q': q, 'eps': eps, 'solution_kind': solution_kind}},
                        )

                    # nodal superconvergence
                    varying_discretization(
                        T = T,
                        spp_degree = 4,
                        igp_list = [8,16,32,64],
                        kind_info = {'kind': 'porous_medium', 'options': {'q': q, 'eps': eps, 'solution_kind': solution_kind}},
                        different_sampling = True,
                        )


    ##########################################
    #### create varying_quadrature plots #####
    ##########################################

    if create_varying_quadrature_plots:

        # toda
        varying_quadrature(
            T = T,
            spp_degree = 3,
            num_quad_nodes_list = [1,2,3,4], # tested polynomial degrees
            kind_info = {'kind': 'toda'},
            )

    ##########################################
    #### create varying_projection plots #####
    ##########################################

    if create_varying_projection_plots:

        # toda
        varying_projection(
            T = T,
            spp_degree = 3,
            num_proj_nodes_list = [1,2,3,4], # tested projection nodes
            kind_info = {'kind': 'toda'},
            )


#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this file implements a helper class to style print output using
# ansi codes and a helper function to prepare matplotlib figures
# for publication.
#


import matplotlib
import matplotlib.pyplot as plt
import jax.numpy as jnp
import re

def mpl_settings(
        figsize: tuple = (5.5,4),
        backend: str = None,
        latex_font: str = 'none',
        dpi: int = None,
        ) -> None:
    """
    sets matplotlib settings for latex

    :return: None
    """

    # plt.rcParams['figure.dpi'] = 140
    plt.rcParams['figure.dpi'] = 500
    # default for paper: (5.5,4)
    plt.rcParams["figure.figsize"] = figsize
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb}')

    plt.rcParams.update({
            "pgf.texsystem": "pdflatex",
            "pgf.rcfonts": False,      # don't setup fonts from rc parameters
            "pgf.preamble": '\\usepackage{amsmath,amssymb}',
            "savefig.transparent": True
            })

    if latex_font == 'times':
        plt.rc('font',**{'family':'serif','serif':['Times']})
    elif latex_font == 'computer modern':
        plt.rc('font',**{'family':'serif'})

    plt.rc('axes.formatter', useoffset=False)
    # plt.rcParams['savefig.transparent'] = True

    if backend is not None:
        matplotlib.use(backend)
        plt.rcParams['figure.dpi'] = 140

    if dpi is not None:
        plt.rcParams['figure.dpi'] = dpi

    return

def scientific_notation_tex_code(number: float) -> str:

    number_string = f'{number:.2e}'

    # regex pattern to match scientific notation
    pattern = r"([+-]?\d+\.\d+)e([+-]?\d+)"

    # replacement format
    replacement = r"\1 \\cdot 10^{\2}"

    # Perform the replacement
    output = re.sub(pattern, replacement, number_string)

    return output

def generate_eoc_table_tex_code(
        tau_list: jnp.ndarray,
        k_list: jnp.ndarray,
        error_list: jnp.ndarray,
        with_average: bool = False,
        error_threshold: float = 1e-14,
        E_subscript: str = '',
        ) -> str:
    """
    generates latex code for an eoc table.

    the formula for the experimental order of convergence (eoc) is

    eoc = log(error_2/error_1) / log(tau_2/tau_1)

    :param tau_list: jnp.ndarray, list of tau values (time discretization step sizes)
    :param k_list: jnp.ndarray, list of k values (polynomial degrees)
    :param error_list: jnp.ndarray, list of errors, shape (tau, k) (rows: tau, columns: k)
    :param with_average: bool, if True, an average line is added to the table
    :param error_threshold: float, if error is below this threshold, eoc is not shown in the table and does not count towards the average
    :param E_subscript: string, subscript $E_{error_type}$ in the table
    :return: str, latex code
    """

    # first, prepare eoc list
    log_error_div = jnp.log(error_list[1:,:]/error_list[:-1,:])
    log_tau_div = jnp.log(tau_list[1:]/tau_list[:-1])
    eoc_list = jnp.einsum('tk,t->tk', log_error_div, 1/log_tau_div)
    eoc_list = jnp.concatenate((-jnp.inf*jnp.ones((1,k_list.shape[0])), eoc_list), axis=0) # put -1 in the first row where we have no eoc

    # first line of tex code, defining the number of columns
    latex = '\n\\begin{tabular}{|c'
    for k in k_list:
        latex += '|cc'
    latex += '|}\n\\hline\n'

    # second line of tex code, defining the header
    latex += '    \\multirow{2}{\\widthof{$\\tau$}}{$\\tau$}'
    for k_index, k in enumerate(k_list):
        if k_index == k_list.shape[0]-1:
            arg = '|c|'
        else:
            arg = '|c'
        latex += ' & \\multicolumn{2}{'+arg+'}{$k='+str(k)+'$}'
    latex += '\\Tstrut \\\\\n    '

    # third line of tex code
    for k in k_list:
        latex += f' & error $E_{{{E_subscript}}}$ & EOC'

    # actual content
    latex += '\\Bstrut \\\\ \\hline'
    for tau_index, tau in enumerate(tau_list):
        latex += f'\n    ${scientific_notation_tex_code(tau)}$'
        for k_index, k in enumerate(k_list):
            if eoc_list[tau_index, k_index] < 0 or error_list[tau_index-1, k_index] < error_threshold: # if eoc is not available / sensible
                eoc_string = '-'
            else: # if eoc is available, format properly
                eoc_string = f'${eoc_list[tau_index, k_index]:.2f}$'
            latex += f' & ${scientific_notation_tex_code(error_list[tau_index, k_index])}$ & {eoc_string}'
        if tau_index == 0:
            latex += '\\Tstrut'
        elif tau_index == tau_list.shape[0]-1:
            latex += '\\Bstrut'
        latex += ' \\\\'

    # average line - old version, counts everything
    # if with_average:
    #     latex += '\n\\hline\n     '
    #     for k_index, k in enumerate(k_list):
    #         eoc_avg = jnp.mean(eoc_list[1:,k_index])
    #         latex += f' & & $\\hspace{{-8pt}}\\diameter {eoc_avg:.2f}$'

    # average line
    if with_average:
        latex += '\n\\hline\n     '
        for k_index, k in enumerate(k_list):
            eoc_avg_list = []
            for tau_index, tau in enumerate(tau_list):
                if error_list[tau_index-1, k_index] > error_threshold and eoc_list[tau_index, k_index] != -jnp.inf:
                    eoc_avg_list.append(eoc_list[tau_index, k_index])
            eoc_avg = jnp.mean(jnp.array(eoc_avg_list))
            latex += f' & & $\\hspace{{-11pt}}\\diameter\; {eoc_avg:.2f}$'
    latex += '\\Tstrut\\Bstrut \\\\ \\hline\n'

    # last line
    latex += '\\end{tabular}\n'

    # print(latex)
    return latex

class style:
    blue = '\033[38;5;027m'
    success = '\033[38;5;028m'
    warning = '\033[38;5;208m'
    fail = '\033[38;5;196m'
    #
    bold = '\033[1m'
    underline = '\033[4m'
    italic = '\033[3m'
    end = '\033[0m'



if __name__ == "__main__":

    tau_list = jnp.array([0.2,0.1,0.01,0.001])
    k_list = jnp.array([2,4])
    error_list = jnp.array([[100,50],[50,25],[25,12.5],[12.5,6.25]])
    # eoc_list = jnp.array([[-1,-1],[1,1]])

    tex_code = generate_eoc_table_tex_code(
        tau_list,
        k_list,
        error_list,
        )

    print(tex_code)

    # scientific_notation_tex_code(9.999e-15)
    # scientific_notation_tex_code(1.2345e+4)
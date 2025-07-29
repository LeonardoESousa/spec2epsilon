import warnings
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import curve_fit
from scipy.optimize import minimize
# pylint: disable=unbalanced-tuple-unpacking

THECOLOR = "black"
cmap = plt.get_cmap("cividis")


def set_fontsize(ax):
    fig_size = ax.get_figure().get_size_inches()
    # define font size based dynamically on figure size
    fontsize = max(fig_size[0] * 100 / 72, 14)
    return fontsize


def check(ax, xmin, xmax):
    x = sorted([xmin, xmax])
    y = None
    for elem in ax.get_children():
        try:
            vert = elem.get_paths()[0].vertices
            xs = list(sorted(vert[:, 0]))
            if xs == x and 0 not in vert[:, 1]:
                y = vert[1, 1]
        except (IndexError, AttributeError):
            pass
    return y


def fill(ax, xmin, xmax, y, text):
    fontsize = set_fontsize(ax)
    newy = check(ax, xmin, xmax)
    try:
        ax.fill_between([xmin, xmax], y, newy, alpha=0.5, hatch="x", color=cmap(0.5))
        txt_x = xmin + (xmax - xmin) / 2
        for txt in ax.texts:
            if txt.get_position()[0] == txt_x and txt.get_position()[1] != -0.4:
                txt.set_visible(False)
        ax.text(
            x=txt_x,
            y= 0.95 * min(newy, y),
            s=text,
            ha="center",
            va="top",
            color=THECOLOR,
            fontsize=fontsize,
        )
    except TypeError :
        ax.text(
            x=xmin + (xmax - xmin) / 2,
            y=0.95 * y,
            s=text,
            ha="center",
            va="top",
            color=THECOLOR,
            fontsize=fontsize,
        )



def format_number(rate, error_rate, unit="s^-1"):
    # Check if the rate is zero
    if rate <= 1e-99:
        return f"0 ± 0 {unit}"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exp = np.floor(np.nan_to_num(np.log10(rate)))

    # Adjust exponent to ensure the first number is >= 1
    if rate / 10**exp < 1:
        exp -= 1

    num = 2
    # Determine the number of significant figures for rate and error_rate
    rate_sig_figs = max(num, -int(np.floor(np.log10(error_rate / 10**exp))))  # Ensure at least 1 significant figure
    error_rate_sig_figs = max(num, -int(np.floor(np.log10(error_rate / 10**exp))))  # Ensure at least 1 significant figure

    # Format the string without using LaTeX
    if exp > num:
        formatted_rate = f"{rate/10**exp:.{rate_sig_figs}f}"
        formatted_error_rate = f"{error_rate/10**exp:.{error_rate_sig_figs}f}"
        formatted_string = f"({formatted_rate} ± {formatted_error_rate}) x 10^{int(exp)} {unit}"
    else:
        formatted_rate = f"{rate:.{rate_sig_figs}f}"
        formatted_error_rate = f"{error_rate:.{error_rate_sig_figs}f}"
        formatted_string = f"{formatted_rate} ± {formatted_error_rate} {unit}"

    return formatted_string

#################################################################################################################################
##PREVENTS OVERWRITING#########################################
def naming(arquivo, folder="."):
    new_arquivo = arquivo
    if arquivo in os.listdir(folder):
        duplo = True
        vers = 2
        while duplo:
            new_arquivo = str(vers) + arquivo
            if new_arquivo in os.listdir(folder):
                vers += 1
            else:
                duplo = False
    return new_arquivo


###############################################################

# Define the linear function with two independent variables
def model(x, chi, e_vac):
    alpha_st, alpha_opt = x
    return e_vac - chi * (2 * alpha_st - alpha_opt)

# Linear fit of emission vs. epsilon (with constraints on m and n)
def linear_fit(x1, emission):
    
    #initial guess
    p0 = [0, 10]

    # Perform the fit
    coeffs, cov = curve_fit(model, x1, emission, nan_policy='omit', p0=p0)
    return coeffs, cov

def get_dielectric(films, fit, nr=1.4, num_samples=10000):
    """
    Calculate dielectric constants using coefficients from linear fit,
    propagating uncertainties via Monte Carlo simulation,
    using sampling from a multivariate Gaussian.
    
    - films: scalar or 1D array of film thicknesses (or whatever x-axis you have)
    - fit: tuple (mean_coeffs, cov_matrix) from your curve_fit
    - nr: scalar or 1D array matching films
    - num_samples: number of MC draws
    """
    mean, cov = fit

    # ensure numpy arrays and proper shapes
    films = np.atleast_1d(films)
    nr    = np.atleast_1d(nr)
    if nr.shape not in [(1,), films.shape]:
        raise ValueError("nr must be scalar or same shape as films")
    
    # compute alpha_opt per film
    alpha_opt = (nr**2 - 1) / (nr**2 + 1)

    # draw MC samples of (chi, e_vac)
    dist = np.random.multivariate_normal(mean, cov, size=num_samples)
    chi_s   = dist[:, 0]           # shape (num_samples,)
    e_vac_s = dist[:, 1]           # shape (num_samples,)

    # now compute w for each sample × each film:
    #   w_{j,i} = (e_vac_s[j] - films[i]) / (2*chi_s[j]) + alpha_opt[i] / 2
    num = e_vac_s[:, None] - films[None, :]        # (num_samples, n_films)
    den = 2 * chi_s[:, None]                       # (num_samples, 1)
    w   = num/den + alpha_opt[None, :]/2           # broadcasts scalar or per‑film alpha_opt
    w   = np.clip(w, -1, 1)

    # dielectric ε = (1 + w)/(1 - w)
    eps = (1 + w) / (1 - w)                        # (num_samples, n_films)

    # summary statistics along the sample axis
    median = np.median(eps, axis=0)
    lower  = np.percentile(eps, 15, axis=0)
    upper  = np.percentile(eps, 85, axis=0)

    # if films was scalar, return scalars
    if median.size == 1:
        return median.item(), lower.item(), upper.item()
    return median, lower, upper



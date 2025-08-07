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
def model(alpha_st, chi, e_vac):
    return e_vac - chi * (2 * alpha_st - 0.3243)

# Linear fit of emission vs. epsilon (with constraints on m and n)
def linear_fit(x1, emission):
    
    #initial guess
    p0 = [0, 10]

    # Perform the fit
    coeffs, cov = curve_fit(model, x1, emission, nan_policy='omit', p0=p0)
    return coeffs, cov

import numpy as np

def get_dielectric(films, fit, nr=1.4, num_samples=10000):
    """
    Calculate dielectric constants using coefficients from linear fit,
    propagating uncertainties via Monte Carlo simulation.
    
    - Uses sampling from a multivariate Gaussian defined by the fit.
    - Filters out unphysical dielectric values (ε < nr²).
    - Uses 68% confidence interval (16th to 84th percentiles).
    
    Parameters:
        films : float or array-like
            Input x-values (e.g., film data points) where ε is to be computed.
        fit : tuple
            Tuple of (mean, covariance matrix) from linear regression.
        nr : float
            Refractive index of the reference medium.
        num_samples : int
            Number of Monte Carlo samples.
            
    Returns:
        median_dielectric : np.ndarray
        lower : np.ndarray
        upper : np.ndarray
    """
    mean, cov = fit
    alpha_opt = (nr**2 - 1) / (nr**2 + 1)

    # Generate samples from multivariate normal distribution
    distributions = np.random.multivariate_normal(mean, cov, num_samples)

    # Compute w
    w = (distributions[:, 1][:, np.newaxis] - films) / (2 * distributions[:, 0][:, np.newaxis]) + alpha_opt / 2
    w = np.clip(w, -1, 1)  # avoid division issues at w = ±1

    # Compute dielectric samples and filter invalid values
    raw_eps = (1 + w) / (1 - w)
    dielectric_samples = np.where(raw_eps >= nr**2, raw_eps, np.nan)

    # Compute statistics ignoring invalid samples
    median_dielectric = np.nanmedian(dielectric_samples, axis=0)
    lower = np.nanpercentile(dielectric_samples, 16, axis=0)
    upper = np.nanpercentile(dielectric_samples, 84, axis=0)

    return median_dielectric[0], lower[0], upper[0]



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


def format_rate(rate, error_rate):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exp = int(np.nan_to_num(np.log10(rate)))
    if exp < -100:
        exp = -100
    if exp != 0:
        formatted_string = f"${rate/10**exp:.1f}\\pm{error_rate/10**exp:.1f}\\times10^{{{exp}}}$ $s^{{-1}}$"
    else:
        formatted_string = f"${rate/10**exp:.1f}\\pm{error_rate/10**exp:.1f}$ $s^{{-1}}$"
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
def line(x1, m, b):
    return np.abs(b) - m * (2 * x1 -0.3243)

# Linear fit of emission vs. epsilon (with constraints on m and n)
def linear_fit(x1, emission):
    
    #initial guess
    p0 = [0, 10]

    # Perform the fit
    coeffs, cov = curve_fit(line, x1, emission, nan_policy='omit', p0=p0)
    return coeffs, cov

def get_dielectric(films, fit, nr=1.4, num_samples=10000):
    """
    Calculate dielectric constants using coefficients from linear fit,
    propagating uncertainties via Monte Carlo simulation,
    using sampling from the confidence ellipse instead of a multivariate Gaussian.
    """
    mean, cov = fit

    alpha_opt = (nr**2-1) / (nr**2+1)

    # Generate samples from multivariate normal distribution
    distributions = np.random.multivariate_normal(mean, cov, num_samples)

    # Compute dielectric constant
    w =  (distributions[:, 1] - films) / (2 *distributions[:, 0] ) + alpha_opt/2
    w = np.clip(w, -1, 1)  # Ensuring w does not exceed 1
    dielectric_samples = (1 + w) / (1 - w)

    # Compute confidence intervals
    median_dielectric = np.median(dielectric_samples, axis=0)
    lower = np.percentile(dielectric_samples, 15, axis=0)
    upper = np.percentile(dielectric_samples, 85, axis=0)

    return median_dielectric, lower, upper


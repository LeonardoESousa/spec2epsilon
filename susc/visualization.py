import warnings
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import curve_fit
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

def line(x,m,b):
    return -1*m*x + b


#linear fit of emission vs. epsilon
def linear_fit(alphas,emission):
    #kwargs = {'loss': 'cauchy', 'method': 'trf'}
    #sigma = [4*i for i in alphas if np.isnan(i) == False]
    coeffs, cov = curve_fit(line, alphas, emission,nan_policy='omit')#, **kwargs) #sigma=sigma,
    return coeffs, cov

def get_dielectric(films, fit, num_samples=10000):
    """
    Calculate dielectric constants using coefficients from linear fit,
    propagating uncertainties via Monte Carlo simulation.
    """
    mean, cov = fit
    # Generate distributions for susc and vac based on the covariance matrix
    distributions = np.random.multivariate_normal(mean, cov, size=num_samples)

    w = -1*(films - distributions[:, 1]) / distributions[:, 0]
    w = np.clip(w, -1, 1)  # Ensuring w does not exceed 1
    dielectric_samples = (1 + w) / (1 - w)
    median_dielectric = np.median(dielectric_samples, axis=0)
    lower = np.percentile(dielectric_samples, 15, axis=0)
    upper = np.percentile(dielectric_samples, 85, axis=0)
    return median_dielectric, lower, upper
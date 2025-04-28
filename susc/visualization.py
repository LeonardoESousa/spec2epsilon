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
def line(vars, m, n, b):
    x1, x2 = vars
    return -m * x1 + n * x2 + b

# Linear fit of emission vs. epsilon (with constraints on m and n)
def linear_fit(x1, x2, emission):
    # Combine x1 and x2 into a tuple for curve_fit
    vars = (x1, x2)
    
    # Set bounds: m, n and b >= 0
    bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
    
    #initial guess
    p0 = [0, 10, 10]

    # Perform the fit
    coeffs, cov = curve_fit(line, vars, emission, bounds=bounds, nan_policy='omit', p0=p0)
    return coeffs, cov

def ellipses(means, cov_matrix, num_points=200):
    """
    Generates a 3D confidence ellipsoid for the given means and covariance matrix.

    Parameters:
    means (array-like): The mean values of (m, n, b).
    cov_matrix (array-like): The 3x3 covariance matrix.
    num_points (int): Number of points per axis for visualization.

    Returns:
    np.ndarray: Points forming the 3D ellipsoid (shape: (num_points**2, 3)).
    """
    # Eigenvalue decomposition
    values, vectors = np.linalg.eigh(cov_matrix)

    # Generate unit sphere
    u = np.linspace(0, 2 * np.pi, num_points)
    v = np.linspace(0, np.pi, num_points)
    x = np.outer(np.cos(u), np.sin(v)).flatten()
    y = np.outer(np.sin(u), np.sin(v)).flatten()
    z = np.outer(np.ones_like(u), np.cos(v)).flatten()

    # Stack into unit sphere points
    unit_sphere = np.column_stack((x, y, z))

    # Scale and transform unit sphere using covariance matrix
    scaling_factors = np.sqrt(values)  # Scale by standard deviation
    ellipsoid_points = unit_sphere @ np.diag(scaling_factors) @ vectors.T

    # Shift to mean position
    ellipsoid_points += means

    # Ensure m, n, b remain positive
    #ellipsoid_points = np.clip(ellipsoid_points, 0, None)

    return ellipsoid_points


def sample_from_ellipse(mean, cov, num_samples=10000, confidence_level=1.0):
    """
    Sample points from the confidence ellipse defined by the mean and covariance matrix.

    Parameters:
    mean (array-like): The mean values of (m, n, b).
    cov (array-like): The 3x3 covariance matrix.
    num_samples (int): Number of samples to generate.
    confidence_level (float): Confidence level scaling (e.g., 1σ, 2σ).

    Returns:
    np.ndarray: Samples (num_samples, 3) from the confidence ellipse.
    """
    # Eigenvalue decomposition of covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Scale eigenvalues for the confidence level (default = 1σ)
    scaling_factors = np.sqrt(eigenvalues) * confidence_level
    ellipse_transform = eigenvectors @ np.diag(scaling_factors)

    # Sample uniformly within the unit sphere
    unit_samples = np.random.randn(num_samples, 3)  # Standard normal
    unit_samples /= np.linalg.norm(unit_samples, axis=1, keepdims=True)  # Normalize to unit sphere

    # Apply transformation to match the confidence ellipse
    samples = mean + unit_samples @ ellipse_transform.T

    # Keep only valid samples where m > 0
    valid_samples = samples[samples[:, 0] > 0]

    return valid_samples

def get_dielectric(films, fit, num_samples=10000):
    """
    Calculate dielectric constants using coefficients from linear fit,
    propagating uncertainties via Monte Carlo simulation,
    using sampling from the confidence ellipse instead of a multivariate Gaussian.
    """
    nr = 1.4
    alpha = (nr**2 - 1) / (nr**2 + 1)
    mean, cov = fit

    # Generate samples from the confidence ellipse
    distributions = sample_from_ellipse(mean, cov, num_samples)

    # Compute dielectric constant
    w = -1 * (films - distributions[:, 2] - distributions[:, 1] * alpha) / distributions[:, 0]
    w = np.clip(w, -1, 1)  # Ensuring w does not exceed 1
    dielectric_samples = (1 + w) / (1 - w)

    # Compute confidence intervals
    median_dielectric = np.median(dielectric_samples, axis=0)
    lower = np.percentile(dielectric_samples, 15, axis=0)
    upper = np.percentile(dielectric_samples, 85, axis=0)

    return median_dielectric, lower, upper


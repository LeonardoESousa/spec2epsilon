import warnings
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.stats import chi2
# pylint: disable=unbalanced-tuple-unpacking

THECOLOR = "black"
cmap = plt.get_cmap("cividis")





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


###############################################################

def load_data(file):
    data = pd.read_csv(file, skipinitialspace=True)  # trims spaces right after commas
    data.columns = data.columns.str.strip()
    cols = {c: c for c in data.columns}
    if "solvent" in data.columns and "Solvent" not in data.columns:
        cols["solvent"] = "Solvent"
    data.rename(columns=cols, inplace=True)
    # remove whitespace from every string cell
    data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    data['epsilon'] = pd.to_numeric(data['epsilon'], errors='coerce')
    data['nr'] = pd.to_numeric(data['nr'], errors='coerce')
    # all numerical values above 100 are considered have to be converted to eV
    for col in data.columns:
        if col.lower() not in ['solvent', 'epsilon', 'nr']:
            data[col] = data[col].apply(lambda x: 1240/float(x) if float(x) > 100 else float(x))
    return data

# Define the linear function with two independent variables
def model(x, chi, e_vac):
    alpha_st, alpha_opt = x
    return e_vac - chi * (2 * alpha_st - alpha_opt)

# Linear fit of emission vs. epsilon (with constraints on m and n)
def characterize(x1, emission):
    
    #initial guess
    p0 = [0, 10]

    # Perform the fit
    coeffs, cov = curve_fit(model, x1, emission, nan_policy='omit', p0=p0)
    return coeffs, cov



def compute_dielectric(films, fit, nr=1.4, num_samples=10000):
    """
    Calculate dielectric constants using coefficients from linear fit,
    propagating uncertainties via Monte Carlo simulation.
    
    - Filters out unphysical dielectric values (ε < nr²).
    - Optionally caps very large ε to epsilon_max.
    - Supports scalar or array inputs for films and nr.
    - Uses 68% confidence interval (16th to 84th percentiles).

    Parameters:
        films : float or array-like
            Emission energies (or x-values) where ε is to be computed.
        fit : tuple
            Tuple of (mean, covariance matrix) from a linear fit.
        nr : float or array-like
            Refractive index(s) at each film point.
        num_samples : int
            Number of Monte Carlo samples.
        
    Returns:
        median : np.ndarray or float
        lower  : np.ndarray or float
        upper  : np.ndarray or float
    """
    mean, cov = fit

    # Ensure input shapes
    films = np.atleast_1d(films)
    nr    = np.atleast_1d(nr)
    if nr.shape not in [(1,), films.shape]:
        raise ValueError("nr must be scalar or same shape as films")

    # Compute alpha_opt
    alpha_opt = (nr**2 - 1) / (nr**2 + 1)

    # Monte Carlo samples
    dist = np.random.multivariate_normal(mean, cov, size=num_samples)
    chi_s   = dist[:, 0]  # shape (num_samples,)
    e_vac_s = dist[:, 1]  # shape (num_samples,)

    # Compute w
    num = e_vac_s[:, None] - films[None, :]  # shape (samples, films)
    den = 2 * chi_s[:, None]                 # shape (samples, 1)
    w   = num / den + alpha_opt[None, :] / 2
    # Enforce physical constraints (w must be in (alpha_opt, 1) )
    w = np.where(w < alpha_opt[None, :], np.nan, w)
    w = np.where(w > 1, np.nan, w)

    # Compute ε
    eps = (1 + w) / (1 - w)
    
    # Compute statistics (ignoring NaN)
    median = np.nanmedian(eps, axis=0)
    lower  = np.nanpercentile(eps, 16, axis=0)
    upper  = np.nanpercentile(eps, 84, axis=0)

    # Return scalars if input was scalar
    if median.size == 1:
        return median.item(), lower.item(), upper.item()
    return median, lower, upper



def dielectric(data, film, molecule, opt, cov):
    # filter data for the specific film and molecule
    inference = data[data['epsilon'].isna()]
    emi = inference[inference['Solvent'] == film][molecule].to_numpy()
    nrs = inference[inference['Solvent'] == film]['nr'].to_numpy()
    median, lower, upper = compute_dielectric(emi, (opt,cov), nr=nrs)
    return median, lower, upper


def plot_confidence_ellipse(fit, ax, confidence=0.68, num_points=200, **kwargs):
    """
    Plot a confidence ellipse using a scatter plot, based on (mean, cov).

    Parameters:
        fit : tuple
            (mean, covariance matrix), with 2D mean and 2x2 cov matrix.
        ax : matplotlib.axes.Axes
            Axis object to draw the ellipse in.
        confidence : float
            Confidence level (default: 0.68 for 1σ).
        num_points : int
            Number of points to sample around the ellipse.
        **kwargs :
            Additional keyword arguments passed to ax.plot (e.g., color, linestyle).
    
    Returns:
        Line2D object from ax.plot
    """
    mean, cov = fit
    if len(mean) != 2 or cov.shape != (2, 2):
        raise ValueError("fit must contain a 2D mean and a 2x2 covariance matrix")

    # Radius of ellipse for given confidence level
    chi2_val = chi2.ppf(confidence, df=2)
    radius = np.sqrt(chi2_val)

    # Parametric angles
    theta = np.linspace(0, 2 * np.pi, num_points)

    # Unit circle
    circle = np.stack([np.cos(theta), np.sin(theta)])  # shape: (2, num_points)

    # Transform circle using Cholesky decomposition
    ellipse = mean[:, None] + radius * np.linalg.cholesky(cov) @ circle

    # Plot as line
    return ax.plot(ellipse[0], ellipse[1], **kwargs)

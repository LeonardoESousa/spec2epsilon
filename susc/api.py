# susc/api.py
import numpy as np
import pandas as pd

from .visualization import load_data, characterize, dielectric


def analysis(
    file,
    epsilon_col="epsilon",
    nr_col="nr",
):
    """
    End-to-end (no plotting):
      - Identify all molecule columns (every column except solvent/epsilon/nr)
      - Fit χ & E_vac per molecule using rows with known epsilon & nr
      - Find all solvents whose epsilon is missing (NaN) -> those need epsilon
      - For each (molecule, solvent_needing_epsilon), estimate epsilon interval

    Returns
    -------
    summary : pandas.DataFrame
        Columns: ["solvent", "molecule", "epsilon_median", "epsilon_lower", "epsilon_upper"].
    fits : pandas.DataFrame
        Index: molecule. Columns: ["E_vac", "E_vac_err", "chi", "chi_err"].
    plot_data : pandas.DataFrame
        Long-form data to let users plot emission vs (2*alpha_st - alpha_opt).
        Columns: ["molecule", "solvent", "x", "emission"].
    """
    df = load_data(file).copy()

    # Handle solvent column name being either 'Solvent' or 'solvent'
    if "solvent" in df.columns:
        solvent_col = "solvent"
    elif "Solvent" in df.columns:
        solvent_col = "Solvent"
    else:
        raise ValueError("No solvent column found (expected 'solvent' or 'Solvent').")

    required = {epsilon_col, nr_col, solvent_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Missing required columns: %s" % (sorted(missing),))

    # Identify molecule columns
    molecules = [c for c in df.columns if c not in (epsilon_col, nr_col, solvent_col)]

    # Solvents needing epsilon (epsilon is NaN)
    need_eps = df.loc[df[epsilon_col].isna(), solvent_col].dropna().unique().tolist()

    summary_rows = []
    fits_rows = []
    plot_rows = []

    # Precompute arrays for regressors (known points only)
    eps_all = df[epsilon_col].to_numpy()
    nr_all = df[nr_col].to_numpy()
    mask_known = (~np.isnan(eps_all)) & (~np.isnan(nr_all))

    if mask_known.sum() < 2:
        raise ValueError("Not enough rows with known epsilon & nr to perform fits.")

    eps_known = eps_all[mask_known]
    nr_known = nr_all[mask_known]
    alphas_st_known = (eps_known - 1) / (eps_known + 1)
    alphas_opt_known = (nr_known**2 - 1) / (nr_known**2 + 1)
    x_known = 2 * alphas_st_known - alphas_opt_known

    # Also keep the solvent labels for the known rows
    solvents_known = df.loc[mask_known, solvent_col].to_numpy()

    for mol in molecules:
        # Use only rows where emission for this molecule is present + regressors known
        y_col = df[mol].to_numpy()
        mask = mask_known & (~df[mol].isna().to_numpy())
        if mask.sum() < 2:
            # Not enough points to fit this molecule; skip gracefully
            continue

        y_fit = y_col[mask]

        # Restrict precomputed arrays to this molecule's usable rows
        idx = mask[mask_known]  # boolean mask over the known subset
        x_mol = x_known[idx]
        alphas_st_m = alphas_st_known[idx]
        alphas_opt_m = alphas_opt_known[idx]
        solvents_m = solvents_known[idx]

        # Fit for this molecule
        opt, cov = characterize((alphas_st_m, alphas_opt_m), y_fit)
        chi, e_vac = opt
        err = np.sqrt(np.diag(cov))
        fits_rows.append({
            "molecule": mol,
            "E_vac": float(e_vac),
            "E_vac_err": float(err[1]),
            "chi": float(chi),
            "chi_err": float(err[0]),
        })

        # Collect plotting rows: emission vs x for this molecule
        for s, x_val, y_val in zip(solvents_m, x_mol, y_fit):
            plot_rows.append({
                "molecule": mol,
                "solvent": s,
                "x": float(x_val),
                "emission": float(y_val),
            })

        # For every solvent needing epsilon, estimate ε for this molecule
        for solv in need_eps:
            try:
                median, lower, upper = dielectric(df, solv, mol, (chi, e_vac), cov)
                summary_rows.append({
                    "solvent": solv,
                    "molecule": mol,
                    "epsilon_median": float(median),
                    "epsilon_lower": float(lower),
                    "epsilon_upper": float(upper),
                })
            except Exception:
                # On failure, include row with NaNs for epsilon values
                summary_rows.append({
                    "solvent": solv,
                    "molecule": mol,
                    "epsilon_median": np.nan,
                    "epsilon_lower": np.nan,
                    "epsilon_upper": np.nan,
                })

    summary = (
        pd.DataFrame(summary_rows)
        .sort_values(["epsilon_median", "solvent", "molecule"], na_position="last")
        .reset_index(drop=True)
    )
    fits_df = pd.DataFrame(fits_rows).set_index("molecule")[["E_vac", "E_vac_err", "chi", "chi_err"]]
    plot_data = pd.DataFrame(plot_rows).sort_values(["molecule", "solvent"]).reset_index(drop=True)
    return summary, fits_df, plot_data

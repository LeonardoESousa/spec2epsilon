# susc/api.py
import io
import re
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .visualization import load_data, characterize, dielectric, confidence_ellipse


def _uniquify_name(base_name, used_names):
    if base_name not in used_names:
        used_names.add(base_name)
        return base_name

    idx = 2
    while True:
        candidate = f"{base_name}_{idx}"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        idx += 1


def _build_fit_export_payload(fit_entries):
    export_payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "molecules": [],
    }

    for row in fit_entries:
        export_payload["molecules"].append({
            "molecule": row["molecule"],
            "E_vac": float(row["E_vac"]),
            "chi": float(row["chi"]),
            "covariance_matrix": row["covariance_matrix"],
        })

    return export_payload


def _build_fit_zip_bytes(export_payload):
    zip_buffer = io.BytesIO()
    used_file_stems = set()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fit_data in export_payload["molecules"]:
            molecule = fit_data["molecule"]
            # Keep molecule names readable while avoiding filesystem-invalid characters.
            safe_base = re.sub(r"[^A-Za-z0-9._-]+", "_", molecule).strip("._") or "molecule"
            safe_name = _uniquify_name(safe_base, used_file_stems)
            mol_payload = {
                "schema_version": export_payload["schema_version"],
                "generated_utc": export_payload["generated_utc"],
                "molecule": molecule,
                "E_vac": fit_data["E_vac"],
                "chi": fit_data["chi"],
                "covariance_matrix": fit_data["covariance_matrix"],
            }
            mol_buffer = io.BytesIO()
            np.save(mol_buffer, mol_payload, allow_pickle=True)
            mol_buffer.seek(0)
            zf.writestr(f"{safe_name}.npy", mol_buffer.getvalue())

    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def analysis(
    file,
    epsilon_col="epsilon",
    nr_col="nr",
    ellipse=False,
    ignore_list=None,
    download=False,
):
    """
    End-to-end (no plotting):
      - Identify all molecule columns (every column except solvent/epsilon/nr)
      - Fit χ & E_vac per molecule using rows with known epsilon & nr
      - Find all solvents whose epsilon is missing (NaN) -> those need epsilon
      - For each (molecule, solvent_needing_epsilon), estimate epsilon interval

    Parameters
    ----------
    download : bool, default False
        If True, writes `spec2epsilon_fit_results.zip` in the current working directory
        with one `.npy` file per fitted molecule.

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
    if ignore_list is None:
        ignore_list = []

    # Handle solvent column name being either 'Solvent' or 'solvent'
    if "solvent" in df.columns:
        solvent_col = "solvent"
    elif "Solvent" in df.columns:
        solvent_col = "Solvent"
    else:
        raise ValueError("No solvent column found (expected 'solvent' or 'Solvent').")
    # Remove ignored solvents
    for ignore in ignore_list:
        df = df[~df[solvent_col].str.lower().str.contains(ignore.lower(), na=False)]

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
    ellipse_curves = {}

    # Precompute arrays for regressors (known points only)
    eps_all = df[epsilon_col].to_numpy()
    nr_all = df[nr_col].to_numpy()
    mask_known = (~np.isnan(eps_all)) & (~np.isnan(nr_all))

    if mask_known.sum() <= 2:
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

        if ellipse:
            #confidence ellipse
            ellipse_curve = confidence_ellipse((opt, cov), confidence=0.68, num_points=200)
            ellipse_curves[mol] = ellipse_curve

        # get R²
        y_mean = np.mean(y_fit)
        ss_tot = np.sum((y_fit - y_mean) ** 2)
        y_pred = e_vac - chi * x_mol
        residuals = (y_fit - y_pred)
        ss_res = np.sum(residuals ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        fits_rows.append({
            "molecule": mol,
            "E_vac": float(e_vac),
            "E_vac_err": float(err[1]),
            "chi": float(chi),
            "chi_err": float(err[0]),
            "R2": float(r_squared),
            "covariance_matrix": cov.tolist() if cov is not None else None,
        })

        # Collect plotting rows: emission vs x for this molecule
        for s, x_val, y_val, res_val in zip(solvents_m, x_mol, y_fit, residuals):
            plot_rows.append({
                "molecule": mol,
                "solvent": s,
                "x": float(x_val),
                "emission": float(y_val),
                "residual": float(res_val),
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
    if len(need_eps) != 0:
        summary = (
            pd.DataFrame(summary_rows)
            .sort_values(["epsilon_median", "solvent", "molecule"], na_position="last")
            .reset_index(drop=True)
        )
    else:
        summary = pd.DataFrame(
            columns=["solvent", "molecule", "epsilon_median", "epsilon_lower", "epsilon_upper"]
        )
    fits_df = pd.DataFrame(fits_rows)[["molecule","E_vac", "E_vac_err", "chi", "chi_err", "R2"]]
    plot_data = pd.DataFrame(plot_rows).sort_values(["molecule", "x"]).reset_index(drop=True)

    if download and not fits_df.empty:
        export_payload = _build_fit_export_payload(fits_rows)
        zip_bytes = _build_fit_zip_bytes(export_payload)
        with open("spec2epsilon_fit_results.zip", "wb") as f:
            f.write(zip_bytes)

    if ellipse:
        return summary, fits_df, plot_data, ellipse_curves
    else:
        return summary, fits_df, plot_data
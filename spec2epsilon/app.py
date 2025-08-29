
# app.py
# ---
# A Streamlit app.
# Functionality:
# - Upload one or more CSV "ensemble" files
# - Choose which solvents to include per molecule
# - Fit and visualize correlations using functions from spec2epsilon.visualization
# - Show residuals, properties table, and (if epsilon is missing) inferred dielectric tables
# - Download the current plot as a high-resolution PNG
#
# NOTES
# - This app expects the package `spec2epsilon` to be importable and to provide:
#     visualization.load_data, visualization.characterize, visualization.model,
#     visualization.format_number, visualization.compute_dielectric, visualization.set_fontsize (optional)
# - If "dashstyle.mplstyle" exists in the working directory, it will be applied to matplotlib.
# - If "./figs/favicon.ico" exists, it will be used as page icon.
#
# Run locally:
#   pip install streamlit matplotlib numpy pandas
#   # plus your package that provides `spec2epsilon`
#   streamlit run app.py
#
# File format expectations:
#   Each CSV should include columns:
#     - 'Solvent' (or 'solvent')
#     - 'epsilon' (may contain NaNs for inference mode)
#     - 'nr'      (refractive index)
#     - one or more molecule columns with emission energies (in eV) or wavelengths (>100 assumed nm, auto-converted to eV)
# ---

import io
import os
import math
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.options.mode.chained_assignment = None

# --- Try to import the visualization helpers ---
viz_err = None
try:
    from spec2epsilon import visualization
except Exception as e:
    viz_err = e
    visualization = None

# --- Page config ---
def _resolve_icon():
    # Prefer local favicon if available; fallback to emoji
    candidates = ["./figs/favicon.ico", "figs/favicon.ico"]
    for path in candidates:
        if os.path.exists(path):
            return path
    return "🧪"

st.set_page_config(
    page_title="spec2epsilon",
    page_icon=_resolve_icon(),
    layout="wide"
)

st.markdown("<h1 style='margin-bottom:0'>spec2epsilon</h1>", unsafe_allow_html=True)
st.caption("Estimate solvent dielectric constants from fluorescence spectra")

# --- Apply custom matplotlib style if present ---
mplstyle = os.path.join(os.getcwd(), "dashstyle.mplstyle")
if os.path.exists(mplstyle):
    try:
        plt.style.use([mplstyle])
    except Exception:
        pass

# --- Sidebar: About ---
with st.sidebar:
    st.subheader("How to use")
    st.write(
    "- Upload one or more CSV files in the following format.\n"
    "- Pick solvents per molecule (defaults: all).\n"
    "- Columns expected in each CSV: `Solvent/solvent`, `epsilon`, `nr`, and one or more molecule columns with emission energies in eV or nm.\n"
    "- Empty entries in the epsilon column will be treated values to be inferred."
    )
    
    st.markdown("**Example CSV format:**")
    st.code(
        """Solvent,epsilon,Mol1,Mol2,Mol3
Hexane,2.0165,389,395,387
14-dioxane,2.2099,422,445,435
Toluene,2.38,416,434,425
PhCl,5.6968,425,465,448
EtOAc,6.253,435,476,460
Film1,1.6000,430,470,452
Film2,1.6000,,440,460,465
""",
        language="csv"
    )

    
    
if viz_err is not None:
    st.warning(
        "Could not import `spec2epsilon.visualization`. "
        f"Install it or ensure it's on PYTHONPATH.\n\nError: `{viz_err}`"
    )

# --- Helpers ---
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c: c for c in df.columns}
    # Unify "solvent" -> "Solvent"
    if "solvent" in df.columns and "Solvent" not in df.columns:
        cols["solvent"] = "Solvent"
    df = df.rename(columns=cols)
    return df

def _load_csv_files(uploaded_files) -> List[pd.DataFrame]:
    """Use visualization.load_data."""
    datas = []
    for uf in uploaded_files:
        name = uf.name
        raw = uf.getvalue()
        bio = io.BytesIO(raw)
        data = visualization.load_data(bio)
        data = _normalize_columns(data)

        # Convert emissions in nm (>100) to eV (1240/nm) for molecule columns
        # Keep 'Solvent', 'epsilon', 'nr' unchanged.
        for col in data.columns:
            if col not in ["Solvent", "epsilon", "nr"]:
                # Coerce to float
                def _to_ev(x):
                    try:
                        val = float(x)
                        return 1240.0 / val if val > 100 else val
                    except Exception:
                        return np.nan
                data[col] = data[col].apply(_to_ev)
        data.name = os.path.splitext(os.path.basename(name))[0]
        datas.append(data)
    return datas

def _collect_molecules(datas: List[pd.DataFrame]) -> List[str]:
    molecules = []
    for df in datas:
        molecules.extend([c for c in df.columns if c not in ["Solvent", "epsilon", "nr", "solvent"]])
    # Keep order but unique
    seen = set()
    uniq = []
    for m in molecules:
        if m not in seen:
            seen.add(m)
            uniq.append(m)
    return uniq

def _collect_solvents_for_molecule(datas: List[pd.DataFrame], molecule: str) -> List[str]:
    sv = []
    for df in datas:
        if "Solvent" in df.columns and molecule in df.columns:
            sv.extend(df["Solvent"].dropna().astype(str).unique().tolist())
    # unique preserve order
    seen = set()
    uniq = []
    for s in sv:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq

# --- Sidebar: File upload and controls ---
uploaded = st.file_uploader(
    "Upload one or more .csv files",
    type=["csv"],
    accept_multiple_files=True,
    help="CSV files with columns: Solvent/solvent, epsilon, nr, and molecule name"
)

if not uploaded:
    st.info("Upload CSV files to begin.")
    st.stop()

datas = _load_csv_files(uploaded)
if len(datas) == 0:
    st.error("No data could be loaded from the uploaded files.")
    st.stop()

# Determine molecules across all data
all_molecules = _collect_molecules(datas)
if len(all_molecules) == 0:
    st.error("No molecule columns found. Ensure your CSVs include one or more molecule emission columns.")
    st.stop()

# Configure solvent selections per molecule
st.subheader("Selections")
with st.expander("Solvents", expanded=True):
    selections: Dict[str, List[str]] = {}
    ncols = min(3, max(1, len(all_molecules)))  # up to 3 columns for controls
    # chunk molecules to distribute in columns
    chunks = [all_molecules[i::ncols] for i in range(ncols)]
    cols = st.columns(ncols)
    for col, mols in zip(cols, chunks):
        with col:
            for mol in mols:
                options = _collect_solvents_for_molecule(datas, mol)
                default = options[:]  # all by default
                selections[mol] = st.multiselect(
                    f"{mol}",
                    options=options,
                    default=default,
                    key=f"solv_{mol}"
                )

# --- Analysis and plotting ---
st.subheader("Characterization")
fig, ax = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
ax0, ax1 = ax

fits: Dict[str, Tuple[Tuple[float, float], np.ndarray]] = {}
stats_rows = []  # [molecule, e_vac_fmt, chi_fmt]
inference_tables: Dict[str, pd.DataFrame] = {}

# Plot each molecule across each dataset
for df in datas:
    # Pre-check for core columns
    if not set(["Solvent", "epsilon", "nr"]).issubset(df.columns):
        st.warning(f"File `{getattr(df, 'name', 'unknown')}` is missing one of the required columns: Solvent, epsilon, nr. Skipping.")
        continue

    for molecule in [c for c in df.columns if c not in ["Solvent", "epsilon", "nr", "solvent"]]:
        # Respect user's solvent selections for this molecule
        allowed_solvents = selections.get(molecule, [])
        if len(allowed_solvents) == 0:
            continue

        data_mol = df[df["Solvent"].astype(str).isin(allowed_solvents)].copy()

        # Skip if insufficient valid data
        if data_mol.empty:
            continue

        epsilons = data_mol["epsilon"].to_numpy(dtype=float)
        nr = data_mol["nr"].to_numpy(dtype=float)
        emission = data_mol[molecule].to_numpy(dtype=float)

        # Only fit when we have at least 3 points and all needed columns are finite
        mask = np.isfinite(epsilons) & np.isfinite(nr) & np.isfinite(emission)
        if mask.sum() < 3:
            continue

        alphas_st = (epsilons[mask] - 1.0) / (epsilons[mask] + 1.0)
        alphas_opt = (nr[mask] ** 2 - 1.0) / (nr[mask] ** 2 + 1.0)
        emission_fit = emission[mask]

        opt, cov = visualization.characterize((alphas_st, alphas_opt), emission_fit)
        chi, e_vac = opt  # follows notebook convention
        fits[molecule] = (opt, cov)

        # Model and plotting
        function = visualization.model((alphas_st, alphas_opt), chi, e_vac)
        
        x = 2 * alphas_st - alphas_opt
        ax0.plot(x, function, label=molecule)
        ax0.scatter(x, emission_fit, s=20)

        # Residuals
        ax1.scatter(x, emission_fit - function, marker="x", s=60, linewidths=1.5)

        # Stats row with uncertainties
        error = np.sqrt(np.diag(cov)) if cov is not None else np.array([np.nan, np.nan])
        if hasattr(visualization, "format_number"):
            chi_fmt = visualization.format_number(chi, error[0], "")
            e_vac_fmt = visualization.format_number(e_vac, error[1], "")
        else:
            chi_fmt = f"{chi:.3f} ± {error[0]:.3f}" if np.isfinite(error[0]) else f"{chi:.3f}"
            e_vac_fmt = f"{e_vac:.3f} ± {error[1]:.3f}" if np.isfinite(error[1]) else f"{e_vac:.3f}"
        stats_rows.append([molecule, e_vac_fmt, chi_fmt])

    # Inference block (epsilon NaNs)
    if df["epsilon"].isna().any() and len(fits) > 0 and hasattr(visualization, "compute_dielectric"):
        inference = df[df["epsilon"].isna()].copy()
        if not inference.empty:
            for molecule in [c for c in df.columns if c not in ["Solvent", "epsilon", "nr", "solvent"]]:
                if molecule not in fits:
                    # need a prior fit for this molecule; skip if not yet fitted
                    continue
                rows = []
                for film in inference["Solvent"].dropna().astype(str).unique().tolist():
                    sub = inference[inference["Solvent"].astype(str) == film]
                    emi = sub[molecule].to_numpy(dtype=float)
                    nrs = sub["nr"].to_numpy(dtype=float)

                    if len(emi) == 0 or not np.isfinite(emi[0]):
                        continue

                    median, lower, upper = visualization.compute_dielectric(emi, fits[molecule], nr=nrs)
                    rows.append([
                        film,
                        emi[0],
                        f"{1240.0/emi[0]:.0f}" if emi[0] != 0 else "∞",
                        median,
                        f"[{lower:.2f} , {upper:.2f}]"
                    ])
                if rows:
                    df_inf = pd.DataFrame(rows, columns=["Film", "Emission (eV)", "Emission (nm)", "ε", "Interval"])
                    df_inf = df_inf.sort_values(by="ε", ascending=True, kind="mergesort")
                    #format Emission
                    df_inf["Emission (eV)"] = df_inf["Emission (eV)"].apply(lambda x: f"{x:.2f}")
                    df_inf["ε"] = df_inf["ε"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "∞")
                    inference_tables[molecule] = df_inf

# Finalize axes
ax0.set_xlabel(r"$2\alpha_{st} - \alpha_{opt}$")
ax0.set_ylabel("Energy (eV)")
ax0.set_title("a)", loc="left")
# add legend only if items present
if len(ax0.get_lines()) > 0:
    ax0.legend(loc="best")

ax1.set_xlabel(r"$2\alpha_{st} - \alpha_{opt}$")
ax1.set_ylabel("Residuals (eV)")
ax1.set_title("b)", loc="left")

st.pyplot(fig, width='stretch')

# Download figure
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=600)
st.download_button(
    "Download PNG (600 dpi)",
    data=buf.getvalue(),
    file_name="susceptibilities.png",
    mime="image/png",
    help="Save the current plot as a high-resolution PNG"
)


# Properties (stats) table
if stats_rows:
    stats_df = pd.DataFrame(stats_rows, columns=["Molecule", "<E_vac> (eV)", "<χ> (eV)"])
    stats_df = stats_df.sort_values(by="<χ> (eV)", key=lambda s: s.astype(str), ascending=False, kind="mergesort")
    #remove index
    stats_df = stats_df.reset_index(drop=True)
    st.dataframe(stats_df, width='stretch')
else:
    st.info("No stats to display yet (need ≥3 valid points per molecule to fit).")

# Inference tables: display side-by-side
if len(inference_tables) > 0:
    st.subheader("**Inferred ε**")
    num = len(inference_tables)
    max_cols = min(3, num)
    # chunk into rows of up to 3 columns
    mol_keys = list(inference_tables.keys())
    for i in range(0, num, max_cols):
        cols = st.columns(min(max_cols, num - i))
        for col, mk in zip(cols, mol_keys[i:i+max_cols]):
            with col:
                st.caption(mk)
                st.dataframe(inference_tables[mk], width='stretch')
else:
    st.caption("No ε inference performed (no rows with missing `epsilon`).")

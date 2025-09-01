# app.py
# ---
# A Streamlit app with Plotly + MathJax enabled globally.

import io
import os
import math
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# Plotly
import plotly.graph_objects as go
import plotly.express as px

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

js_path = os.path.join(os.path.dirname(__file__), "load-mathjax.js")
with open(js_path, "r") as f:
    js = f.read()
    st.components.v1.html(f"<script>{js}</script>", height=0)

st.markdown("<h1 style='margin-bottom:0'>spec2epsilon</h1>", unsafe_allow_html=True)
st.caption("Estimate solvent dielectric constants from fluorescence spectra")

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
    if "solvent" in df.columns and "Solvent" not in df.columns:
        cols["solvent"] = "Solvent"
    df = df.rename(columns=cols)
    return df

def _load_csv_files(uploaded_files) -> List[pd.DataFrame]:
    datas = []
    for uf in uploaded_files:
        name = uf.name
        raw = uf.getvalue()
        bio = io.BytesIO(raw)
        data = visualization.load_data(bio)
        data = _normalize_columns(data)
        for col in data.columns:
            if col not in ["Solvent", "epsilon", "nr"]:
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

all_molecules = _collect_molecules(datas)
if len(all_molecules) == 0:
    st.error("No molecule columns found.")
    st.stop()

# Configure solvent selections per molecule
st.subheader("Selections")
with st.expander("Solvents", expanded=True):
    selections: Dict[str, List[str]] = {}
    ncols = min(3, max(1, len(all_molecules)))
    chunks = [all_molecules[i::ncols] for i in range(ncols)]
    cols = st.columns(ncols)
    for col, mols in zip(cols, chunks):
        with col:
            for mol in mols:
                options = _collect_solvents_for_molecule(datas, mol)
                default = options[:]
                selections[mol] = st.multiselect(
                    f"{mol}",
                    options=options,
                    default=default,
                    key=f"solv_{mol}"
                )

# --- Analysis and plotting ---
st.subheader("Characterization")

fits: Dict[str, Tuple[Tuple[float, float], np.ndarray]] = {}
stats_rows = []
inference_tables: Dict[str, pd.DataFrame] = {}

palette = px.colors.qualitative.Plotly
if len(all_molecules) > len(palette):
    extra = px.colors.qualitative.Safe + px.colors.qualitative.Vivid + px.colors.qualitative.Set3
    palette = (palette + extra) * ((len(all_molecules) // len(palette)) + 1)
color_map: Dict[str, str] = {m: palette[i] for i, m in enumerate(all_molecules)}

fig_corr = go.Figure()  
fig_res = go.Figure()   

for df in datas:
    if not set(["Solvent", "epsilon", "nr"]).issubset(df.columns):
        st.warning(f"File `{getattr(df, 'name', 'unknown')}` is missing required columns. Skipping.")
        continue

    for molecule in [c for c in df.columns if c not in ["Solvent", "epsilon", "nr", "solvent"]]:
        allowed_solvents = selections.get(molecule, [])
        if len(allowed_solvents) == 0:
            continue

        data_mol = df[df["Solvent"].astype(str).isin(allowed_solvents)].copy()
        if data_mol.empty:
            continue

        epsilons = data_mol["epsilon"].to_numpy(dtype=float)
        nr = data_mol["nr"].to_numpy(dtype=float)
        emission = data_mol[molecule].to_numpy(dtype=float)

        mask = np.isfinite(epsilons) & np.isfinite(nr) & np.isfinite(emission)
        if mask.sum() < 3:
            continue

        alphas_st = (epsilons[mask] - 1.0) / (epsilons[mask] + 1.0)
        alphas_opt = (nr[mask] ** 2 - 1.0) / (nr[mask] ** 2 + 1.0)
        emission_fit = emission[mask]

        opt, cov = visualization.characterize((alphas_st, alphas_opt), emission_fit)
        chi, e_vac = opt
        fits[molecule] = (opt, cov)

        function = visualization.model((alphas_st, alphas_opt), chi, e_vac)
        x = 2 * alphas_st - alphas_opt

        color = color_map[molecule]

        solvents = data_mol["Solvent"].to_numpy()[mask]

        fig_corr.add_trace(go.Scatter(
            x=x, y=function,
            mode="lines",
            name=molecule,
            legendgroup=molecule,
            line=dict(color=color, width=2),
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "Solvent=%{customdata}<br>"
                "x=%{x:.3f}<br>Model (eV)=%{y:.3f}<extra></extra>"
            ),
            customdata=solvents
        ))
        fig_corr.add_trace(go.Scatter(
            x=x, y=emission_fit,
            mode="markers",
            name=molecule + " (obs)",
            legendgroup=molecule,
            showlegend=False,
            marker=dict(color=color, size=7, line=dict(color=color, width=0.5)),
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "Solvent=%{customdata}<br>"
                "x=%{x:.3f}<br>Emission =%{y:.3f} eV<extra></extra>"
            ),
            customdata=solvents
        ))

        residuals = emission_fit - function
        fig_res.add_trace(go.Scatter(
            x=x, y=residuals,
            mode="markers",
            name=molecule,
            legendgroup=molecule,
            showlegend=True,
            marker=dict(color=color, size=9),
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "Solvent=%{customdata}<br>"
                "x=%{x:.3f}<br>Residual (eV)=%{y:.3f}<extra></extra>"
            ),
            customdata=solvents
        ))

        error = np.sqrt(np.diag(cov)) if cov is not None else np.array([np.nan, np.nan])
        if hasattr(visualization, "format_number"):
            chi_fmt = visualization.format_number(chi, error[0], "")
            e_vac_fmt = visualization.format_number(e_vac, error[1], "")
        else:
            chi_fmt = f"{chi:.3f} ± {error[0]:.3f}" if np.isfinite(error[0]) else f"{chi:.3f}"
            e_vac_fmt = f"{e_vac:.3f} ± {error[1]:.3f}" if np.isfinite(error[1]) else f"{e_vac:.3f}"
        stats_rows.append([molecule, e_vac_fmt, chi_fmt])

    if df["epsilon"].isna().any() and len(fits) > 0 and hasattr(visualization, "compute_dielectric"):
        inference = df[df["epsilon"].isna()].copy()
        if not inference.empty:
            for molecule in [c for c in df.columns if c not in ["Solvent", "epsilon", "nr", "solvent"]]:
                if molecule not in fits:
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
                    df_inf["Emission (eV)"] = df_inf["Emission (eV)"].apply(lambda x: f"{x:.2f}")
                    df_inf["ε"] = df_inf["ε"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "∞")
                    inference_tables[molecule] = df_inf

if len(fig_corr.data) > 0:
    fig_corr.update_layout(xaxis_title=r"$2 \alpha_{st} - \alpha_{opt}$", yaxis_title="Energy (eV)",legend=dict(
        font=dict(size=16)), margin=dict(l=20, r=20, t=20, b=20)
    )
    # Enforce font sizes even if something overrides template
    fig_corr.update_xaxes(title_font=dict(size=20), tickfont=dict(size=14))
    fig_corr.update_yaxes(title_font=dict(size=20), tickfont=dict(size=14))
if len(fig_res.data) > 0:
    fig_res.update_layout(xaxis_title=r"$2 \alpha_{st} - \alpha_{opt}$", yaxis_title="Residuals (eV)", legend=dict(
        font=dict(size=16)), margin=dict(l=20, r=20, t=20, b=20)
    )
    # Enforce font sizes even if something overrides template
    fig_res.update_xaxes(title_font=dict(size=20), tickfont=dict(size=14))
    fig_res.update_yaxes(title_font=dict(size=20), tickfont=dict(size=14))

# Correlation figure
st.plotly_chart(
    fig_corr,
    use_container_width=True,
    config={
        "toImageButtonOptions": {
            "format": "png",      # png | svg | jpeg | webp
            "filename": "correlation",
            'width': None,
            'height': None,
            'scale': 2
        }
    }
)

# Residuals figure
st.plotly_chart(
    fig_res,
    use_container_width=True,
    config={
        "toImageButtonOptions": {
            "format": "png",
            "filename": "residuals",
            "height": None,
            "width": None,
            "scale": 2
        }
    }
)

if stats_rows:
    stats_df = pd.DataFrame(stats_rows, columns=["Molecule", "<E_vac> (eV)", "<χ> (eV)"])
    stats_df = stats_df.sort_values(by="<χ> (eV)", key=lambda s: s.astype(str), ascending=False, kind="mergesort")
    stats_df = stats_df.reset_index(drop=True)
    st.dataframe(stats_df, width='stretch')
else:
    st.info("No stats to display yet (need ≥3 valid points per molecule to fit).")

if len(inference_tables) > 0:
    st.subheader("**Inferred ε**")
    num = len(inference_tables)
    max_cols = min(3, num)
    mol_keys = list(inference_tables.keys())
    for i in range(0, num, max_cols):
        cols = st.columns(min(max_cols, num - i))
        for col, mk in zip(cols, mol_keys[i:i+max_cols]):
            with col:
                st.caption(mk)
                st.dataframe(inference_tables[mk], width='stretch')
else:
    st.caption("No ε inference performed (no rows with missing `epsilon`).")

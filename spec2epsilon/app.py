# streamlit_app.py
# ---
# Streamlit + Plotly app for spec2epsilon
# - Two tabs: Results (default) and Data (editable)
# - LaTeX labels via global MathJax v2 injection
# - Modebar download tuned for decent publication defaults

import io
import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from spec2epsilon import visualization

import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.options.mode.chained_assignment = None


# --- Page config ---
def _resolve_icon():
    for path in ("./figs/favicon.ico", "figs/favicon.ico"):
        if os.path.exists(path):
            return path
    return "🧪"

st.set_page_config(page_title="spec2epsilon", page_icon=_resolve_icon(), layout="wide")

# MathJax loader (v2) for Plotly LaTeX
js_path = os.path.join(os.path.dirname(__file__), "load-mathjax.js")
if os.path.exists(js_path):
    with open(js_path, "r", encoding="utf-8") as f:
        js = f.read()
    components.html(f"<script>{js}</script>", height=0)

st.markdown("<h1 style='margin-bottom:0'>spec2epsilon</h1>", unsafe_allow_html=True)
st.caption("Estimate solvent dielectric constants from fluorescence spectra")

# --- Sidebar: About ---
with st.sidebar:
    st.subheader("How to use")
    st.write(
        "- Upload one or more CSV files.\n"
        "- Required columns: `Solvent/solvent`, `epsilon`, `nr`, plus 1+ column with molecule's emission energy (eV or nm).\n"
        "- Empty `epsilon` cells can be inferred when a fit is available.\n"
        "- Review & edit data in the **Data** tab.\n"
        "- Choose solvents per molecule in **Selections**."
        
    )
    st.markdown("**Example CSV format:**")
    st.code(
        "Solvent,epsilon,nr,Mol1,Mol2\n"
        "Hexane,2.0165,1.375,389,395\n"
        "Toluene,2.38,1.496,416,434\n"
        "THF,7.58,1.407,430,470\n"
        "Film1,,1.60,440,465\n",
        language="csv",
    )

# --- Helpers ---

def _load_csv_files(uploaded_files) -> List[pd.DataFrame]:
    """Load each CSV via visualization.load_data (preferred) or pandas.read_csv; attach .name."""
    datas: List[pd.DataFrame] = []
    for uf in uploaded_files:
        raw = uf.getvalue()
        bio = io.BytesIO(raw)
        data = visualization.load_data(bio)
        data.name = os.path.splitext(os.path.basename(uf.name))[0]
        datas.append(data)
    return datas

def _collect_molecules(datas: List[pd.DataFrame]) -> List[str]:
    molecules: List[str] = []
    for df in datas:
        molecules.extend([c for c in df.columns if c not in ["Solvent", "epsilon", "nr", "solvent"]])
    # Preserve order
    seen, uniq = set(), []
    for m in molecules:
        if m not in seen:
            seen.add(m)
            uniq.append(m)
    return uniq

def _collect_solvents_for_molecule(datas: List[pd.DataFrame], molecule: str) -> List[str]:
    sv: List[str] = []
    for df in datas:
        if "Solvent" in df.columns and molecule in df.columns:
            sv.extend(df["Solvent"].dropna().astype(str).unique().tolist())
    seen, uniq = set(), []
    for s in sv:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq

# --- Upload ---
uploaded = st.file_uploader(
    "Upload one or more .csv files",
    type=["csv"],
    accept_multiple_files=True,
    help="Columns: Solvent/solvent, epsilon, nr, and 1+ molecule emission columns (eV or nm).",
)
if not uploaded:
    st.info("Upload CSV files to begin.")
    st.stop()

raw_datas = _load_csv_files(uploaded)
if not raw_datas:
    st.error("No data could be loaded from the uploaded files.")
    st.stop()

# --- Tabs: Results (default) | Data ---
TAB_RES, TAB_DATA = st.tabs(["Results", "Data"])

# --- DATA TAB (editable) ---
with TAB_DATA:
    st.subheader("Data (editable)")
    edited_datas: List[pd.DataFrame] = []
    for idx, df in enumerate(raw_datas):
        st.markdown(f"**{getattr(df, 'name', f'File {idx+1}')}**")
        edited = st.data_editor(
            df,
            num_rows="dynamic",
            width='stretch',
            key=f"editor_{getattr(df, 'name', str(idx))}",
        )
        edited.name = getattr(df, "name", f"File {idx+1}")
        edited_datas.append(edited)

# Use edited data if present
datas = edited_datas if edited_datas else raw_datas

# --- RESULTS TAB ---
with TAB_RES:
    st.subheader("Characterization")

    all_molecules = _collect_molecules(datas)
    if not all_molecules:
        st.error("No molecule columns found.")
        st.stop()

    # Selections inside Results
    with st.expander("Selections", expanded=True):
        selections: Dict[str, List[str]] = {}
        ncols = min(3, max(1, len(all_molecules)))
        chunks = [all_molecules[i::ncols] for i in range(ncols)]
        cols = st.columns(ncols)
        for col, mols in zip(cols, chunks):
            with col:
                for mol in mols:
                    options = _collect_solvents_for_molecule(datas, mol)
                    selections[mol] = st.multiselect(
                        f"{mol}", options=options, default=options[:], key=f"solv_{mol}"
                    )

    if visualization is None or not hasattr(visualization, "characterize") or not hasattr(visualization, "model"):
        st.error("`spec2epsilon.visualization` must provide `characterize` and `model` for fitting.")
        st.stop()

    fits: Dict[str, Tuple[Tuple[float, float], np.ndarray]] = {}
    stats_rows: List[List[str]] = []
    inference_tables: Dict[str, pd.DataFrame] = {}

    # Color map for molecules
    palette = px.colors.qualitative.Plotly
    if len(all_molecules) > len(palette):
        extra = px.colors.qualitative.Safe + px.colors.qualitative.Vivid + px.colors.qualitative.Set3
        palette = (palette + extra) * ((len(all_molecules) // len(palette)) + 1)
    color_map: Dict[str, str] = {m: palette[i] for i, m in enumerate(all_molecules)}

    # Figures
    fig_corr = go.Figure()
    fig_res = go.Figure()

    # Fit & plot
    for df in datas:
        if not set(["Solvent", "epsilon", "nr"]).issubset(df.columns):
            st.warning(f"File `{getattr(df, 'name', 'unknown')}` is missing required columns. Skipping.")
            continue

        for molecule in [c for c in df.columns if c not in ["Solvent", "epsilon", "nr", "solvent"]]:
            allowed_solvents = selections.get(molecule, [])
            if not allowed_solvents:
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

            alphas_st  = (epsilons[mask] - 1.0) / (epsilons[mask] + 1.0)
            alphas_opt = (nr[mask]**2 - 1.0) / (nr[mask]**2 + 1.0)
            emission_fit = emission[mask]

            opt, cov = visualization.characterize((alphas_st, alphas_opt), emission_fit)
            chi, e_vac = opt
            fits[molecule] = (opt, cov)

            function = visualization.model((alphas_st, alphas_opt), chi, e_vac)
            x = 2 * alphas_st - alphas_opt

            color = color_map[molecule]
            solvents = data_mol["Solvent"].to_numpy()[mask]

            # Correlation
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
                    "x=%{x:.3f}<br>Emission (eV)=%{y:.3f}<extra></extra>"
                ),
                customdata=solvents
            ))

            # Residuals
            residuals = emission_fit - function
            fig_res.add_trace(go.Scatter(
                x=x, y=residuals,
                mode="markers",
                name=molecule,
                legendgroup=molecule,
                marker=dict(color=color, size=9),
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Solvent=%{customdata}<br>"
                    "x=%{x:.3f}<br>Residual (eV)=%{y:.3f}<extra></extra>"
                ),
                customdata=solvents
            ))

            # Stats row
            error = np.sqrt(np.diag(cov)) if cov is not None else np.array([np.nan, np.nan])
            if hasattr(visualization, "format_number"):
                chi_fmt = visualization.format_number(chi, error[0], "")
                e_vac_fmt = visualization.format_number(e_vac, error[1], "")
            else:
                chi_fmt = f"{chi:.3f} ± {error[0]:.3f}" if np.isfinite(error[0]) else f"{chi:.3f}"
                e_vac_fmt = f"{e_vac:.3f} ± {error[1]:.3f}" if np.isfinite(error[1]) else f"{e_vac:.3f}"
            stats_rows.append([molecule, e_vac_fmt, chi_fmt])

        # ε inference (rows with missing epsilon)
        if "epsilon" in df.columns and df["epsilon"].isna().any() and fits and hasattr(visualization, "compute_dielectric"):
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
                            nrs[0],
                            median,
                            f"[{lower:.2f} , {upper:.2f}]"
                        ])
                    if rows:
                        df_inf = pd.DataFrame(rows, columns=["Film", "Emission (eV)", "Emission (nm)", "nr", "ε", "Interval"])
                        df_inf = df_inf.sort_values(by="ε", ascending=True, kind="mergesort")
                        df_inf["Emission (eV)"] = df_inf["Emission (eV)"].apply(lambda x: f"{x:.2f}")
                        df_inf["ε"] = df_inf["ε"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "∞")
                        df_inf["nr"] = df_inf["nr"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "∞")
                        inference_tables[molecule] = df_inf

    # Tight-ish layout and readable fonts
    if len(fig_corr.data) > 0:
        fig_corr.update_layout(
            xaxis_title=r"$2 \alpha_{st} - \alpha_{opt}$",
            yaxis_title="Energy (eV)",
            legend=dict(font=dict(size=16)),
            margin=dict(l=20, r=20, t=20, b=20),
        )
        fig_corr.update_xaxes(title_font=dict(size=20), tickfont=dict(size=14), automargin=True)
        fig_corr.update_yaxes(title_font=dict(size=20), tickfont=dict(size=14), automargin=True)

    if len(fig_res.data) > 0:
        fig_res.update_layout(
            xaxis_title=r"$2 \alpha_{st} - \alpha_{opt}$",
            yaxis_title="Residuals (eV)",
            legend=dict(font=dict(size=16)),
            margin=dict(l=20, r=20, t=20, b=20),
        )
        fig_res.update_xaxes(title_font=dict(size=20), tickfont=dict(size=14), automargin=True)
        fig_res.update_yaxes(title_font=dict(size=20), tickfont=dict(size=14), automargin=True)

    # Render with tuned modebar download (good balance for pubs)
    dl_config = {
        "toImageButtonOptions": {
            "format": "png",        # png | svg | jpeg | webp
            "filename": "correlation",  # updated per fig below
            "width": 1000,          # ~single-column @ 300dpi ≈ 1000 px
            "height": 625,
            "scale": 2              # keep fonts consistent
        }
    }
    st.plotly_chart(fig_corr, width='stretch', config=dl_config)

    dl_config_res = {
        "toImageButtonOptions": {
            "format": "png",
            "filename": "residuals",
            "width": 100,
            "height": 625,
            "scale": 2
        }
    }
    st.plotly_chart(fig_res, width='stretch', config=dl_config_res)

    if stats_rows:
        stats_df = pd.DataFrame(stats_rows, columns=["Molecule", "<E_vac> (eV)", "<χ> (eV)"])
        st.dataframe(stats_df, width='stretch')
    else:
        st.info("No stats to display yet (need ≥3 valid points per molecule to fit).")

    # Inferred ε
    st.subheader("Inferred ε")
    
    if inference_tables:
        keys = list(inference_tables.keys())
        max_cols = min(3, len(keys))
        for i in range(0, len(keys), max_cols):
            cols = st.columns(min(max_cols, len(keys) - i))
            for c, k in zip(cols, keys[i:i+max_cols]):
                with c:
                    st.caption(k)
                    st.dataframe(inference_tables[k], width='stretch')
    else:
        st.caption("No ε inference performed (no rows with missing `epsilon`).")

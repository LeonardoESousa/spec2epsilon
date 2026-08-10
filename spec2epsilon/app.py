# streamlit_app.py
# ---
# Streamlit + Plotly app for spec2epsilon
# - Two tabs: Results (default) and Data (editable)
# - LaTeX labels via global MathJax v2 injection
# - Modebar download tuned for decent publication defaults

import io
import json
import os
import re
import warnings
import zipfile
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as pkg_version
from typing import Dict, List, Tuple
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from spec2epsilon import visualization
from spec2epsilon.__version__ import __version__ as APP_VERSION

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
st.title("spec2epsilon")

st.markdown(
    "<p style='font-size:1.1rem; margin-top:0.1rem;'>Estimate solvent dielectric constants from fluorescence spectra</p>",
    unsafe_allow_html=True,
)

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




# --- Sidebar: About ---
with st.sidebar:
    st.subheader("Software")
    try:
        local_version = pkg_version("spec2epsilon")
    except PackageNotFoundError:
        local_version = APP_VERSION

    @st.cache_data(ttl=3600, show_spinner=False)
    def _fetch_latest_pypi_version() -> Tuple[str, str]:
        url = "https://pypi.org/pypi/spec2epsilon/json"
        try:
            with urlopen(url, timeout=3) as response:
                payload = json.loads(response.read().decode("utf-8"))
            latest = str(payload.get("info", {}).get("version", "")).strip()
            if latest:
                return latest, ""
            return "", "Could not read version from PyPI response."
        except URLError as exc:
            reason = getattr(exc, "reason", "network error")
            return "", f"Update check unavailable ({reason})."
        except Exception as exc:
            return "", f"Update check failed ({type(exc).__name__})."

    def _version_tuple(raw: str) -> Tuple[int, ...]:
        parts: List[int] = []
        for token in raw.split("."):
            digits = ""
            for char in token:
                if char.isdigit():
                    digits += char
                else:
                    break
            parts.append(int(digits) if digits else 0)
        return tuple(parts)

    def _is_newer(candidate: str, current: str) -> bool:
        c1 = _version_tuple(candidate)
        c2 = _version_tuple(current)
        max_len = max(len(c1), len(c2))
        c1 = c1 + (0,) * (max_len - len(c1))
        c2 = c2 + (0,) * (max_len - len(c2))
        return c1 > c2

    st.write(f"Version: {local_version}")
    latest_version, update_error = _fetch_latest_pypi_version()
    if latest_version:
        if _is_newer(latest_version, local_version):
            st.warning(f"New version available: {latest_version}")
        else:
            st.success("You are using the latest available version.")
    else:
        st.caption(update_error)

    st.subheader("Cite as")
    st.write("Bueno, Fernando Teixeira, Pedro Henrique de Oliveira Neto, and Leonardo Evaristo de Sousa. 'Determining Static Dielectric Constants from Fluorescence Spectra.' The Journal of Physical Chemistry Letters (2026). DOI: https://doi.org/10.1021/acs.jpclett.5c03806")
    st.subheader("How to use")
    st.write(
        "- Upload one or more CSV files.\n"
        "- Required columns: `Solvent/solvent`, `epsilon`, `nr`, plus 1+ column with molecule's emission energy (eV or nm).\n"
        "- Empty `epsilon` cells can be inferred when a fit is available.\n"
        "- Review & edit data in the **Data** tab.\n"
        "- Choose solvents per molecule in **Solvent Selection**."
        
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

def _collect_solvents_by_epsilon_validity(datas: List[pd.DataFrame]) -> Tuple[List[str], List[str]]:
    """Return (finite_epsilon_solvents, nan_epsilon_solvents), preserving first-seen order."""
    finite_seen, nan_seen = set(), set()
    finite_solvents: List[str] = []
    nan_solvents: List[str] = []

    for df in datas:
        if not set(["Solvent", "epsilon"]).issubset(df.columns):
            continue

        for _, row in df[["Solvent", "epsilon"]].dropna(subset=["Solvent"]).iterrows():
            solvent = str(row["Solvent"])
            epsilon = row["epsilon"]
            if pd.isna(epsilon):
                if solvent not in nan_seen:
                    nan_seen.add(solvent)
                    nan_solvents.append(solvent)
            else:
                if solvent not in finite_seen:
                    finite_seen.add(solvent)
                    finite_solvents.append(solvent)

    return finite_solvents, nan_solvents


def _uniquify_name(base_name: str, used_names: set) -> str:
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

# MathJax loader (v2) for Plotly LaTeX
js_path = os.path.join(os.path.dirname(__file__), "load-mathjax.js")
if os.path.exists(js_path):
    with open(js_path, "r", encoding="utf-8") as f:
        js = f.read()
    components.html(f"<script>{js}</script>", height=0)


# --- RESULTS TAB ---
with TAB_RES:
    st.subheader("Characterization")

    all_molecules = _collect_molecules(datas)
    if not all_molecules:
        st.error("No molecule columns found.")
        st.stop()

    selected_molecules = st.multiselect(
        "Molecules",
        options=all_molecules,
        default=all_molecules,
        key="selected_molecules",
        help="Run fitting and plots only for selected molecules.",
    )
    if not selected_molecules:
        st.warning("Select at least one molecule to run the analysis.")
        st.stop()
    
    finite_epsilon_solvents, nan_epsilon_solvents = _collect_solvents_by_epsilon_validity(datas)
    auto_selected_nan_solvents = set(nan_epsilon_solvents)
    finite_epsilon_solvents = sorted([
        solv for solv in finite_epsilon_solvents
        if solv not in auto_selected_nan_solvents
    ])
    
    with st.expander("Solvent Selection", expanded=True):
        selected_solvents = st.multiselect(
            "Solvents",
            options=finite_epsilon_solvents,
            default=finite_epsilon_solvents,
            key="global_solvents",
        )
    
    # Build per-molecule selections from the single global selection
    selections: Dict[str, List[str]] = {
        mol: [
            solv for solv in _collect_solvents_for_molecule(datas, mol)
            if solv in selected_solvents or solv in auto_selected_nan_solvents
        ]
        for mol in selected_molecules
    }

    if visualization is None or not hasattr(visualization, "characterize") or not hasattr(visualization, "model"):
        st.error("`spec2epsilon.visualization` must provide `characterize` and `model` for fitting.")
        st.stop()

    fits: Dict[str, Tuple[Tuple[float, float], np.ndarray]] = {}
    fit_export_entries: List[Dict[str, object]] = []
    stats_rows: List[List[str]] = []
    inference_tables: Dict[str, pd.DataFrame] = {}

    # Color map for molecules
    palette = px.colors.qualitative.Plotly
    if len(selected_molecules) > len(palette):
        extra = px.colors.qualitative.Safe + px.colors.qualitative.Vivid + px.colors.qualitative.Set3
        palette = (palette + extra) * ((len(selected_molecules) // len(palette)) + 1)
    color_map: Dict[str, str] = {m: palette[i] for i, m in enumerate(selected_molecules)}

    # Figures
    fig_corr = go.Figure()
    fig_res = go.Figure()
    fig_chi_vac = go.Figure()
    chi_vac_legend_seen = set()

    # Fit & plot
    for df in datas:
        if not set(["Solvent", "epsilon", "nr"]).issubset(df.columns):
            st.warning(f"File `{getattr(df, 'name', 'unknown')}` is missing required columns. Skipping.")
            continue

        for molecule in [c for c in df.columns if c in selected_molecules]:
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
            fit_export_entries.append({
                "molecule": molecule,
                "E_vac": float(e_vac),
                "chi": float(chi),
                "covariance_matrix": cov.tolist() if cov is not None else None,
            })
            error = np.sqrt(np.diag(cov)) if cov is not None else np.array([np.nan, np.nan])

            function = visualization.model((alphas_st, alphas_opt), chi, e_vac)
            x = 2 * alphas_st - alphas_opt

            #Compute R^2
            ss_res = np.sum((emission_fit - function) ** 2)
            ss_tot = np.sum((emission_fit - np.mean(emission_fit)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

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

            # Fitted parameter map: point + 68% uncertainty ellipse
            show_legend = molecule not in chi_vac_legend_seen
            chi_vac_legend_seen.add(molecule)
            fig_chi_vac.add_trace(go.Scatter(
                x=[chi], y=[e_vac],
                mode="markers",
                name=molecule,
                legendgroup=molecule,
                showlegend=show_legend,
                marker=dict(color=color, size=9, line=dict(color=color, width=0.5)),
                customdata=np.array([[error[0], error[1]]]),
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "χ (eV)=%{x:.3f}<br>"
                    "E_vac (eV)=%{y:.3f}<br>"
                    "σχ=%{customdata[0]:.3f}<br>"
                    "σE_vac=%{customdata[1]:.3f}<extra></extra>"
                )
            ))

            if cov is not None and np.shape(cov) == (2, 2) and np.all(np.isfinite(cov)):
                try:
                    ellipse = visualization.confidence_ellipse(
                        (np.array([chi, e_vac]), cov), confidence=0.68, num_points=200
                    )
                    fig_chi_vac.add_trace(go.Scatter(
                        x=ellipse[0], y=ellipse[1],
                        mode="lines",
                        name=molecule + " (68% ellipse)",
                        legendgroup=molecule,
                        showlegend=False,
                        line=dict(color=color, width=1.5),
                        hoverinfo="skip",
                    ))
                except Exception:
                    pass

            # Stats row
            if hasattr(visualization, "format_number"):
                chi_fmt = visualization.format_number(chi, error[0], "")
                e_vac_fmt = visualization.format_number(e_vac, error[1], "")
            else:
                chi_fmt = f"{chi:.3f} ± {error[0]:.3f}" if np.isfinite(error[0]) else f"{chi:.3f}"
                e_vac_fmt = f"{e_vac:.3f} ± {error[1]:.3f}" if np.isfinite(error[1]) else f"{e_vac:.3f}"
            r_squared_fmt = f"{r_squared:.2f}"    
            stats_rows.append([molecule, e_vac_fmt, chi_fmt, r_squared_fmt])

        # ε inference (rows with missing epsilon)
        if "epsilon" in df.columns and df["epsilon"].isna().any() and fits and hasattr(visualization, "compute_dielectric"):
            inference = df[df["epsilon"].isna()].copy()
            if not inference.empty:
                for molecule in [c for c in df.columns if c in selected_molecules]:
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

    if len(fig_chi_vac.data) > 0:
        fig_chi_vac.update_layout(
            xaxis_title=r"$\chi\,(\mathrm{eV})$",
            yaxis_title=r"$\Delta E_{vac}\,(\mathrm{eV})$",
            legend=dict(font=dict(size=16)),
            margin=dict(l=20, r=20, t=20, b=20),
        )
        fig_chi_vac.update_xaxes(title_font=dict(size=20), tickfont=dict(size=14), automargin=True)
        fig_chi_vac.update_yaxes(title_font=dict(size=20), tickfont=dict(size=14), automargin=True)

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

    if stats_rows:
        stats_df = pd.DataFrame(stats_rows, columns=["Molecule", "<ΔE_vac> (eV)", "<χ> (eV)", "R²"])
        st.dataframe(stats_df, width='stretch')

        export_payload = {
            "schema_version": 1,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "molecules": fit_export_entries,
        }

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
        st.download_button(
            "Download fit parameters (.zip)",
            data=zip_buffer.getvalue(),
            file_name="spec2epsilon_fit_results.zip",
            mime="application/zip",
            help="Download one .npy per fitted molecule (molecule_name.npy) with E_vac, chi and covariance matrix.",
        )
    else:
        st.info("No stats to display yet (need ≥3 valid points per molecule to fit).")


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

    dl_config_chi = {
        "toImageButtonOptions": {
            "format": "png",
            "filename": "chi_vs_evac",
            "width": 1000,
            "height": 625,
            "scale": 2,
        }
    }
    if len(fig_chi_vac.data) > 0:
        st.plotly_chart(fig_chi_vac, width='stretch', config=dl_config_chi)

    
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

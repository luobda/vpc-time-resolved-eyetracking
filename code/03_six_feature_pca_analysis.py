# -*- coding: utf-8 -*-
"""
VPC six-feature construction and PCA analysis.

The original Excel schema remains Chinese and is never modified on disk.
Internal English aliases are used only in memory for analysis formulas.

Core metrics shown in figure-facing text:
- Memory representation
- Processing noise
- Retrieval latency

The script preserves the original six-feature construction, correlation,
KMO/Bartlett, PCA, loading, and reporting logic. Only the retained PCA
diagnostic figure is exported, as Figure 1.

Dependencies:
    pip install pandas numpy scipy statsmodels scikit-learn matplotlib openpyxl
"""

from __future__ import annotations

import argparse
import os
import zipfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ============================================================
# 0. Parameters
# ======================== 路径配置 ========================
# 以当前脚本所在的 code 文件夹为基准，自动定位同级的 data 文件夹
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = (SCRIPT_DIR / ".." / "data").resolve()

DEFAULT_INPUT_FILE = DATA_DIR / "fitting_results_difficulty.xlsx"

OUTPUT_DIR = "Chapter3_Output"
ZIP_NAME = "Chapter3_Output.zip"

# The current metric named “记忆提取速率” is operationally a latency:
# first significant peak time. Larger values mean slower retrieval.
RETRIEVAL_IS_LATENCY = True

# Larger cognitive processing noise means lower stability.
PROCESSING_NOISE_HIGHER_IS_WORSE = True

# |r| >= .80 is treated as high redundancy.
HIGH_CORR_THRESHOLD = 0.80

# Loadings with |loading| >= .50 are highlighted as major loadings.
MAJOR_LOADING_THRESHOLD = 0.50

# Number of PCs used for later clustering.
N_PC_FOR_CLUSTERING = 6

# Old figure outputs removed or renumbered in this version.
DEPRECATED_OUTPUTS = [
    "Fig_3_3A_feature_framework.png",
    "Fig_3_3A_feature_framework.pdf",
    "Fig_3_3B_feature_correlation.png",
    "Fig_3_3B_feature_correlation.pdf",
    "Fig_3_3C_PCA_diagnostics.png",
    "Fig_3_3C_PCA_diagnostics.pdf",
    "Fig_3_3D_PCA_biplot_optional.png",
    "Fig_3_3D_PCA_biplot_optional.pdf",
]


# ============================================================
# 1. General utilities
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT_FILE,
        help="Path to the trial-level Excel file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_DIR,
        help="Output directory."
    )
    return parser.parse_args()


def zscore(series: pd.Series) -> pd.Series:
    """Population SD Z-score, consistent with scikit-learn StandardScaler."""
    return (series - series.mean()) / series.std(ddof=0)


def format_p(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "<.001"
    return f"{p:.3f}".replace("0.", ".")


def p_to_stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def save_csv(df: pd.DataFrame, output_dir: Path, filename: str, index: bool = False) -> Path:
    path = output_dir / filename
    df.to_csv(path, index=index, encoding="utf-8-sig")
    return path


def remove_deprecated_outputs(output_dir: Path) -> None:
    """Remove obsolete figure files so they are not included in the ZIP archive."""
    for filename in DEPRECATED_OUTPUTS:
        path = output_dir / filename
        if path.exists():
            path.unlink()


def set_publication_style() -> None:
    """Compact style suitable for journal figures."""
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 10,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })


def save_figure(fig: plt.Figure, output_dir: Path, basename: str, dpi: int = 600) -> None:
    """Save both high-resolution PNG and vector PDF."""
    png_path = output_dir / f"{basename}.png"
    pdf_path = output_dir / f"{basename}.pdf"
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)


def clean_axes(ax: plt.Axes) -> None:
    """Hide the top and right spines."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def fit_subject_ols(data: pd.DataFrame, y_col: str, x_col: str = "difficulty_z") -> dict:
    """
    Fit subject-level OLS: y ~ difficulty_z.
    Returns intercept, slope, SE, t, p, CI, R², and n.
    """
    temp = data[[y_col, x_col]].dropna().copy()

    empty = {
        "intercept": np.nan,
        "slope": np.nan,
        "slope_SE": np.nan,
        "t": np.nan,
        "p": np.nan,
        "CI_lower": np.nan,
        "CI_upper": np.nan,
        "r_squared": np.nan,
        "n_trials": len(temp),
    }

    if len(temp) < 3 or temp[x_col].std(ddof=0) == 0:
        return empty

    model = smf.ols(f"{y_col} ~ {x_col}", data=temp).fit()
    ci = model.conf_int()

    return {
        "intercept": model.params["Intercept"],
        "slope": model.params[x_col],
        "slope_SE": model.bse[x_col],
        "t": model.tvalues[x_col],
        "p": model.pvalues[x_col],
        "CI_lower": ci.loc[x_col, 0],
        "CI_upper": ci.loc[x_col, 1],
        "r_squared": model.rsquared,
        "n_trials": int(model.nobs),
    }


def corr_with_pvalues(data: pd.DataFrame, variables: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    corr_matrix = pd.DataFrame(np.eye(len(variables)), index=variables, columns=variables)
    p_matrix = pd.DataFrame(np.zeros((len(variables), len(variables))), index=variables, columns=variables)

    for i, var1 in enumerate(variables):
        for j, var2 in enumerate(variables):
            if i == j:
                corr_matrix.loc[var1, var2] = 1.0
                p_matrix.loc[var1, var2] = 0.0
            elif i < j:
                temp = data[[var1, var2]].dropna()
                if len(temp) >= 3:
                    r, p = stats.pearsonr(temp[var1], temp[var2])
                else:
                    r, p = np.nan, np.nan
                corr_matrix.loc[var1, var2] = r
                corr_matrix.loc[var2, var1] = r
                p_matrix.loc[var1, var2] = p
                p_matrix.loc[var2, var1] = p

    return corr_matrix, p_matrix


def find_high_corr_pairs(corr_matrix: pd.DataFrame, threshold: float = 0.80) -> pd.DataFrame:
    rows = []
    variables = list(corr_matrix.columns)

    for i in range(len(variables)):
        for j in range(i + 1, len(variables)):
            var1, var2 = variables[i], variables[j]
            r = corr_matrix.loc[var1, var2]
            if pd.notna(r) and abs(r) >= threshold:
                rows.append({
                    "Feature 1": var1,
                    "Feature 2": var2,
                    "r": r,
                    "|r|": abs(r),
                })

    return pd.DataFrame(rows)


def kmo_test(data: np.ndarray) -> tuple[float, np.ndarray]:
    """
    KMO test. Useful as a supplementary adequacy index.
    PCA itself does not strictly require KMO, so report cautiously.
    """
    x = np.asarray(data, dtype=float)
    r = np.corrcoef(x, rowvar=False)

    det_r = np.linalg.det(r)
    if det_r <= 1e-12:
        r = r + np.eye(r.shape[0]) * 1e-8

    inv_r = np.linalg.inv(r)

    partial_corr = np.zeros_like(r)
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            if i == j:
                partial_corr[i, j] = 0
            else:
                partial_corr[i, j] = -inv_r[i, j] / np.sqrt(inv_r[i, i] * inv_r[j, j])

    r_no_diag = r.copy()
    np.fill_diagonal(r_no_diag, 0)

    r2 = r_no_diag ** 2
    p2 = partial_corr ** 2

    kmo_total = np.sum(r2) / (np.sum(r2) + np.sum(p2))

    kmo_per_variable = np.zeros(r.shape[0])
    for i in range(r.shape[0]):
        r2_i = np.sum(r2[i, :])
        p2_i = np.sum(p2[i, :])
        kmo_per_variable[i] = r2_i / (r2_i + p2_i)

    return kmo_total, kmo_per_variable


def bartlett_sphericity_test(data: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(data, dtype=float)
    n, p = x.shape
    r = np.corrcoef(x, rowvar=False)

    det_r = np.linalg.det(r)
    if det_r <= 1e-12:
        det_r = 1e-12

    chi_square = -(n - 1 - (2 * p + 5) / 6) * np.log(det_r)
    df = p * (p - 1) / 2
    p_value = stats.chi2.sf(chi_square, df)

    return chi_square, df, p_value


# ============================================================
# 2. Data construction
# ============================================================

def load_and_prepare_data(input_file: Path) -> pd.DataFrame:
    if not input_file.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_file}\n"
            "Please provide --input or edit DEFAULT_INPUT_FILE."
        )

    raw_df = pd.read_excel(input_file)

    # Keep the original Excel schema unchanged. These aliases exist only in
    # the in-memory DataFrame used by the statistical analysis.
    rename_map = {
        "被试": "subject",
        "试次": "trial",
        "新图位置": "new_position",
        "记忆表征能力": "memory_representation",
        "认知加工噪声水平": "processing_noise",
        "记忆提取速率": "retrieval_latency",
        "记忆提取潜伏期": "retrieval_latency",
        "平均整体偏好": "mean_overall_preference",
        "difficulty": "difficulty",
    }

    df = raw_df.rename(columns=rename_map).copy()

    required_cols = [
        "subject",
        "trial",
        "memory_representation",
        "processing_noise",
        "retrieval_latency",
        "difficulty",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing_cols)
            + "\n\nPlease check the Excel headers and rename_map."
        )

    for col in ["memory_representation", "processing_noise", "retrieval_latency", "difficulty"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required_cols).copy()
    df["subject"] = df["subject"].astype(str)
    df["trial"] = df["trial"].astype(int)
    df["difficulty_z"] = zscore(df["difficulty"])
    df = df.sort_values(["subject", "trial"]).reset_index(drop=True)

    return df


def construct_six_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    static_features = (
        df.groupby("subject")
        .agg(
            n_trials=("trial", "count"),
            mean_difficulty=("difficulty", "mean"),
            mean_memory_representation=("memory_representation", "mean"),
            mean_processing_noise=("processing_noise", "mean"),
            mean_retrieval_latency=("retrieval_latency", "mean"),
        )
        .reset_index()
    )

    static_features["mean_representation_good"] = static_features["mean_memory_representation"]

    if PROCESSING_NOISE_HIGHER_IS_WORSE:
        static_features["mean_processing_stability_good"] = -static_features["mean_processing_noise"]
    else:
        static_features["mean_processing_stability_good"] = static_features["mean_processing_noise"]

    if RETRIEVAL_IS_LATENCY:
        static_features["mean_retrieval_efficiency_good"] = -static_features["mean_retrieval_latency"]
    else:
        static_features["mean_retrieval_efficiency_good"] = static_features["mean_retrieval_latency"]

    rows = []

    for subject, sub_df in df.groupby("subject"):
        rep = fit_subject_ols(sub_df, "memory_representation")
        noise = fit_subject_ols(sub_df, "processing_noise")
        latency = fit_subject_ols(sub_df, "retrieval_latency")

        rows.append({
            "subject": subject,
            "n_trials_slope": len(sub_df),

            "rep_intercept": rep["intercept"],
            "rep_slope_raw": rep["slope"],
            "rep_slope_SE": rep["slope_SE"],
            "rep_slope_t": rep["t"],
            "rep_slope_p": rep["p"],
            "rep_r_squared": rep["r_squared"],

            "noise_intercept": noise["intercept"],
            "noise_slope_raw": noise["slope"],
            "noise_slope_SE": noise["slope_SE"],
            "noise_slope_t": noise["t"],
            "noise_slope_p": noise["p"],
            "noise_r_squared": noise["r_squared"],

            "latency_intercept": latency["intercept"],
            "latency_slope_raw": latency["slope"],
            "latency_slope_SE": latency["slope_SE"],
            "latency_slope_t": latency["t"],
            "latency_slope_p": latency["p"],
            "latency_r_squared": latency["r_squared"],
        })

    dynamic_features = pd.DataFrame(rows)

    dynamic_features["representation_maintenance_slope_good"] = dynamic_features["rep_slope_raw"]

    if PROCESSING_NOISE_HIGHER_IS_WORSE:
        dynamic_features["processing_stability_slope_good"] = -dynamic_features["noise_slope_raw"]
    else:
        dynamic_features["processing_stability_slope_good"] = dynamic_features["noise_slope_raw"]

    if RETRIEVAL_IS_LATENCY:
        dynamic_features["retrieval_efficiency_slope_good"] = -dynamic_features["latency_slope_raw"]
    else:
        dynamic_features["retrieval_efficiency_slope_good"] = dynamic_features["latency_slope_raw"]

    subject_features = static_features.merge(dynamic_features, on="subject", how="inner")

    raw_cols = [
        "mean_representation_good",
        "mean_processing_stability_good",
        "mean_retrieval_efficiency_good",
        "representation_maintenance_slope_good",
        "processing_stability_slope_good",
        "retrieval_efficiency_slope_good",
    ]

    z_cols = [
        "z_mean_representation",
        "z_mean_processing_stability",
        "z_mean_retrieval_efficiency",
        "z_representation_maintenance_slope",
        "z_processing_stability_slope",
        "z_retrieval_efficiency_slope",
    ]

    for raw_col, z_col in zip(raw_cols, z_cols):
        subject_features[z_col] = zscore(subject_features[raw_col])

    subject_features = subject_features.dropna(subset=z_cols).reset_index(drop=True)

    # Display labels retain the three fixed core metric names.
    label_map = {
        "z_mean_representation": "Memory Representation Ability",
        "z_mean_processing_stability": "Cognitive Processing Noise Level",
        "z_mean_retrieval_efficiency": "Memory Retrieval Latency",
        "z_representation_maintenance_slope": "Memory Representation Ability slope",
        "z_processing_stability_slope": "Cognitive Processing Noise slope",
        "z_retrieval_efficiency_slope": "Memory Retrieval Latency slope",
    }

    analysis_features = subject_features[z_cols].copy().rename(columns=label_map)

    metadata = {
        "raw_cols": raw_cols,
        "z_cols": z_cols,
        "label_map": label_map,
        "feature_labels": list(analysis_features.columns),
    }

    return subject_features, analysis_features, metadata


# ============================================================
# 3. Tables
# ============================================================

def make_feature_definition_table() -> pd.DataFrame:
    rows = [
        {
            "Feature": "Memory representation mean",
            "Computation": "Subject-level mean of memory representation across valid trials",
            "Direction harmonization": "Original direction retained",
            "Interpretation after harmonization": "Higher = stronger late novelty preference / better memory representation",
        },
        {
            "Feature": "Processing noise mean (reversed)",
            "Computation": "Negative subject-level mean of processing noise",
            "Direction harmonization": "Processing noise reversed",
            "Interpretation after harmonization": "Higher = lower processing noise after direction reversal",
        },
        {
            "Feature": "Retrieval latency mean (reversed)",
            "Computation": "Negative subject-level mean of retrieval latency",
            "Direction harmonization": "Retrieval latency reversed",
            "Interpretation after harmonization": "Higher = shorter retrieval latency after direction reversal",
        },
        {
            "Feature": "Memory representation slope",
            "Computation": "Subject-level slope of memory representation predicted by standardized difficulty",
            "Direction harmonization": "Original direction retained",
            "Interpretation after harmonization": "Higher = better maintenance of Memory representation under increasing difficulty",
        },
        {
            "Feature": "Processing noise slope (reversed)",
            "Computation": "Negative subject-level slope of processing noise predicted by standardized difficulty",
            "Direction harmonization": "Noise slope reversed",
            "Interpretation after harmonization": "Higher = less Processing noise under increasing difficulty after direction reversal",
        },
        {
            "Feature": "Retrieval latency slope (reversed)",
            "Computation": "Negative subject-level slope of retrieval latency predicted by standardized difficulty",
            "Direction harmonization": "Latency slope reversed",
            "Interpretation after harmonization": "Higher = shorter Retrieval latency under increasing difficulty after direction reversal",
        },
    ]

    return pd.DataFrame(rows)


def make_descriptive_table(analysis_features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in analysis_features.columns:
        x = analysis_features[col].dropna()
        rows.append({
            "Feature": col,
            "Mean": x.mean(),
            "SD": x.std(ddof=1),
            "Median": x.median(),
            "Q1": x.quantile(0.25),
            "Q3": x.quantile(0.75),
            "Min": x.min(),
            "Max": x.max(),
            "Skewness": stats.skew(x, bias=False),
            "Kurtosis": stats.kurtosis(x, bias=False),
        })
    return pd.DataFrame(rows)


def make_correlation_long_table(corr_matrix: pd.DataFrame, p_matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    variables = list(corr_matrix.columns)
    for i in range(len(variables)):
        for j in range(i + 1, len(variables)):
            v1, v2 = variables[i], variables[j]
            r = corr_matrix.loc[v1, v2]
            p = p_matrix.loc[v1, v2]
            rows.append({
                "Feature 1": v1,
                "Feature 2": v2,
                "r": r,
                "p": p,
                "p_formatted": format_p(p),
                "significance": p_to_stars(p),
            })
    return pd.DataFrame(rows)


def make_pca_interpretation_table(
    pca_summary: pd.DataFrame,
    loading_matrix: pd.DataFrame,
    n_pc: int = 5,
) -> pd.DataFrame:
    suggested_names = {
        "PC1": "Static processing style",
        "PC2": "Difficulty-adaptation trade-off",
        "PC3": "Retrieval latency dimension",
        "PC4": "Baseline–dynamic retrieval latency dissociation",
        "PC5": "Integrated dynamic adaptation",
    }

    rows = []

    for pc in [f"PC{i}" for i in range(1, n_pc + 1)]:
        pc_loadings = loading_matrix[pc].sort_values(key=lambda s: s.abs(), ascending=False)
        major = pc_loadings[pc_loadings.abs() >= MAJOR_LOADING_THRESHOLD]
        if len(major) == 0:
            major = pc_loadings.iloc[:2]

        major_text = "; ".join([f"{idx} ({val:.2f})" for idx, val in major.items()])

        var_percent = pca_summary.loc[pca_summary["Component"] == pc, "Explained variance (%)"].iloc[0]
        cum_percent = pca_summary.loc[pca_summary["Component"] == pc, "Cumulative variance (%)"].iloc[0]

        rows.append({
            "Component": pc,
            "Explained variance (%)": var_percent,
            "Cumulative variance (%)": cum_percent,
            "Major loadings": major_text,
            "Suggested interpretation": suggested_names.get(pc, ""),
        })

    return pd.DataFrame(rows)


# ============================================================
# 4. Figures
# ============================================================

def plot_pca_diagnostics(
    pca_summary: pd.DataFrame,
    loading_matrix: pd.DataFrame,
    output_dir: Path,
    n_pc_for_loading: int = 6,
) -> None:
    """
    Create an APA 7 / BRM-oriented two-panel PCA diagnostic figure.

    Panel a:
        Explained variance bars and cumulative explained variance line on one
        common percentage axis. Using one axis avoids the visual ambiguity of
        dual y-axes.

    Panel b:
        PCA loading heatmap. Cell values are printed directly, and loadings
        meeting MAJOR_LOADING_THRESHOLD are bolded so interpretation does not
        depend on color alone.

    The overall figure title is intentionally omitted from the image. In the
    manuscript, place the figure number and italicized title above the figure,
    and define visual elements in the figure note below it.
    """
    required_summary_cols = {
        "Component",
        "Explained variance (%)",
        "Cumulative variance (%)",
    }
    missing = required_summary_cols.difference(pca_summary.columns)
    if missing:
        raise ValueError(
            "pca_summary is missing required columns: "
            + ", ".join(sorted(missing))
        )

    components = pca_summary["Component"].astype(str).tolist()
    n_components = len(components)
    if n_components == 0:
        raise ValueError("pca_summary contains no PCA components.")

    x = np.arange(1, n_components + 1)
    explained = pca_summary["Explained variance (%)"].to_numpy(dtype=float)
    cumulative = pca_summary["Cumulative variance (%)"].to_numpy(dtype=float)

    n_show = min(
        int(n_pc_for_loading),
        loading_matrix.shape[1],
        n_components,
    )
    if n_show < 1:
        raise ValueError("No PCA loading columns are available for plotting.")

    pcs_to_show = [f"PC{i}" for i in range(1, n_show + 1)]
    missing_pcs = [pc for pc in pcs_to_show if pc not in loading_matrix.columns]
    if missing_pcs:
        raise ValueError(
            "loading_matrix is missing expected components: "
            + ", ".join(missing_pcs)
        )

    loadings = loading_matrix.loc[:, pcs_to_show].copy()

    # Use the feature labels carried by the analysis matrix.
    y_tick_labels = [str(label) for label in loadings.index]

    # Approximate double-column width. All text sizes come from the global
    # publication style and therefore refer to the final output size.
    fig = plt.figure(figsize=(7.8, 3.55), facecolor="white")
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.00, 1.24],
        wspace=0.85,
    )

    # Color-blind-conscious contrast, with line type and marker shape providing
    # redundant visual encoding.
    bar_color = "#A9C7DF"
    bar_edge = "#4C78A8"
    line_color = "#B24A3A"
    grid_color = "#D9D9D9"

    # ------------------------------------------------------------------
    # Panel a: explained and cumulative variance
    # ------------------------------------------------------------------
    ax_var = fig.add_subplot(gs[0, 0])

    bars = ax_var.bar(
        x,
        explained,
        width=0.68,
        color=bar_color,
        edgecolor=bar_edge,
        linewidth=0.8,
        label="Explained variance",
        zorder=2,
    )

    cumulative_line, = ax_var.plot(
        x,
        cumulative,
        color=line_color,
        marker="o",
        markersize=4.0,
        markerfacecolor="white",
        markeredgecolor=line_color,
        markeredgewidth=0.9,
        linewidth=1.5,
        label="Cumulative variance",
        zorder=3,
    )

    ax_var.set_xlim(0.45, n_components + 0.55)
    ax_var.set_ylim(0, 110)
    ax_var.set_xticks(x)
    ax_var.set_xticklabels(components)
    ax_var.set_yticks(np.arange(0, 101, 20))
    ax_var.set_xlabel("Principal component")
    ax_var.set_ylabel("Variance explained (%)")
    ax_var.set_title(
        "",
        loc="left",
        pad=8,
        fontweight="normal",
    )

    # A light horizontal grid supports value comparison without dominating data.
    ax_var.grid(
        axis="y",
        color=grid_color,
        linewidth=0.55,
        alpha=0.75,
        zorder=0,
    )
    ax_var.set_axisbelow(True)
    clean_axes(ax_var)

    # Label bar heights only. The cumulative series remains uncluttered and can
    # be read from the common percentage axis.
    label_offset = 1
    for rect, value in zip(bars, explained):
        ax_var.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + label_offset,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=7.0,
        )

    # Label cumulative variance points
    for xi, value in zip(x, cumulative):
        ax_var.text(
            xi,
            value + 5.0,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=7.0,
            zorder=4,
        )

    ax_var.legend(
        handles=[bars, cumulative_line],
        loc="upper left",
        frameon=False,
        handlelength=1.8,
        borderaxespad=0.3,
    )

    ax_var.text(
        -0.16,
        1.06,
        "A",
        transform=ax_var.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        clip_on=False,
    )

    # ------------------------------------------------------------------
    # Panel b: feature loadings
    # ------------------------------------------------------------------
    ax_load = fig.add_subplot(gs[0, 1])

    im = ax_load.imshow(
        loadings.to_numpy(dtype=float),
        vmin=-1,
        vmax=1,
        cmap="RdBu_r",
        aspect="equal",
        interpolation="nearest",
    )

    ax_load.set_xticks(np.arange(loadings.shape[1]))
    ax_load.set_yticks(np.arange(loadings.shape[0]))
    ax_load.set_xticklabels(pcs_to_show)
    ax_load.set_yticklabels(y_tick_labels)
    ax_load.set_xlabel("Principal component")
    ax_load.set_title(
        "",
        loc="left",
        pad=8,
        fontweight="normal",
    )
    ax_load.tick_params(axis="both", length=0)
    ax_load.tick_params(axis="y", pad=5)

    # Thin separators improve cell tracking without adding a heavy table grid.
    ax_load.set_xticks(
        np.arange(-0.5, loadings.shape[1], 1),
        minor=True,
    )
    ax_load.set_yticks(
        np.arange(-0.5, loadings.shape[0], 1),
        minor=True,
    )
    ax_load.grid(
        which="minor",
        color="white",
        linestyle="-",
        linewidth=0.7,
    )
    ax_load.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    for i in range(loadings.shape[0]):
        for j in range(loadings.shape[1]):
            value = float(loadings.iloc[i, j])
            is_major = abs(value) >= MAJOR_LOADING_THRESHOLD

            ax_load.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7.1,
                fontweight="bold" if is_major else "normal",
                color="white" if abs(value) >= 0.62 else "black",
            )

    for spine in ax_load.spines.values():
        spine.set_visible(False)

    ax_load.text(
        -0.12,
        1.06,
        "B",
        transform=ax_load.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        clip_on=False,
    )

    cbar = fig.colorbar(
        im,
        ax=ax_load,
        fraction=0.046,
        pad=0.035,
    )
    cbar.set_label("PCA loading")
    cbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    cbar.ax.tick_params(labelsize=7.2, length=2.5, width=0.6)
    cbar.outline.set_linewidth(0.6)

    # No embedded overall title: APA figure number/title belong in the manuscript.
    fig.subplots_adjust(
        left=0.075,
        right=0.955,
        bottom=0.18,
        top=0.88,
    )

    save_figure(
        fig,
        output_dir,
        "Figure1_PCA_diagnostics",
        dpi=600,
    )


# ============================================================
# 5. Main analysis
# ============================================================

def main() -> None:
    args = parse_args()
    input_file = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    remove_deprecated_outputs(output_dir)

    set_publication_style()

    df = load_and_prepare_data(input_file)
    subject_features, analysis_features, metadata = construct_six_features(df)
    feature_labels = metadata["feature_labels"]

    # --------------------------
    # QC
    # --------------------------
    trial_counts = df.groupby("subject").size()

    qc_summary = pd.DataFrame({
        "Item": [
            "Valid observations",
            "Number of subjects",
            "Number of unique trials",
            "Minimum trials per subject",
            "Median trials per subject",
            "Maximum trials per subject",
            "Difficulty mean",
            "Difficulty SD",
            "Difficulty minimum",
            "Difficulty maximum",
        ],
        "Value": [
            len(df),
            df["subject"].nunique(),
            df["trial"].nunique(),
            trial_counts.min(),
            trial_counts.median(),
            trial_counts.max(),
            df["difficulty"].mean(),
            df["difficulty"].std(ddof=1),
            df["difficulty"].min(),
            df["difficulty"].max(),
        ],
    })

    save_csv(qc_summary, output_dir, "Table_S3_3_QC_summary.csv")
    save_csv(subject_features, output_dir, "Table_S3_3_subject_six_features_full.csv")
    save_csv(
        pd.concat([subject_features[["subject"]], analysis_features], axis=1),
        output_dir,
        "Table_S3_3_subject_six_features_standardized.csv",
    )

    # --------------------------
    # Tables: feature definition and descriptive stats
    # --------------------------
    feature_definition = make_feature_definition_table()
    descriptive_table = make_descriptive_table(analysis_features)

    save_csv(feature_definition, output_dir, "Table_3_3_1_feature_definitions.csv")
    save_csv(descriptive_table, output_dir, "Table_S3_3_feature_descriptive_statistics.csv")

    # --------------------------
    # Correlation and redundancy
    # --------------------------
    corr_matrix, p_matrix = corr_with_pvalues(analysis_features, feature_labels)
    corr_long = make_correlation_long_table(corr_matrix, p_matrix)
    high_corr_pairs = find_high_corr_pairs(corr_matrix, HIGH_CORR_THRESHOLD)

    corr_matrix.to_csv(output_dir / "Table_S3_3_feature_correlation_matrix.csv", encoding="utf-8-sig")
    p_matrix.to_csv(output_dir / "Table_S3_3_feature_correlation_p_matrix.csv", encoding="utf-8-sig")
    save_csv(corr_long, output_dir, "Table_S3_3_feature_correlation_long.csv")
    save_csv(high_corr_pairs, output_dir, "Table_S3_3_high_correlation_redundancy_pairs.csv")

    # --------------------------
    # PCA adequacy checks
    # --------------------------
    x = analysis_features.values

    kmo_total, kmo_per_variable = kmo_test(x)
    bartlett_chi2, bartlett_df, bartlett_p = bartlett_sphericity_test(x)

    kmo_table = pd.DataFrame({
        "Feature": feature_labels,
        "KMO": kmo_per_variable,
    })

    adequacy_summary = pd.DataFrame([
        {
            "Test": "KMO overall",
            "Statistic": kmo_total,
            "df": np.nan,
            "p": np.nan,
            "Note": "Supplementary adequacy check; PCA interpretation should rely primarily on variance and loadings.",
        },
        {
            "Test": "Bartlett sphericity",
            "Statistic": bartlett_chi2,
            "df": bartlett_df,
            "p": bartlett_p,
            "Note": "Tests whether the correlation matrix differs from identity.",
        },
    ])

    save_csv(kmo_table, output_dir, "Table_S3_3_KMO_per_feature.csv")
    save_csv(adequacy_summary, output_dir, "Table_S3_3_KMO_Bartlett_summary.csv")

    # --------------------------
    # PCA
    # --------------------------
    x_scaled = StandardScaler().fit_transform(analysis_features)

    pca = PCA()
    scores = pca.fit_transform(x_scaled)

    eigenvalues = pca.explained_variance_
    explained_ratio = pca.explained_variance_ratio_
    cumulative_ratio = np.cumsum(explained_ratio)

    component_names = [f"PC{i + 1}" for i in range(len(feature_labels))]

    pca_summary = pd.DataFrame({
        "Component": component_names,
        "Eigenvalue": eigenvalues,
        "Explained variance ratio": explained_ratio,
        "Explained variance (%)": explained_ratio * 100,
        "Cumulative variance ratio": cumulative_ratio,
        "Cumulative variance (%)": cumulative_ratio * 100,
    })

    # PCA loadings = eigenvectors * sqrt(eigenvalue)
    loadings = pca.components_.T * np.sqrt(eigenvalues)

    loading_matrix = pd.DataFrame(
        loadings,
        index=feature_labels,
        columns=component_names,
    )

    score_df = pd.DataFrame(scores, columns=component_names)
    score_df.insert(0, "subject", subject_features["subject"].values)

    pca_interpretation = make_pca_interpretation_table(
        pca_summary=pca_summary,
        loading_matrix=loading_matrix,
        n_pc=N_PC_FOR_CLUSTERING,
    )

    save_csv(pca_summary, output_dir, "Table_3_3_2_PCA_variance_explained.csv")
    loading_matrix.to_csv(output_dir / "Table_3_3_3_PCA_loading_matrix.csv", encoding="utf-8-sig")
    save_csv(pca_interpretation, output_dir, "Table_3_3_4_PCA_interpretation_summary.csv")
    save_csv(score_df, output_dir, "Table_S3_3_PCA_component_scores.csv")

    decision_summary = pd.DataFrame([
        {
            "Criterion": "Number of PCs with eigenvalue > 1",
            "Value": int((eigenvalues > 1).sum()),
        },
        {
            "Criterion": "Cumulative variance of PC1 only (%)",
            "Value": cumulative_ratio[0] * 100,
        },
        {
            "Criterion": "Cumulative variance of first 2 PCs (%)",
            "Value": cumulative_ratio[1] * 100 if len(cumulative_ratio) >= 2 else np.nan,
        },
        {
            "Criterion": "Cumulative variance of first 3 PCs (%)",
            "Value": cumulative_ratio[2] * 100 if len(cumulative_ratio) >= 3 else np.nan,
        },
        {
            "Criterion": f"Cumulative variance of first {N_PC_FOR_CLUSTERING} PCs (%)",
            "Value": cumulative_ratio[N_PC_FOR_CLUSTERING - 1] * 100
            if len(cumulative_ratio) >= N_PC_FOR_CLUSTERING else np.nan,
        },
        {
            "Criterion": f"High-correlation pairs with |r| >= {HIGH_CORR_THRESHOLD}",
            "Value": len(high_corr_pairs),
        },
        {
            "Criterion": "KMO overall",
            "Value": kmo_total,
        },
        {
            "Criterion": "Bartlett p-value",
            "Value": bartlett_p,
        },
    ])

    save_csv(decision_summary, output_dir, "Table_S3_3_PCA_decision_summary.csv")

    # --------------------------
    # Figure 1
    # --------------------------
    plot_pca_diagnostics(
        pca_summary,
        loading_matrix,
        output_dir,
        N_PC_FOR_CLUSTERING,
    )

    # --------------------------
    # Text report
    # --------------------------
    report_path = output_dir / "Section3_3_analysis_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Section 3.3: Quantitative representation system for VPC visual memory processing\n")
        f.write("=" * 90 + "\n\n")

        f.write("1. QC summary\n")
        f.write(qc_summary.to_string(index=False))
        f.write("\n\n")

        f.write("2. Feature definitions\n")
        f.write(feature_definition.to_string(index=False))
        f.write("\n\n")

        f.write("3. Six-feature descriptive statistics\n")
        f.write(descriptive_table.round(6).to_string(index=False))
        f.write("\n\n")

        f.write("4. Correlation matrix\n")
        f.write(corr_matrix.round(6).to_string())
        f.write("\n\n")

        f.write("5. High-correlation redundancy check\n")
        if len(high_corr_pairs) == 0:
            f.write(f"No feature pair reached |r| >= {HIGH_CORR_THRESHOLD}.\n")
        else:
            f.write(high_corr_pairs.round(6).to_string(index=False))
        f.write("\n\n")

        f.write("6. PCA adequacy checks\n")
        f.write(f"KMO overall = {kmo_total:.6f}\n")
        f.write(kmo_table.round(6).to_string(index=False))
        f.write("\n\n")
        f.write(f"Bartlett: chi2({int(bartlett_df)}) = {bartlett_chi2:.6f}, p = {bartlett_p:.8g}\n")
        f.write("\n\n")

        f.write("7. PCA variance explained\n")
        f.write(pca_summary.round(6).to_string(index=False))
        f.write("\n\n")

        f.write("8. PCA loading matrix\n")
        f.write(loading_matrix.round(6).to_string())
        f.write("\n\n")

        f.write("9. PCA interpretation summary\n")
        f.write(pca_interpretation.round(6).to_string(index=False))
        f.write("\n\n")

        f.write("10. Interpretation caution\n")
        f.write(
            "If PC1 explains about 30% and PC1+PC2 explains about 50%, avoid writing that "
            "'the first two PCs explain 30%'. That value is PC1 alone. Also avoid claiming that "
            "PCA 'proves' no redundancy. A more precise phrasing is: the gradual variance decay, "
            "low number of high-correlation pairs, and meaningful loading pattern support a "
            "multidimensional feature structure.\n"
        )

    # --------------------------
    # Zip output
    # --------------------------
    zip_path = Path(ZIP_NAME)
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(output_dir):
            for file in files:
                full_path = Path(root) / file
                arcname = full_path.relative_to(output_dir)
                zf.write(full_path, arcname=arcname)

    print("\n" + "=" * 90)
    print("Section 3.3 analysis finished.")
    print("=" * 90)
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")
    print(f"Zip file: {zip_path}")
    print("\nMain manuscript tables:")
    print("1. Table_3_3_1_feature_definitions.csv")
    print("2. Table_3_3_2_PCA_variance_explained.csv")
    print("3. Table_3_3_3_PCA_loading_matrix.csv")
    print("4. Table_3_3_4_PCA_interpretation_summary.csv")
    print("\nFigure:")
    print("1. Figure1_PCA_diagnostics.png/pdf")
    print("\nPCA decision summary:")
    print(decision_summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Section 3.4 analysis:
classification and interpretability based on quantitative VPC representations.

Input:
    Chapter3_Output.zip
    containing:
    - Table_S3_3_subject_six_features_standardized.csv
    - Table_S3_3_PCA_component_scores.csv

The upstream Excel schema is not modified. This script operates only on the
exported Section 3.3 tables.

Outputs include clustering diagnostics tables, subtype statistics,
membership probabilities, bootstrap stability results, Figure 1,
a text report, and a ZIP archive.
"""

import zipfile
import shutil
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
)

warnings.filterwarnings("ignore")

# ============================================================
# Lightweight deterministic K-means implementation
# ============================================================

def kmeans_pp_init(X, k, rng):
    n = X.shape[0]
    centers = np.empty((k, X.shape[1]), dtype=float)
    first = rng.integers(n)
    centers[0] = X[first]
    closest_dist_sq = np.sum((X - centers[0]) ** 2, axis=1)

    for c in range(1, k):
        total = closest_dist_sq.sum()
        if total <= 1e-12:
            idx = rng.integers(n)
        else:
            idx = rng.choice(n, p=closest_dist_sq / total)
        centers[c] = X[idx]
        dist_sq = np.sum((X[:, None, :] - centers[None, :c + 1, :]) ** 2, axis=2)
        closest_dist_sq = dist_sq.min(axis=1)
    return centers


def simple_kmeans(X, k, n_init=25, max_iter=100, random_state=42):
    rng_master = np.random.default_rng(random_state)
    best = None

    for _ in range(n_init):
        rng = np.random.default_rng(int(rng_master.integers(0, 2**32 - 1)))
        centers = kmeans_pp_init(X, k, rng)

        for _ in range(max_iter):
            dist_sq = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            labels = dist_sq.argmin(axis=1)
            new_centers = centers.copy()

            for c in range(k):
                if np.any(labels == c):
                    new_centers[c] = X[labels == c].mean(axis=0)
                else:
                    new_centers[c] = X[rng.integers(X.shape[0])]

            if np.allclose(new_centers, centers, atol=1e-8):
                centers = new_centers
                break
            centers = new_centers

        dist_sq = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = dist_sq.argmin(axis=1)
        inertia = dist_sq[np.arange(X.shape[0]), labels].sum()

        if best is None or inertia < best["inertia"]:
            best = {
                "centers": centers.copy(),
                "labels": labels.copy(),
                "inertia": float(inertia),
            }

    return best


def assign_to_centers(X, centers):
    dist_sq = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    return dist_sq.argmin(axis=1)


def kmeans_membership_probabilities(X, centers, sigma2=None, eps=1e-12):
    """
    Convert distances to K-means centers into soft membership probabilities.

    Notes
    -----
    Standard K-means produces hard labels rather than probabilities. Here,
    probabilities are computed with an isotropic Gaussian-softmax rule:

        p(cluster c | x) ∝ exp(-||x - center_c||^2 / (2 * sigma2))

    sigma2 is estimated from the final K-means within-cluster SSE when provided.
    Therefore, these values should be interpreted as distance-based membership
    probabilities, not as model-calibrated clinical/statistical probabilities.
    """
    dist_sq = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)

    if sigma2 is None:
        nearest_dist_sq = dist_sq.min(axis=1)
        sigma2 = nearest_dist_sq.mean() / max(X.shape[1], 1)

    sigma2 = max(float(sigma2), eps)

    logits = -dist_sq / (2.0 * sigma2)
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / probs.sum(axis=1, keepdims=True)

    return probs


# ============================================================
# Configuration
# ============================================================

INPUT_ZIP = Path("Chapter3_Output.zip")
OUTPUT_DIR = Path("Chapter4_Output")
ZIP_OUTPUT = Path("Chapter4_Output.zip")
FIGURE1_BASENAME = "Figure1_cluster_subtype_profiles"

# ============================================================
# Reproducibility controls
# ============================================================
# Fixed seed used throughout clustering so repeated runs are reproducible.
RANDOM_SEED = 20260829

# Increase K-means restarts to reduce sensitivity to a particular initialization.
KMEANS_N_INIT = 200
BOOTSTRAP_KMEANS_N_INIT = 50

# Row ordering must not depend on subject identifiers.  Rounding also removes
# irrelevant machine-precision differences between otherwise equivalent files.
ROW_SORT_DECIMALS = 12


# ============================================================
# Figure utilities
# ============================================================

def set_publication_style():
    """Apply the publication-oriented style used by the retained figure."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 17,
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 15,
        "figure.dpi": 200,
        "savefig.dpi": 900,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def clean_axes(ax):
    """Remove top and right spines for publication-style Cartesian axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig, output_dir, basename):
    """Save the figure as 900-dpi PNG and vector PDF for publication-quality output."""
    fig.savefig(
        output_dir / f"{basename}.png",
        bbox_inches="tight",
        dpi=900,
        facecolor="white",
    )
    fig.savefig(
        output_dir / f"{basename}.pdf",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def plot_figure1(
    cluster_quality,
    k_final,
    high_df,
    final_df,
    feature_cols,
    pc_cols,
    output_dir,
):
    """Generate Figure 1: cluster diagnostics and subtype profiles."""
    diag = cluster_quality.copy()
    k_values = diag["K"].to_numpy()

    high_subtype_order_en = [
        "Representation-maintenance",
        "Stability-retrieval compensation",
        "Retrieval-inefficient stability compensation",
    ]
    high_subtype_display_labels = [
        "memory-maintenance",
        "retrieval-compensation",
        "slow-retrieval",
    ]

    # Combined APA-style figure:
    # A: cluster-number diagnostics
    # B: subtype profiles for the three clusters obtained after screening
    # C: mean PCA component scores for all four final subtypes
    fig = plt.figure(figsize=(16.2, 11.8), facecolor="white")
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        height_ratios=[1.0, 1.30],
        width_ratios=[1.30, 1.0],
        hspace=0.42,
        wspace=0.42,
    )

    diagnostic_color = "#2F6B93"
    reference_color = "#6E6E6E"
    grid_color = "#D9D9D9"

    panel_fontsize = 24
    title_fontsize = 18
    label_fontsize = 19
    tick_fontsize = 17

    # Panel A: cluster-number diagnostics.
    gs_a = gs[0, 0].subgridspec(1, 2, wspace=0.72)
    ax_a1 = fig.add_subplot(gs_a[0, 0])
    ax_a2 = fig.add_subplot(gs_a[0, 1])

    diagnostic_specs = [
        (ax_a1, "SSE", "SSE", "SSE"),
        (ax_a2, "Calinski-Harabasz", "CH index", "Calinski-Harabasz index"),
    ]

    for ax_a, metric_col, y_label, title in diagnostic_specs:
        ax_a.plot(
            k_values,
            diag[metric_col],
            color=diagnostic_color,
            marker="o",
            markersize=6.5,
            linewidth=2.2,
            zorder=3,
        )
        ax_a.axvline(
            k_final,
            color=reference_color,
            linestyle="--",
            linewidth=1.5,
            zorder=2,
        )
        ax_a.set_xticks(k_values)
        ax_a.set_xlabel("Number of clusters,\nK", fontsize=label_fontsize, labelpad=6)
        ax_a.set_ylabel(y_label, fontsize=label_fontsize)
        ax_a.tick_params(axis="both", labelsize=tick_fontsize, direction="out")
        ax_a.grid(
            axis="y",
            color=grid_color,
            linewidth=0.6,
            alpha=0.65,
            zorder=0,
        )
        clean_axes(ax_a)

    ax_a1.text(
        k_final + 0.10,
        ax_a1.get_ylim()[1] - 0.06 * np.ptp(ax_a1.get_ylim()),
        f"Selected K = {k_final}",
        color=reference_color,
        fontsize=15.0,
        ha="left",
        va="top",
    )
    ax_a1.text(
        -0.28,
        1.12,
        "A",
        transform=ax_a1.transAxes,
        fontsize=panel_fontsize,
        fontweight="bold",
        ha="left",
        va="top",
    )

    # Panel B: normalized subtype profiles.
    ax_b = fig.add_subplot(gs[0, 1], polar=True)

    centers_z_high = (
        high_df.groupby("subtype_en", observed=False)[feature_cols]
        .mean()
        .reindex(high_subtype_order_en)
    )
    min_vals_high = high_df[feature_cols].min()
    max_vals_high = high_df[feature_cols].max()
    denom_high = (max_vals_high - min_vals_high).replace(0, np.nan)
    centers_norm_high = ((centers_z_high - min_vals_high) / denom_high).fillna(0.5)

    radar_labels = [
        "MRA",
        "CPN",
        "MRL",
        "MRA slope",
        "CPN slope",
        "MRL slope",
    ]
    angles = np.linspace(0, 2 * np.pi, len(radar_labels), endpoint=False).tolist()
    angles += angles[:1]

    radar_colors = ["#2F6B93", "#D97732", "#4F8A6B"]
    radar_linestyles = ["-", "--", ":"]
    radar_markers = ["o", "s", "^"]

    for group, label, color, linestyle, marker in zip(
        high_subtype_order_en,
        high_subtype_display_labels,
        radar_colors,
        radar_linestyles,
        radar_markers,
    ):
        values = centers_norm_high.loc[group].tolist()
        values += values[:1]
        ax_b.plot(
            angles,
            values,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=5.8,
            linewidth=2.4,
            label=label,
            zorder=3,
        )
        ax_b.fill(angles, values, color=color, alpha=0.035, zorder=1)

    ax_b.set_xticks(angles[:-1])
    ax_b.set_xticklabels(radar_labels, fontsize=18.0)
    ax_b.tick_params(axis="x", pad=12)
    ax_b.set_ylim(0, 1)
    ax_b.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax_b.set_yticklabels([".25", ".50", ".75", "1.00"], fontsize=15)
    ax_b.set_rlabel_position(16)
    ax_b.grid(color=grid_color, linewidth=0.6, alpha=0.75)
    ax_b.spines["polar"].set_color("#A8A8A8")
    ax_b.spines["polar"].set_linewidth(0.7)

    ax_b.legend(
        loc="upper left",
        bbox_to_anchor=(1.36, 1.05),
        frameon=False,
        borderaxespad=0.0,
        handlelength=3.0,
        handletextpad=0.8,
        labelspacing=0.85,
        fontsize=15.0,
    )

    # Abbreviation legend for the radar-chart dimensions.
    # Positioned in the dedicated right-side margin to avoid overlap with radar labels.
    ax_b.text(
        1.36,
        0.50,
        "MRA = Memory Representation Ability\n"
        "CPN = Cognitive Processing Noise Level\n"
        "MRL = Memory Retrieval Latency",
        transform=ax_b.transAxes,
        fontsize=14.5,
        ha="left",
        va="top",
        linespacing=1.50,
        clip_on=False,
    )
    ax_b.text(
        -0.19,
        1.12,
        "B",
        transform=ax_b.transAxes,
        fontsize=panel_fontsize,
        fontweight="bold",
        ha="left",
        va="top",
    )

    # Panel C: mean PCA component scores for all final subtypes.
    ax_c = fig.add_subplot(gs[1, :])

    combined_heatmap_order_en = [
        "Low representation",
        "Representation-maintenance",
        "Stability-retrieval compensation",
        "Retrieval-inefficient stability compensation",
    ]
    combined_heatmap_labels = [
        "Low-representation",
        "memory-maintenance",
        "retrieval-compensation",
        "slow-retrieval",
    ]

    center_for_combined_heatmap = (
        final_df.groupby("subtype_en", observed=False)[pc_cols]
        .mean()
        .reindex(combined_heatmap_order_en)
    )

    heatmap_values = center_for_combined_heatmap.to_numpy(dtype=float)
    finite_values = heatmap_values[np.isfinite(heatmap_values)]
    color_limit = np.max(np.abs(finite_values)) if finite_values.size else 1.0
    color_limit = max(float(color_limit), 1e-6)

    # Match the clearer heatmap treatment used in Figure1_PCA_diagnostics
    # Panel B: preserve square cells and avoid unnecessary raster resampling.
    im_c = ax_c.imshow(
        heatmap_values,
        cmap="RdBu_r",
        vmin=-color_limit,
        vmax=color_limit,
        aspect="equal",
        interpolation="nearest",
        resample=False,
    )

    ax_c.set_xticks(np.arange(len(pc_cols)))
    ax_c.set_xticklabels(pc_cols, fontsize=18)
    ax_c.set_yticks(np.arange(len(combined_heatmap_order_en)))
    ax_c.set_yticklabels(combined_heatmap_labels, fontsize=18)
    ax_c.tick_params(axis="both", which="major", length=0)
    ax_c.tick_params(axis="y", pad=7)
    ax_c.set_xlabel("Principal component", fontsize=label_fontsize, labelpad=7)

    # Thin white separators improve cell tracking while keeping the heatmap crisp.
    ax_c.set_xticks(np.arange(-0.5, heatmap_values.shape[1], 1), minor=True)
    ax_c.set_yticks(np.arange(-0.5, heatmap_values.shape[0], 1), minor=True)
    ax_c.grid(which="minor", color="white", linestyle="-", linewidth=0.7)
    ax_c.tick_params(which="minor", bottom=False, left=False)

    for i in range(heatmap_values.shape[0]):
        for j in range(heatmap_values.shape[1]):
            value = heatmap_values[i, j]
            if not np.isfinite(value):
                label = "NA"
                text_color = "#333333"
            else:
                label = f"{value:.2f}"
                # Use the same high-contrast logic as the PCA-loading heatmap.
                text_color = "white" if abs(value) / color_limit >= 0.62 else "black"

            ax_c.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=16.0,
                color=text_color,
                fontweight="normal",
            )

    for spine in ax_c.spines.values():
        spine.set_visible(False)

    cbar_c = fig.colorbar(
        im_c,
        ax=ax_c,
        fraction=0.025,
        pad=0.025,
        aspect=24,
    )
    cbar_c.set_label("Mean PC score", fontsize=17)
    cbar_c.ax.tick_params(labelsize=15, width=0.9)
    cbar_c.outline.set_linewidth(0.6)

    ax_c.text(
        -0.055,
        1.10,
        "C",
        transform=ax_c.transAxes,
        fontsize=panel_fontsize,
        fontweight="bold",
        ha="left",
        va="top",
    )

    fig.subplots_adjust(left=0.14, right=0.74, top=0.95, bottom=0.10)

    # ------------------------------------------------------------------
    # Visually center Panel C as a complete unit.
    #
    # Simply centering ax_c itself is not sufficient because the long
    # subtype labels extend substantially to the left, while the colorbar
    # and its label extend to the right.  Here we first assign the desired
    # heatmap/colorbar geometry, render the figure once, measure the actual
    # tight bounding boxes of the full Panel C content, and then shift both
    # axes together so that the complete visual footprint is centered on
    # the figure.
    # ------------------------------------------------------------------
    bbox_c = ax_c.get_position()

    # For an equal-aspect image, derive the axes width from its height so that
    # each heatmap cell is physically square. This prevents the 4 x 5 matrix
    # from being stretched across the full lower panel.
    n_rows, n_cols = heatmap_values.shape
    figure_aspect = fig.get_figheight() / fig.get_figwidth()
    square_cell_width = bbox_c.height * figure_aspect * (n_cols / n_rows)
    heatmap_width = min(0.54, square_cell_width)

    initial_left = 0.5 - heatmap_width / 2
    ax_c.set_position([
        initial_left,
        bbox_c.y0,
        heatmap_width,
        bbox_c.height,
    ])

    cbar_width = 0.012
    cbar_gap = 0.014
    cbar_c.ax.set_position([
        initial_left + heatmap_width + cbar_gap,
        bbox_c.y0,
        cbar_width,
        bbox_c.height,
    ])

    # Draw once so Matplotlib knows the true extents of tick labels,
    # axis labels, panel letter C, and the colorbar label.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    heatmap_tight = ax_c.get_tightbbox(renderer)
    colorbar_tight = cbar_c.ax.get_tightbbox(renderer)

    visual_left_px = min(heatmap_tight.x0, colorbar_tight.x0)
    visual_right_px = max(heatmap_tight.x1, colorbar_tight.x1)
    visual_center_px = 0.5 * (visual_left_px + visual_right_px)
    figure_center_px = 0.5 * fig.bbox.width

    # Convert the required horizontal pixel shift to figure coordinates.
    shift_fig = (figure_center_px - visual_center_px) / fig.bbox.width

    heatmap_pos = ax_c.get_position()
    ax_c.set_position([
        heatmap_pos.x0 + shift_fig,
        heatmap_pos.y0,
        heatmap_pos.width,
        heatmap_pos.height,
    ])

    colorbar_pos = cbar_c.ax.get_position()
    cbar_c.ax.set_position([
        colorbar_pos.x0 + shift_fig,
        colorbar_pos.y0,
        colorbar_pos.width,
        colorbar_pos.height,
    ])
    save_figure(fig, output_dir, FIGURE1_BASENAME)


# ============================================================
# Main analysis
# ============================================================

def main():
    zip_33 = INPUT_ZIP
    out_dir = OUTPUT_DIR

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Extract Section 3.3 files to a temporary directory only.
    # This prevents "section3_3_extracted" from being saved in the output folder or zip.
    with tempfile.TemporaryDirectory(prefix="section3_3_extracted_") as tmp_extract:
        extract_dir = Path(tmp_extract)

        with zipfile.ZipFile(zip_33, "r") as zf:
            zf.extractall(extract_dir)

        features = pd.read_csv(extract_dir / "Table_S3_3_subject_six_features_standardized.csv")
        pcs = pd.read_csv(extract_dir / "Table_S3_3_PCA_component_scores.csv")

    data = features.merge(pcs, on="subject", how="inner")

    feature_cols = [
        "Memory Representation Ability",
        "Cognitive Processing Noise Level",
        "Memory Retrieval Latency",
        "Memory Representation Ability slope",
        "Cognitive Processing Noise slope",
        "Memory Retrieval Latency slope"
    ]
    pc_cols = ["PC1", "PC2", "PC3", "PC4", "PC5"]

    # ------------------------------------------------------------------
    # Deterministic row order independent of subject identifiers
    # ------------------------------------------------------------------
    # Name-based and anonymized files contain the same measurements but can
    # arrive in different row orders because their subject labels differ.
    # K-means++ samples row indices, so a different row order can otherwise
    # produce a different initialization even when pairwise distances are
    # mathematically identical.
    #
    # Primary keys: the six standardized features (sign-stable across PCA runs).
    # Secondary keys: absolute PC scores, which are invariant to whole-axis
    # PCA sign flips such as PC1 -> -PC1 or PC4 -> -PC4.
    # Values are rounded only for sorting, not for any statistical analysis.
    sort_key_cols = []

    for idx, col in enumerate(feature_cols):
        key = f"__sort_feature_{idx}"
        data[key] = pd.to_numeric(data[col], errors="coerce").round(ROW_SORT_DECIMALS)
        sort_key_cols.append(key)

    for idx, col in enumerate(pc_cols):
        key = f"__sort_abs_pc_{idx}"
        data[key] = (
            pd.to_numeric(data[col], errors="coerce")
            .abs()
            .round(ROW_SORT_DECIMALS)
        )
        sort_key_cols.append(key)

    data = (
        data.sort_values(sort_key_cols, kind="mergesort")
        .drop(columns=sort_key_cols)
        .reset_index(drop=True)
    )

    low_threshold = -1.0
    data["screen_group"] = np.where(
        data["Memory Representation Ability"] < low_threshold,
        "Low representation",
        "High representation",
    )

    low_df = data[data["screen_group"] == "Low representation"].copy()
    high_df = data[data["screen_group"] == "High representation"].copy()
    X_high = high_df[pc_cols].to_numpy()

    # ------------------------------------------------------------
    # K diagnostics
    # ------------------------------------------------------------
    k_rows = []
    previous_sse = None
    models = {}

    for k in range(2, 9):
        km = simple_kmeans(
            X_high,
            k,
            n_init=KMEANS_N_INIT,
            random_state=RANDOM_SEED + k,
        )
        labels = km["labels"]
        sse = km["inertia"]
        sse_drop = np.nan if previous_sse is None else (previous_sse - sse) / previous_sse * 100
        previous_sse = sse

        k_rows.append({
            "K": k,
            "SSE": sse,
            "SSE decrease from previous K (%)": sse_drop,
            "Calinski-Harabasz": calinski_harabasz_score(X_high, labels),
            "Silhouette": silhouette_score(X_high, labels),
            "Davies-Bouldin": davies_bouldin_score(X_high, labels),
            "Cluster sizes": "; ".join(map(str, np.bincount(labels))),
        })
        models[k] = km

    cluster_quality = pd.DataFrame(k_rows)
    cluster_quality.to_csv(
        out_dir / "Table_4_1_cluster_number_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------
    # Final clustering: high-representation group, K = 3
    # ------------------------------------------------------------
    k_final = 3
    final_km = models[k_final]
    high_df["raw_cluster"] = final_km["labels"]

    raw_centers = high_df.groupby("raw_cluster")[feature_cols].mean()
    retrieval_inefficient_raw = raw_centers["Memory Retrieval Latency"].idxmin()
    representation_compromised_raw = raw_centers.drop(index=retrieval_inefficient_raw)["Memory Representation Ability slope"].idxmin()
    representation_maintenance_raw = [
        c for c in raw_centers.index
        if c not in [retrieval_inefficient_raw, representation_compromised_raw]
    ][0]

    raw_to_name = {
        representation_maintenance_raw: "Representation-maintenance",
        representation_compromised_raw: "Stability-retrieval compensation",
        retrieval_inefficient_raw: "Retrieval-inefficient stability compensation",
    }
    raw_to_cn = {
        representation_maintenance_raw: "高表征—记忆维持型",
        representation_compromised_raw: "中等表征—提取补偿型",
        retrieval_inefficient_raw: "中等表征—慢速提取型",
    }

    high_df["subtype_en"] = high_df["raw_cluster"].map(raw_to_name)
    high_df["subtype_cn"] = high_df["raw_cluster"].map(raw_to_cn)

    low_df["raw_cluster"] = -1
    low_df["subtype_en"] = "Low representation"
    low_df["subtype_cn"] = "低表征能力组"

    final_df = pd.concat([low_df, high_df], ignore_index=True)

    subtype_order_cn = [
        "低表征能力组",
        "高表征—记忆维持型",
        "中等表征—提取补偿型",
        "中等表征—慢速提取型",
    ]
    subtype_order_en = [
        "Low representation",
        "Representation-maintenance",
        "Stability-retrieval compensation",
        "Retrieval-inefficient stability compensation",
    ]

    final_df["subtype_cn"] = pd.Categorical(final_df["subtype_cn"], categories=subtype_order_cn, ordered=True)
    final_df["subtype_en"] = pd.Categorical(final_df["subtype_en"], categories=subtype_order_en, ordered=True)

    # ------------------------------------------------------------
    # Subject-level membership probabilities
    # ------------------------------------------------------------
    # Important:
    # Standard K-means gives hard labels rather than statistical posterior
    # probabilities. In the original two-step classification, the low-
    # representation group was produced by a deterministic threshold rule,
    # whereas the three high-representation subtypes were obtained by K-means.
    #
    # To make the four subtype membership scores comparable, this version uses
    # a unified four-center Gaussian-softmax rule:
    #
    #   1) The final hard labels are kept unchanged:
    #      - Low representation: Mean representation < -1.0
    #      - Remaining participants: K = 3 K-means in PC1-PC5 space
    #
    #   2) Four subtype centers are defined in the same PC1-PC5 space:
    #      - Low representation center: mean PC score vector of the screened
    #        low-representation participants
    #      - Three high-representation centers: final K-means centers
    #
    #   3) Every participant, including screened-low participants, receives
    #      four distance-based softmax membership probabilities calculated
    #      from distances to the same four centers.
    #
    # Therefore, the output probabilities should be interpreted as
    # distance-based soft membership scores, not model-calibrated clinical or
    # statistical posterior probabilities.
    prob_col_map = {
        "Low representation": "prob_low_representation",
        "Representation-maintenance": "prob_representation_maintenance",
        "Stability-retrieval compensation": "prob_stability_retrieval_compensation",
        "Retrieval-inefficient stability compensation": "prob_retrieval_inefficient_stability_compensation",
    }
    prob_cols = [prob_col_map[name] for name in subtype_order_en]

    # Define four centers in the same PC space.
    # The low-representation center is not produced by K-means; it is the
    # empirical centroid of the screened-low participants in PC1-PC5 space.
    if len(low_df) == 0:
        raise ValueError(
            "No low-representation participants were found. "
            "Cannot compute the low-representation four-center softmax center."
        )

    low_center_pc = low_df[pc_cols].to_numpy().mean(axis=0)

    centers_by_subtype_en = {
        "Low representation": low_center_pc,
    }
    for raw_idx in range(k_final):
        centers_by_subtype_en[raw_to_name[raw_idx]] = final_km["centers"][raw_idx]

    four_centers = np.vstack([
        centers_by_subtype_en[name]
        for name in subtype_order_en
    ])

    # Estimate one common sigma2 value from the within-subtype squared error
    # under the final hard classification. This places the four probability
    # columns on the same temperature/scale.
    X_low = low_df[pc_cols].to_numpy()
    low_sse = float(np.sum((X_low - low_center_pc) ** 2))
    high_sse = float(final_km["inertia"])
    sigma2_four = (low_sse + high_sse) / max(len(final_df) * len(pc_cols), 1)
    sigma2_four = max(float(sigma2_four), 1e-12)

    X_all = final_df[pc_cols].to_numpy()
    four_probs = kmeans_membership_probabilities(
        X_all,
        four_centers,
        sigma2=sigma2_four,
    )

    probability_df = final_df[
        ["subject", "screen_group", "raw_cluster", "subtype_en", "subtype_cn"]
    ].copy()

    for col_idx, subtype_name in enumerate(subtype_order_en):
        probability_df[prob_col_map[subtype_name]] = four_probs[:, col_idx]

    probability_df["assigned_group_probability"] = probability_df.apply(
        lambda row: row[prob_col_map[str(row["subtype_en"])]],
        axis=1,
    )
    probability_df["max_group_probability"] = probability_df[prob_cols].max(axis=1)
    probability_df["max_probability_group_en"] = [
        subtype_order_en[int(idx)]
        for idx in np.argmax(four_probs, axis=1)
    ]

    max_group_cn_map = dict(zip(subtype_order_en, subtype_order_cn))
    probability_df["max_probability_group_cn"] = probability_df[
        "max_probability_group_en"
    ].map(max_group_cn_map)

    probability_df["probability_method"] = (
        "Unified four-center Gaussian-softmax of squared Euclidean distances "
        "in PC1-PC5 space. Hard labels remain two-step: low-representation "
        "threshold screening followed by K=3 K-means among high-representation "
        "participants."
    )
    probability_df["softmax_sigma2"] = sigma2_four

    # Save the four centers used for the unified softmax probabilities so the
    # probability calculation is transparent and reproducible.
    four_center_df = pd.DataFrame(four_centers, columns=pc_cols)
    four_center_df.insert(0, "subtype_en", subtype_order_en)
    four_center_df.insert(1, "subtype_cn", subtype_order_cn)
    four_center_df["center_method"] = [
        "Empirical centroid of screened-low participants in PC1-PC5 space",
        "Final K-means center among high-representation participants",
        "Final K-means center among high-representation participants",
        "Final K-means center among high-representation participants",
    ]
    four_center_df["softmax_sigma2"] = sigma2_four
    four_center_df.to_csv(
        out_dir / "Table_S4_four_center_softmax_centers.csv",
        index=False,
        encoding="utf-8-sig",
    )

    probability_df.sort_values(["subtype_cn", "subject"]).to_csv(
        out_dir / "Table_S4_subject_group_membership_probabilities.csv",
        index=False,
        encoding="utf-8-sig",
    )

    final_df.sort_values(["subtype_cn", "subject"]).to_csv(
        out_dir / "Table_S4_subject_classification_and_features.csv",
        index=False,
        encoding="utf-8-sig",
    )

    count_table = final_df["subtype_cn"].value_counts().reindex(subtype_order_cn).reset_index()
    count_table.columns = ["Subtype", "N"]
    count_table["Percent"] = count_table["N"] / len(final_df) * 100
    count_table.to_csv(out_dir / "Table_4_0_subtype_counts.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------
    # Low vs high screening summary
    # ------------------------------------------------------------
    def mean_sd(x):
        return f"{x.mean():.3f} ± {x.std(ddof=1):.3f}"

    screen_rows = []
    for feat in feature_cols:
        low = low_df[feat].dropna()
        high = high_df[feat].dropna()
        t = stats.ttest_ind(low, high, equal_var=False)
        diff = low.mean() - high.mean()
        se = np.sqrt(low.var(ddof=1) / len(low) + high.var(ddof=1) / len(high))
        df_welch = (low.var(ddof=1) / len(low) + high.var(ddof=1) / len(high)) ** 2 / (
            (low.var(ddof=1) / len(low)) ** 2 / (len(low) - 1)
            + (high.var(ddof=1) / len(high)) ** 2 / (len(high) - 1)
        )
        ci = stats.t.interval(0.95, df_welch, loc=diff, scale=se)
        pooled = np.sqrt(
            ((len(low) - 1) * low.var(ddof=1) + (len(high) - 1) * high.var(ddof=1))
            / (len(low) + len(high) - 2)
        )
        d = diff / pooled if pooled > 0 else np.nan

        screen_rows.append({
            "Feature": feat,
            "Low representation M±SD": mean_sd(low),
            "High representation M±SD": mean_sd(high),
            "Mean difference": diff,
            "95% CI lower": ci[0],
            "95% CI upper": ci[1],
            "Welch t": t.statistic,
            "df": df_welch,
            "p": t.pvalue,
            "Cohen d": d,
        })

    screen_table = pd.DataFrame(screen_rows)
    screen_table["p_FDR"] = multipletests(screen_table["p"], method="fdr_bh")[1]
    screen_table.to_csv(out_dir / "Table_4_2_low_vs_high_screening_summary.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------
    # Subtype centers and inferential tests
    # ------------------------------------------------------------
    center_table = final_df.groupby("subtype_cn", observed=False)[feature_cols].agg(["mean", "std", "count"])
    center_table.to_csv(out_dir / "Table_4_3_final_subtype_feature_centers_full.csv", encoding="utf-8-sig")

    center_mean = final_df.groupby("subtype_cn", observed=False)[feature_cols].mean().reset_index()
    center_mean.to_csv(out_dir / "Table_4_4_final_subtype_feature_centers_mean.csv", index=False, encoding="utf-8-sig")

    anova_rows = []
    for feat in feature_cols:
        groups = [final_df.loc[final_df["subtype_cn"] == g, feat].dropna() for g in subtype_order_cn]
        f_val, p_val = stats.f_oneway(*groups)
        h_val, p_kw = stats.kruskal(*groups)

        all_values = final_df[feat].dropna()
        grand_mean = all_values.mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
        ss_total = sum((all_values - grand_mean) ** 2)
        eta_sq = ss_between / ss_total if ss_total > 0 else np.nan

        anova_rows.append({
            "Feature": feat,
            "F": f_val,
            "df_between": len(groups) - 1,
            "df_within": len(all_values) - len(groups),
            "p_ANOVA": p_val,
            "eta_squared": eta_sq,
            "Kruskal_H": h_val,
            "p_Kruskal": p_kw,
        })

    anova_table = pd.DataFrame(anova_rows)
    anova_table["p_ANOVA_FDR"] = multipletests(anova_table["p_ANOVA"], method="fdr_bh")[1]
    anova_table["p_Kruskal_FDR"] = multipletests(anova_table["p_Kruskal"], method="fdr_bh")[1]
    anova_table.to_csv(out_dir / "Table_4_5_subtype_ANOVA_Kruskal.csv", index=False, encoding="utf-8-sig")

    posthoc_rows = []
    for feat in feature_cols:
        pairs = []
        for i in range(len(subtype_order_cn)):
            for j in range(i + 1, len(subtype_order_cn)):
                g1, g2 = subtype_order_cn[i], subtype_order_cn[j]
                x1 = final_df.loc[final_df["subtype_cn"] == g1, feat].dropna()
                x2 = final_df.loc[final_df["subtype_cn"] == g2, feat].dropna()
                t_val = stats.ttest_ind(x1, x2, equal_var=False)
                diff = x1.mean() - x2.mean()
                pooled = np.sqrt(
                    ((len(x1) - 1) * x1.var(ddof=1) + (len(x2) - 1) * x2.var(ddof=1))
                    / (len(x1) + len(x2) - 2)
                )
                d = diff / pooled if pooled > 0 else np.nan
                pairs.append({
                    "Feature": feat,
                    "Group 1": g1,
                    "Group 2": g2,
                    "Mean 1": x1.mean(),
                    "Mean 2": x2.mean(),
                    "Mean difference G1-G2": diff,
                    "Welch t": t_val.statistic,
                    "p_raw": t_val.pvalue,
                    "Cohen d": d,
                })

        pair_df = pd.DataFrame(pairs)
        pair_df["p_Bonferroni_within_feature"] = np.minimum(pair_df["p_raw"] * len(pair_df), 1.0)
        posthoc_rows.append(pair_df)

    posthoc_table = pd.concat(posthoc_rows, ignore_index=True)
    posthoc_table.to_csv(out_dir / "Table_4_6_pairwise_posthoc_Bonferroni.csv", index=False, encoding="utf-8-sig")
    posthoc_table[posthoc_table["p_Bonferroni_within_feature"] < 0.05].to_csv(
        out_dir / "Table_S4_significant_pairwise_posthoc.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------
    # Bootstrap stability
    # ------------------------------------------------------------
    B = 200
    rng = np.random.default_rng(RANDOM_SEED)
    original_labels = high_df["raw_cluster"].to_numpy()
    ari_values = []

    for _ in range(B):
        idx = rng.integers(0, len(high_df), size=len(high_df))
        X_boot = X_high[idx]
        km_boot = simple_kmeans(
            X_boot,
            k_final,
            n_init=BOOTSTRAP_KMEANS_N_INIT,
            random_state=int(rng.integers(0, 1_000_000)),
        )
        labels_pred_full = assign_to_centers(X_high, km_boot["centers"])
        ari_values.append(adjusted_rand_score(original_labels, labels_pred_full))

    ari_values = np.array(ari_values)
    bootstrap_summary = pd.DataFrame([{
        "Bootstrap iterations": B,
        "ARI mean": ari_values.mean(),
        "ARI SD": ari_values.std(ddof=1),
        "ARI median": np.median(ari_values),
        "ARI Q1": np.quantile(ari_values, 0.25),
        "ARI Q3": np.quantile(ari_values, 0.75),
        "ARI min": ari_values.min(),
        "ARI max": ari_values.max(),
        "Percent ARI >= 0.70": (ari_values >= 0.70).mean() * 100,
        "Percent ARI >= 0.50": (ari_values >= 0.50).mean() * 100,
        "Percent ARI < 0.10": (ari_values < 0.10).mean() * 100,
    }])
    bootstrap_summary.to_csv(out_dir / "Table_4_7_bootstrap_stability_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"Bootstrap": np.arange(1, B + 1), "ARI": ari_values}).to_csv(
        out_dir / "Table_S4_bootstrap_ARI_values.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------
    # Figure 1
    # ------------------------------------------------------------
    set_publication_style()
    plot_figure1(
        cluster_quality=cluster_quality,
        k_final=k_final,
        high_df=high_df,
        final_df=final_df,
        feature_cols=feature_cols,
        pc_cols=pc_cols,
        output_dir=out_dir,
    )

    report = []
    report.append("Section 3.4 clustering and interpretability analysis")
    report.append("=" * 72)
    report.append("")
    report.append(f"Total participants: {len(data)}")
    report.append(f"Low-representation threshold: Mean representation < {low_threshold}")
    report.append(f"Low-representation group: n = {len(low_df)} ({len(low_df)/len(data)*100:.2f}%)")
    report.append(f"High-representation group: n = {len(high_df)} ({len(high_df)/len(data)*100:.2f}%)")
    report.append("")
    report.append("Cluster-number diagnostics:")
    report.append(cluster_quality.to_string(index=False))
    report.append("")
    report.append("Subject-level membership probabilities:")
    report.append("Saved to Table_S4_subject_group_membership_probabilities.csv")
    report.append("")
    report.append("Final subtype counts:")
    report.append(count_table.to_string(index=False))
    report.append("")
    report.append("Final subtype standardized centers:")
    report.append(center_mean.to_string(index=False))
    report.append("")
    report.append("ANOVA/Kruskal results:")
    report.append(anova_table.to_string(index=False))
    report.append("")
    report.append("Bootstrap stability:")
    report.append(bootstrap_summary.to_string(index=False))

    with open(out_dir / "Section4_analysis_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    zip_out = ZIP_OUTPUT
    with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in out_dir.rglob("*"):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(out_dir))

    print(f"Analysis complete. Outputs: {out_dir}")
    print(f"Zip file: {zip_out}")


if __name__ == "__main__":
    main()

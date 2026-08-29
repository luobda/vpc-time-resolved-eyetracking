# -*- coding: utf-8 -*-
"""
Paper figure generation script.

Figures:
  Figure 1  Trial-level difficulty validity
  Figure 2  Within-participant comparison of easier and harder trials
  Figure 3  Distributions and validity of the three core metrics
  Figure 4  Validation of the trial-level difficulty index

The source Excel columns remain unchanged. All figure-facing text is English.
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# 0. Configuration
# ======================== 路径配置 ========================
# 以当前脚本所在的 code 文件夹为基准，自动定位同级的 data 文件夹
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = (SCRIPT_DIR / ".." / "data").resolve()

TRIAL_FILE = DATA_DIR / "fitting_results_difficulty.xlsx"

# Source-data columns
SUBJECT_COL = "被试"
TRIAL_COL = "试次"
DIFFICULTY_COL = "difficulty"
MEMORY_COL = "记忆表征能力"
NOISE_COL = "认知加工噪声水平"
LATENCY_COL = "记忆提取速率"

# Output directory
OUTPUT_DIR = Path("Chapter1_Output")

# Theoretical chance level for memory representation
CHANCE_LEVEL = 0

# Number of easier trials after ranking by difficulty
N_EASY = 15

DPI = 300
SAVE_PDF = True

# Source column -> figure display label
METRICS = {
    MEMORY_COL: "Memory representation",
    NOISE_COL: "Processing noise",
    LATENCY_COL: "Retrieval latency",
}


# Axis units
UNITS = {
    MEMORY_COL: "",
    NOISE_COL: "",
    LATENCY_COL: " (s)",
}

# Core metric colors
C_REP   = "#2F6F95"
C_NOISE = "#C8762E"
C_LAT   = "#4F8A6B"
METRIC_COLOR = {
    MEMORY_COL: C_REP,
    NOISE_COL: C_NOISE,
    LATENCY_COL: C_LAT,
}
C_EASY  = "#6FA8C7"
C_HARD  = "#C05A53"
C_EASY_MEAN = "#168BC1"
C_HARD_MEAN = "#D4473A"


# Publication-style colors
PUB_MEMORY_FILL   = "#AFCBE3"   # Memory representation
PUB_MEMORY_LINE   = "#2F6F95"
PUB_MEMORY_DARK   = "#1F4E6B"

PUB_NOISE_FILL   = "#B7DDD8"   # Processing noise
PUB_NOISE_LINE   = "#4F8A6B"
PUB_NOISE_DARK   = "#2F6651"

PUB_LAT_FILL   = "#F3C7A6"   # Retrieval latency
PUB_LAT_LINE   = "#C8762E"
PUB_LAT_DARK   = "#8E561F"

PUB_CHANCE     = "#7A7A7A"   # Reference line


# ----------------------------------------------------------------------
# 1. Global style
# ----------------------------------------------------------------------
def setup_style():
    """Configure the global Matplotlib style for English-only figures."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 110,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.linewidth": 1.0,
        "axes.edgecolor": "#333333",
        "axes.titlepad": 8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "legend.frameon": False,
        "legend.fontsize": 10,
    })


def despine(ax, left=True, bottom=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if not left:
        ax.spines["left"].set_visible(False)
    if not bottom:
        ax.spines["bottom"].set_visible(False)


def p_to_stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."


def fmt_p(p):
    return "< .001" if p < .001 else f"= {p:.3f}".replace("0.", ".")


# ----------------------------------------------------------------------
# 2. Data loading and aggregation
# ----------------------------------------------------------------------
def load_data():
    """Load trial-level data and construct the three analysis levels.

    Returns
    -------
    trial : pandas.DataFrame
        Raw trial-level data.
    subj : pandas.DataFrame
        Participant-level means for the three metrics.
    per_trial : pandas.DataFrame
        Trial-level metric means and mean difficulty across participants.
    """
    trial = pd.read_excel(TRIAL_FILE)

    subj = trial.groupby(SUBJECT_COL)[list(METRICS)].mean()

    per_trial = trial.groupby(TRIAL_COL).agg(
        {**{m: "mean" for m in METRICS}, DIFFICULTY_COL: "mean"}
    ).reset_index()

    return trial, subj, per_trial


def save_figure(fig, name, dpi=DPI, pad_inches=None):
    """Save PNG output and optionally save PDF according to SAVE_PDF."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    options = {"bbox_inches": "tight", "facecolor": "white"}
    if pad_inches is not None:
        options["pad_inches"] = pad_inches

    paths = []
    png_path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(png_path, dpi=dpi, **options)
    paths.append(str(png_path))

    if SAVE_PDF:
        pdf_path = OUTPUT_DIR / f"{name}.pdf"
        fig.savefig(pdf_path, **options)
        paths.append(str(pdf_path))

    print("  Saved: " + " / ".join(paths))


def report_core_metric_descriptives(subj):
    """Print participant-level descriptive statistics for the three core metrics."""
    display_names = {
        MEMORY_COL: "记忆表征能力",
        NOISE_COL: "认知加工噪声水平",
        LATENCY_COL: "记忆提取潜伏期",
    }

    print("\n" + "=" * 72)
    print("三个核心指标的描述性统计（被试层面）")
    print("=" * 72)

    for col in METRICS:
        values = pd.to_numeric(subj[col], errors="coerce").dropna()
        n = len(values)

        if n == 0:
            print(f"{display_names[col]}: 无有效数据")
            continue

        mean = values.mean()
        sd = values.std(ddof=1)
        median = values.median()
        vmin = values.min()
        vmax = values.max()

        print(
            f"{display_names[col]}: "
            f"M={mean:.4f}, SD={sd:.4f}, Median={median:.4f}, "
            f"Range=[{vmin:.4f}, {vmax:.4f}] (n={n})"
        )

    print("=" * 72)


# ======================================================================
# Figure 1  Trial-level difficulty validity
# ======================================================================
def fig_1_difficulty_validity(per_trial):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    for panel, ax, col in zip("abc", axes, METRICS):
        x = per_trial[DIFFICULTY_COL].values
        y = per_trial[col].values
        color = METRIC_COLOR[col]
        r, p = stats.pearsonr(x, y)

        ax.scatter(x, y, s=55, color=color, alpha=0.75,
                   edgecolor="white", linewidth=0.6, zorder=3)

        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ys = slope * xs + intercept
        n = len(x)
        resid = y - (slope * x + intercept)
        s_err = np.sqrt(np.sum(resid ** 2) / (n - 2))
        sxx = np.sum((x - x.mean()) ** 2)
        band = stats.t.ppf(0.975, n - 2) * s_err * np.sqrt(1 / n + (xs - x.mean()) ** 2 / sxx)
        sig = p < .05

        ax.plot(xs, ys, color=color, lw=2.2, ls="-" if sig else "--", zorder=4)
        ax.fill_between(xs, ys - band, ys + band, color=color, alpha=0.13, zorder=2)

        star = p_to_stars(p)
        ax.text(0.04, 0.05,
                f"Pearson $r$ = {r:.3f}{'' if star=='n.s.' else ' ' + star}\n$p$ {fmt_p(p)}",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=10.5,
                bbox=dict(boxstyle="round,pad=0.4",
                          fc="#FBFBFB" if not sig else "#F4F7F9",
                          ec=color if sig else "#CCCCCC", lw=1.1))

        ax.set_xlabel("Trial-level difficulty")
        ax.set_ylabel(METRICS[col] + UNITS[col])
        ax.set_title(f"({panel}) {METRICS[col]}", fontweight="bold", color=color)
        despine(ax)

    fig.suptitle("Figure 1  Validity of the trial-level difficulty index",
                 fontweight="bold", fontsize=13.5, y=1.03)
    fig.tight_layout()
    save_figure(fig, "Figure1_difficulty_validity")
    return fig


# ======================================================================
# Figure 2  Within-participant easier/harder comparison
# ======================================================================
def fig_2_easy_vs_hard(trial):
    trial_diff = trial.groupby(TRIAL_COL)[DIFFICULTY_COL].mean().sort_values()
    easy_trials = trial_diff.index[:N_EASY]
    hard_trials = trial_diff.index[N_EASY:]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.0))
    rng = np.random.default_rng(7)

    for panel, ax, col in zip("abc", axes, METRICS):
        e = trial[trial[TRIAL_COL].isin(easy_trials)].groupby(SUBJECT_COL)[col].mean()
        h = trial[trial[TRIAL_COL].isin(hard_trials)].groupby(SUBJECT_COL)[col].mean()
        idx = e.index.intersection(h.index)
        e, h = e[idx].values, h[idx].values
        n = len(idx)
        diff = h - e
        t, p = stats.ttest_rel(e, h)
        dz = diff.mean() / diff.std(ddof=1)
        tcrit = stats.t.ppf(0.975, n - 1)

        pos = [1, 2]

        jx_e = pos[0] + (rng.random(n) - 0.5) * 0.14
        jx_h = pos[1] + (rng.random(n) - 0.5) * 0.14
        for i in range(n):
            ax.plot([jx_e[i], jx_h[i]], [e[i], h[i]],
                    color="#B9B9B9", lw=0.5, alpha=0.45, zorder=1)

        ax.scatter(jx_e, e, s=16, color=C_EASY, alpha=0.28,
                   edgecolor="none", zorder=2)
        ax.scatter(jx_h, h, s=16, color=C_HARD, alpha=0.28,
                   edgecolor="none", zorder=2)

        for x0, vals, c, ec in [(pos[0], e, C_EASY_MEAN, "#126E99"),
                                (pos[1], h, C_HARD_MEAN, "#A83228")]:
            ax.errorbar(x0, vals.mean(),
                        yerr=tcrit * vals.std(ddof=1) / np.sqrt(n),
                        fmt="o", color=c, ecolor=ec, alpha=1.0,
                        elinewidth=2.8, capsize=6, capthick=2.8,
                        markersize=12.5, markeredgecolor="white",
                        markeredgewidth=1.8, zorder=6)

        ymax = max(e.max(), h.max())
        ymin = min(e.min(), h.min())
        yr = (ymax - ymin) or 1
        yb = ymax + 0.06 * yr
        ax.plot([1, 1, 2, 2], [yb, yb + 0.02 * yr, yb + 0.02 * yr, yb],
                color="#333", lw=1.3)
        label = p_to_stars(p)
        ax.text(1.5, yb + 0.03 * yr, label, ha="center", va="bottom",
                fontsize=13 if p < .05 else 10,
                fontweight="bold" if p < .05 else "normal")

        ax.text(0.5, 0.015,
                f"$t$({n-1}) = {t:.2f}, $p$ {fmt_p(p)}" +
                (f", $d_z$ = {abs(dz):.2f}" if p < .05 else ""),
                transform=ax.transAxes, ha="center", va="bottom", fontsize=9.5,
                bbox=dict(boxstyle="round,pad=0.35", fc="#F7F7F7", ec="#CCCCCC", lw=0.9))

        ax.set_xlim(0.5, 2.5)
        ax.set_xticks(pos)
        ax.set_xticklabels(["Easier", "Harder"])
        ax.set_ylabel(METRICS[col] + UNITS[col])
        ax.set_ylim(top=yb + 0.12 * yr)
        ax.set_title(f"({panel}) {METRICS[col]}", fontweight="bold",
                     color=METRIC_COLOR[col])
        despine(ax)

    handles = [Patch(fc=C_EASY, label="Easier"), Patch(fc=C_HARD, label="Harder")]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Figure 2  Within-participant differences between easier and harder trials",
                 fontweight="bold", fontsize=13.5, y=1.03)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    save_figure(fig, "Figure2_easy_vs_hard")
    return fig


# ======================================================================
# Figure 3 / Figure 4 publication style
# ======================================================================
BRM_DPI = 600
BRM_TEXT = "#222222"
BRM_AXIS = "#333333"
BRM_GRID = "#D9D9D9"


def _brm_rcparams():
    """Return publication-style rcParams used only for Figures 3 and 4."""
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "axes.linewidth": 0.8,
        "axes.edgecolor": BRM_AXIS,
        "axes.labelcolor": BRM_TEXT,
        "xtick.color": BRM_TEXT,
        "ytick.color": BRM_TEXT,
        "text.color": BRM_TEXT,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }


def _style_brm_axis(ax, grid_axis="y"):
    """Apply a clean publication-style axis with minimal grid lines."""
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", which="major", length=3, width=0.8)
    if grid_axis:
        ax.grid(axis=grid_axis, color=BRM_GRID, linewidth=0.55, alpha=0.65, zorder=0)
    ax.set_axisbelow(True)


def _panel_heading(ax, panel_label, title):
    """Draw a panel label and descriptive panel title separately."""
    ax.text(
        0.00, 1.045, panel_label,
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=10.5, fontweight="bold",
        clip_on=False,
    )
    ax.set_title(title, loc="left", x=0.105, pad=6, fontweight="normal")


def _hist_with_density(ax, values, xlabel, panel_label, panel_title,
                       fill_color, line_color, dark_color,
                       add_chance_line=False, chance_level=0.5):
    """Draw a histogram, density curve, rug marks, mean, and optional chance line."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        _panel_heading(ax, panel_label, panel_title)
        _style_brm_axis(ax, grid_axis=None)
        return

    bins = np.histogram_bin_edges(values, bins="fd")
    if len(bins) < 3:
        bins = np.histogram_bin_edges(values, bins="auto")

    ax.hist(
        values,
        bins=bins,
        density=True,
        color=fill_color,
        edgecolor=dark_color,
        linewidth=0.65,
        alpha=0.78,
        zorder=1,
    )

    if len(values) > 2 and len(np.unique(values)) > 1:
        data_range = values.max() - values.min()
        pad = max(data_range * 0.10, 1e-6)
        xs = np.linspace(values.min() - pad, values.max() + pad, 300)
        try:
            kde = stats.gaussian_kde(values)
            ax.plot(xs, kde(xs), color=line_color, lw=1.6, zorder=3)
        except np.linalg.LinAlgError:
            pass

    mean_value = values.mean()
    ax.axvline(mean_value, color=dark_color, lw=1.15, ls=":", zorder=4)
    ax.text(
        mean_value, 0.98, "Mean",
        transform=ax.get_xaxis_transform(),
        ha="center", va="top", fontsize=7.8, color=dark_color,
    )

    if add_chance_line:
        ax.axvline(chance_level, color=PUB_CHANCE, lw=1.15, ls="--", zorder=2)
        ax.text(
            chance_level, 0.86, "Chance",
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=7.8, color=PUB_CHANCE,
        )

    # rug marks show every participant without adding another legend.
    rug_height = ax.get_ylim()[1] * 0.025
    ax.vlines(values, 0, rug_height, color=dark_color, lw=0.45, alpha=0.45, zorder=4)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    _panel_heading(ax, panel_label, panel_title)
    _style_brm_axis(ax, grid_axis="y")


def fig_3_distributions_and_validity(subj):
    """
    Figure 3. Publication-style version.

    The figure file contains only the figure body. The Figure number, title,
    and Note should be supplied in the manuscript rather than drawn in the PNG/PDF.
    """
    with mpl.rc_context(_brm_rcparams()):
        rep = subj[MEMORY_COL].dropna().to_numpy(dtype=float)
        noise = subj[NOISE_COL].dropna().to_numpy(dtype=float)
        lat = subj[LATENCY_COL].dropna().to_numpy(dtype=float)

        n = len(rep)
        m = np.mean(rep)
        sd = np.std(rep, ddof=1)
        se = sd / np.sqrt(n)
        t, p = stats.ttest_1samp(rep, CHANCE_LEVEL)
        d = (m - CHANCE_LEVEL) / sd if sd > 0 else np.nan
        tcrit = stats.t.ppf(0.975, n - 1)
        ci_low, ci_high = m - tcrit * se, m + tcrit * se

        # Publication-width layout with typography set at final size.
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.65), constrained_layout=True)
        axA, axB, axC, axD = axes.flatten()

        _hist_with_density(
            axA, rep,
            xlabel="Memory representation score",
            panel_label="A",
            panel_title="",
            fill_color=PUB_MEMORY_FILL,
            line_color=PUB_MEMORY_LINE,
            dark_color=PUB_MEMORY_DARK,
            add_chance_line=True,
            chance_level=CHANCE_LEVEL,
        )

        _hist_with_density(
            axB, noise,
            xlabel="Cognitive processing noise score",
            panel_label="B",
            panel_title="",
            fill_color=PUB_NOISE_FILL,
            line_color=PUB_NOISE_LINE,
            dark_color=PUB_NOISE_DARK,
        )

        _hist_with_density(
            axC, lat,
            xlabel="Latency to first detectable novelty-orienting peak (s)",
            panel_label="C",
            panel_title="",
            fill_color=PUB_LAT_FILL,
            line_color=PUB_LAT_LINE,
            dark_color=PUB_LAT_DARK,
        )

        # Panel d: participant values, distribution, mean, and 95% CI.
        # Use one shared x center so the violin, scatter, mean/CI, and tick align.
        x_center = 1.0
        vp = axD.violinplot(rep, positions=[x_center], widths=0.42, showextrema=False)
        for body in vp["bodies"]:
            body.set_facecolor(PUB_MEMORY_FILL)
            body.set_edgecolor(PUB_MEMORY_DARK)
            body.set_linewidth(0.75)
            body.set_alpha(0.60)

        rng = np.random.default_rng(1234)
        xj = x_center + (rng.random(n) - 0.5) * 0.12
        axD.scatter(
            xj, rep,
            s=13,
            facecolor="white",
            edgecolor=PUB_MEMORY_LINE,
            linewidth=0.55,
            alpha=0.72,
            zorder=3,
        )
        axD.errorbar(
            x_center, m,
            yerr=[[m - ci_low], [ci_high - m]],
            fmt="o",
            color=PUB_MEMORY_DARK,
            ecolor=PUB_MEMORY_DARK,
            elinewidth=1.7,
            capsize=4,
            capthick=1.2,
            markersize=5.8,
            markerfacecolor=PUB_MEMORY_DARK,
            markeredgecolor="white",
            markeredgewidth=0.65,
            zorder=5,
        )
        axD.axhline(CHANCE_LEVEL, color=PUB_CHANCE, lw=1.15, ls="--", zorder=1)
        axD.text(
            0.98, CHANCE_LEVEL, "Chance",
            ha="left", va="bottom", fontsize=7.8, color=PUB_CHANCE,
        )

        stat_text = (
            f"$M$ = {m:.3f}, 95% CI [{ci_low:.3f}, {ci_high:.3f}]\n"
            f"$t$({n - 1}) = {t:.2f}, $p$ {fmt_p(p)}, $d$ = {d:.2f}"
        )
        axD.text(
            0.03, 0.97, stat_text,
            transform=axD.transAxes,
            ha="left", va="top", fontsize=7.9,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.86, pad=1.5),
        )

        y_min = min(np.nanmin(rep), CHANCE_LEVEL)
        y_max = max(np.nanmax(rep), ci_high)
        y_range = y_max - y_min if y_max > y_min else 1.0
        axD.set_ylim(y_min - 0.08 * y_range, y_max + 0.28 * y_range)
        axD.set_xlim(0.62, 1.30)
        axD.set_xticks([x_center])
        axD.set_xticklabels(["Participant sample"])
        axD.set_ylabel("Memory representation score")
        _panel_heading(axD, "D", "")
        _style_brm_axis(axD, grid_axis="y")

        save_figure(fig, "Figure3_distributions_and_validity", dpi=BRM_DPI, pad_inches=0.04)
        print(
            f"[Figure 3] n={n}  Memory representation: M={m:.4f}, SD={sd:.4f}, "
            f"t({n-1})={t:.2f}, p={p:.2e}, d={d:.2f}"
        )
        return fig


def _mean_ci95(values):
    """Return the mean and the half-width of its two-sided 95% t confidence interval."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 2:
        return np.nan, np.nan
    m = values.mean()
    se = values.std(ddof=1) / np.sqrt(n)
    ci = stats.t.ppf(0.975, n - 1) * se
    return m, ci


def _regression_line_with_ci(ax, x, y, color, lw=1.6):
    """Draw an OLS line and its 95% confidence band for the conditional mean."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]

    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 200)
    ys = slope * xs + intercept

    n = len(x)
    y_hat = slope * x + intercept
    resid = y - y_hat
    sxx = np.sum((x - x.mean()) ** 2)

    if n > 2 and sxx > 0:
        s_err = np.sqrt(np.sum(resid ** 2) / (n - 2))
        band = stats.t.ppf(0.975, n - 2) * s_err * np.sqrt(
            1 / n + (xs - x.mean()) ** 2 / sxx
        )
        ax.fill_between(xs, ys - band, ys + band, color=color, alpha=0.14, linewidth=0, zorder=1)

    ax.plot(xs, ys, color=color, lw=lw, zorder=3)
    return slope, intercept


def _get_easy_hard_trials(trial, per_trial):
    """Rank trials by difficulty and retain the original stable 15/15 split."""
    trial_diff = per_trial.set_index(TRIAL_COL)[DIFFICULTY_COL].sort_values()
    n_trials = len(trial_diff)
    n_easy = N_EASY if n_trials >= 2 * N_EASY else n_trials // 2
    easy_trials = trial_diff.index[:n_easy]
    hard_trials = trial_diff.index[n_easy:]
    median_diff = np.median(trial_diff.values)
    return trial_diff, easy_trials, hard_trials, median_diff


def _paired_easy_hard_values(trial, easy_trials, hard_trials, metric_col):
    """Calculate each participant's mean metric for easier and harder trials."""
    easy = (
        trial[trial[TRIAL_COL].isin(easy_trials)]
        .groupby(SUBJECT_COL)[metric_col]
        .mean()
    )
    hard = (
        trial[trial[TRIAL_COL].isin(hard_trials)]
        .groupby(SUBJECT_COL)[metric_col]
        .mean()
    )
    idx = easy.index.intersection(hard.index)
    return easy.loc[idx].to_numpy(), hard.loc[idx].to_numpy(), idx


def _paired_comparison_panel(ax, trial, easy_trials, hard_trials,
                             metric_col, title, ylabel, panel_label, rng):
    """Draw paired participant values and group means with 95% confidence intervals."""
    easy, hard, idx = _paired_easy_hard_values(
        trial=trial,
        easy_trials=easy_trials,
        hard_trials=hard_trials,
        metric_col=metric_col,
    )

    valid = np.isfinite(easy) & np.isfinite(hard)
    easy, hard = easy[valid], hard[valid]
    n = len(easy)
    diff = hard - easy
    t, p = stats.ttest_rel(easy, hard)
    diff_sd = diff.std(ddof=1)
    dz = diff.mean() / diff_sd if diff_sd > 0 else np.nan

    easy_mean, easy_ci = _mean_ci95(easy)
    hard_mean, hard_ci = _mean_ci95(hard)

    x_easy, x_hard = 1.0, 2.0
    jx_easy = x_easy + (rng.random(n) - 0.5) * 0.10
    jx_hard = x_hard + (rng.random(n) - 0.5) * 0.10

    for i in range(n):
        ax.plot(
            [jx_easy[i], jx_hard[i]], [easy[i], hard[i]],
            color="#A9A9A9", lw=0.45, alpha=0.42, zorder=1,
        )

    # Shape and fill differ as well as color, so the comparison remains readable in grayscale.
    ax.scatter(
        jx_easy, easy,
        s=12, marker="o", facecolor="white", edgecolor=C_EASY_MEAN,
        linewidth=0.55, alpha=0.70, zorder=2,
    )
    ax.scatter(
        jx_hard, hard,
        s=12, marker="s", facecolor=C_HARD, edgecolor=C_HARD_MEAN,
        linewidth=0.45, alpha=0.50, zorder=2,
    )

    ax.errorbar(
        x_easy, easy_mean, yerr=easy_ci,
        fmt="o", color=C_EASY_MEAN, ecolor=C_EASY_MEAN,
        markersize=6.4, elinewidth=1.6, capsize=3.5, capthick=1.1,
        markerfacecolor="white", markeredgewidth=1.2, zorder=5,
    )
    ax.errorbar(
        x_hard, hard_mean, yerr=hard_ci,
        fmt="s", color=C_HARD_MEAN, ecolor=C_HARD_MEAN,
        markersize=6.0, elinewidth=1.6, capsize=3.5, capthick=1.1,
        markerfacecolor=C_HARD_MEAN, markeredgecolor="white", markeredgewidth=0.65,
        zorder=5,
    )

    y_min = min(np.nanmin(easy), np.nanmin(hard), easy_mean - easy_ci, hard_mean - hard_ci)
    y_max = max(np.nanmax(easy), np.nanmax(hard), easy_mean + easy_ci, hard_mean + hard_ci)
    y_range = y_max - y_min if y_max > y_min else 1.0
    ax.set_ylim(y_min - 0.08 * y_range, y_max + 0.30 * y_range)

    stat_text = (
        f"$t$({n - 1}) = {t:.2f}, $p$ {fmt_p(p)}\n"
        f"$d_z$ = {dz:.2f}"
    )
    ax.text(
        0.03, 0.97, stat_text,
        transform=ax.transAxes,
        ha="left", va="top", fontsize=7.7,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.86, pad=1.2),
    )

    ax.set_xlim(0.62, 2.38)
    ax.set_xticks([x_easy, x_hard])
    ax.set_xticklabels(["Easier", "Harder"])
    ax.set_ylabel(ylabel)
    _panel_heading(ax, panel_label, title)
    _style_brm_axis(ax, grid_axis="y")

    print(
        f"[Figure 4{panel_label.upper()}] {title}: "
        f"Easy M={easy_mean:.4f}, Hard M={hard_mean:.4f}, "
        f"t({n-1})={t:.2f}, p={p:.4g}, dz={dz:.2f}"
    )


def fig_4_trial_level_difficulty_validation(trial, per_trial):
    """
    Figure 4. Publication-style version.

    The trial split, statistics, and source data are unchanged; only the visual
    encoding, terminology, typography, and export quality are revised.
    """
    with mpl.rc_context(_brm_rcparams()):
        trial_diff, easy_trials, hard_trials, median_diff = _get_easy_hard_trials(
            trial=trial,
            per_trial=per_trial,
        )

        easy_diff = trial_diff.loc[easy_trials].to_numpy(dtype=float)
        hard_diff = trial_diff.loc[hard_trials].to_numpy(dtype=float)
        all_diff = trial_diff.to_numpy(dtype=float)

        fig = plt.figure(figsize=(7.2, 6.15), constrained_layout=True)
        gs = fig.add_gridspec(
            2, 6,
            height_ratios=[1.0, 1.08],
            hspace=0.30,
            wspace=0.48,
        )
        axA = fig.add_subplot(gs[0, 0:3])
        axB = fig.add_subplot(gs[0, 3:6])
        axC = fig.add_subplot(gs[1, 0:2])
        axD = fig.add_subplot(gs[1, 2:4])
        axE = fig.add_subplot(gs[1, 4:6])

        # a: distribution and prespecified ranked split.
        bins = np.histogram_bin_edges(all_diff, bins=8)
        axA.hist(
            easy_diff, bins=bins,
            color=C_EASY, edgecolor="#3A657D", linewidth=0.6,
            alpha=0.70, hatch="///", label=f"Easier ($n$ = {len(easy_diff)})",
            zorder=2,
        )
        axA.hist(
            hard_diff, bins=bins,
            color=C_HARD, edgecolor="#7A302C", linewidth=0.6,
            alpha=0.70, hatch="\\\\", label=f"Harder ($n$ = {len(hard_diff)})",
            zorder=2,
        )
        axA.axvline(median_diff, color=BRM_AXIS, lw=1.15, ls="--", zorder=4)
        axA.text(
            median_diff, 0.97, "Median",
            transform=axA.get_xaxis_transform(),
            ha="center", va="top", fontsize=7.8,
        )
        rug_y = -0.18
        axA.scatter(
            easy_diff, np.full_like(easy_diff, rug_y),
            s=10, marker="o", facecolor="white", edgecolor=C_EASY_MEAN,
            linewidth=0.55, clip_on=False, zorder=5,
        )
        axA.scatter(
            hard_diff, np.full_like(hard_diff, rug_y),
            s=10, marker="s", facecolor=C_HARD, edgecolor=C_HARD_MEAN,
            linewidth=0.45, alpha=0.70, clip_on=False, zorder=5,
        )
        ymax = axA.get_ylim()[1]
        axA.set_ylim(-0.45, ymax * 1.12)
        axA.set_xlabel("Trial-level difficulty")
        axA.set_ylabel("Number of trials")
        axA.legend(loc="upper left", ncol=1, handlelength=1.8, borderaxespad=0.3)
        _panel_heading(axA, "a", "Difficulty-score distribution")
        _style_brm_axis(axA, grid_axis="y")

        # b: trial-level association.
        x = per_trial[DIFFICULTY_COL].to_numpy(dtype=float)
        y = per_trial[MEMORY_COL].to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        x_valid, y_valid = x[valid], y[valid]
        r, p = stats.pearsonr(x_valid, y_valid)

        axB.scatter(
            x_valid, y_valid,
            s=22, marker="o", facecolor="white", edgecolor=C_REP,
            linewidth=0.75, alpha=0.90, zorder=3,
        )
        _regression_line_with_ci(axB, x_valid, y_valid, color=C_REP, lw=1.65)
        axB.text(
            0.04, 0.96,
            f"Pearson $r$ = {r:.3f}, $p$ {fmt_p(p)}",
            transform=axB.transAxes,
            ha="left", va="top", fontsize=7.9,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.86, pad=1.2),
        )
        axB.set_xlabel("Trial-level difficulty")
        axB.set_ylabel("Memory representation")
        _panel_heading(axB, "b", "Trial-level association")
        _style_brm_axis(axB, grid_axis="y")

        rng = np.random.default_rng(2024)
        _paired_comparison_panel(
            ax=axC,
            trial=trial,
            easy_trials=easy_trials,
            hard_trials=hard_trials,
            metric_col=MEMORY_COL,
            title="Memory representation",
            ylabel="Memory representation",
            panel_label="c",
            rng=rng,
        )
        _paired_comparison_panel(
            ax=axD,
            trial=trial,
            easy_trials=easy_trials,
            hard_trials=hard_trials,
            metric_col=NOISE_COL,
            title="Processing noise",
            ylabel="Processing noise",
            panel_label="d",
            rng=rng,
        )
        _paired_comparison_panel(
            ax=axE,
            trial=trial,
            easy_trials=easy_trials,
            hard_trials=hard_trials,
            metric_col=LATENCY_COL,
            title="Retrieval latency",
            ylabel="Retrieval latency (s)",
            panel_label="e",
            rng=rng,
        )

        save_figure(fig, "Figure4_trial_level_difficulty_validation", dpi=BRM_DPI, pad_inches=0.04)
        print(f"[Figure 4B] Pearson r={r:.4f}, p={p:.4g}")
        print(
            f"[Figure 4A] median difficulty={median_diff:.4f}; "
            f"easy trials={len(easy_trials)}, hard trials={len(hard_trials)}"
        )
        return fig


# ----------------------------------------------------------------------
# 5. Main workflow
# ----------------------------------------------------------------------
def main():
    """Run data loading, analysis, and figure export."""
    setup_style()
    trial, subj, per_trial = load_data()

    # Descriptive statistics for the three core metrics
    report_core_metric_descriptives(subj)

    # Figures 1–2
    fig_1_difficulty_validity(per_trial)
    fig_2_easy_vs_hard(trial)

    # Figure 3
    fig_3_distributions_and_validity(subj)

    # Figure 4
    fig_4_trial_level_difficulty_validation(trial, per_trial)

    print(f"\nDone. Files saved to: {OUTPUT_DIR.resolve()}")
    # plt.show()  # 如需交互式查看可取消注释


if __name__ == "__main__":
    main()

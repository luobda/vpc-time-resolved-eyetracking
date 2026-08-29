# -*- coding: utf-8 -*-
"""
VPC individual-heterogeneity analysis.

Main tasks:
1. Read the original Excel file without modifying its Chinese schema.
2. Fit participant-level regressions for the three core metrics:
   - Memory representation
   - Processing noise
   - Retrieval latency
3. Compute static participant features and dynamic difficulty-response slopes.
4. Summarize slope distributions, response directions, mixed-model comparisons,
   and static/dynamic correlations.
5. Generate the retained publication-style individual-heterogeneity figure.
6. Export analysis tables, a text summary, and a ZIP archive.

Dependencies:
    pip install pandas numpy scipy statsmodels matplotlib openpyxl
"""

import os
import sys
import zipfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")


# ============================================================
# 0. 参数设置
# ======================== 路径配置 ========================
# 以当前脚本所在的 code 文件夹为基准，自动定位同级的 data 文件夹
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = (SCRIPT_DIR / ".." / "data").resolve()

DEFAULT_INPUT_FILE = DATA_DIR / "fitting_results_difficulty.xlsx"

# 可通过命令行或环境变量指定输入文件：
# python 2_个体异质性_参考论文绘图版.py "你的数据.xlsx"
INPUT_FILE = (
    sys.argv[1]
    if len(sys.argv) >= 2
    else os.environ.get("VPC_INPUT_FILE", DEFAULT_INPUT_FILE)
)

OUTPUT_DIR = os.environ.get("VPC_OUTPUT_DIR", "Chapter2_Output_2")
ZIP_NAME = os.environ.get(
    "VPC_ZIP_NAME",
    "Chapter2_Output_2.zip"
)

# Reproducible jitter for the retained individual-heterogeneity figure.
RANDOM_SEED = 20250520

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 记忆提取速率在当前算法中是“第一个显著峰值出现时间”，
# 因此数值越大代表提取越慢，本质上是 latency。
# 如果你的数据中该列已经被转换为“越大越好”的提取效率，请改为 False。
RETRIEVAL_IS_LATENCY = True

# 认知加工噪声：数值越大表示波动越大、稳定性越低。
PROCESSING_NOISE_HIGHER_IS_WORSE = True

# 清理旧版本中已取消或已重新编号的输出，避免残留文件进入 ZIP。
DEPRECATED_OUTPUTS = [
    "Fig3_2_2_individual_heterogeneity_reference_style_raw_slopes.png",
    "Fig3_2_2_individual_heterogeneity_reference_style_adaptive_slopes.png",
    "Fig3_2_2_individual_heterogeneity_reference_style_adaptive_slopes.pdf",
    "Fig4B_individual_lines_memory_representation.png",
    "Fig4C_individual_lines_processing_noise.png",
    "Fig4D_individual_lines_retrieval_latency.png",
    "Fig4E_scatter_representation_vs_stability.png",
    "Fig4F_scatter_representation_vs_retrieval.png",
    "Fig4G_scatter_stability_vs_retrieval.png",
    "paper_ready_results_section_3_2_2.txt",
]


# ============================================================
# 1. 工具函数
# ============================================================

def zscore(series):
    """标准化为均值0、标准差1；常量序列返回 NaN，避免除以 0。"""
    s = pd.Series(series, index=getattr(series, "index", None), dtype="float64")
    sd = s.std(ddof=0)
    if pd.isna(sd) or np.isclose(sd, 0):
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean()) / sd


def save_csv(dataframe, filename, index=False):
    """保存 CSV，使用 utf-8-sig 方便 Excel 打开中文。"""
    path = os.path.join(OUTPUT_DIR, filename)
    dataframe.to_csv(path, index=index, encoding="utf-8-sig")
    return path


def remove_deprecated_outputs():
    """删除旧版本已取消的输出文件，避免历史残留混入本次结果。"""
    for filename in DEPRECATED_OUTPUTS:
        path = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(path):
            os.remove(path)


def fit_mixedlm(formula, data, group_col="subject"):
    """
    拟合随机截距线性混合效应模型。
    用于把 2.1 群体固定效应与 2 个体斜率均值进行对照。
    """
    model = smf.mixedlm(formula, data=data, groups=data[group_col])

    methods = ["powell", "lbfgs", "bfgs", "cg", "nm"]
    last_error = None

    for method in methods:
        try:
            result = model.fit(
                reml=False,
                method=method,
                maxiter=2000,
                disp=False
            )
            if result.converged:
                return result
            last_error = RuntimeError(f"Model did not converge with {method}")
        except Exception as e:
            last_error = e

    raise RuntimeError(f"MixedLM failed for formula: {formula}\nLast error: {last_error}")


def fit_subject_ols(data, y_col, x_col="difficulty_z"):
    """
    对单名被试拟合 y ~ difficulty_z 的个体水平 OLS 回归。
    返回截距、斜率、SE、t、p、CI、R²、有效试次数。
    """
    temp = data[[y_col, x_col]].dropna().copy()

    if len(temp) < 3:
        return {
            "intercept": np.nan,
            "slope": np.nan,
            "slope_SE": np.nan,
            "t": np.nan,
            "p": np.nan,
            "CI_lower": np.nan,
            "CI_upper": np.nan,
            "r_squared": np.nan,
            "n_trials": len(temp)
        }

    if temp[x_col].std(ddof=0) == 0:
        return {
            "intercept": np.nan,
            "slope": np.nan,
            "slope_SE": np.nan,
            "t": np.nan,
            "p": np.nan,
            "CI_lower": np.nan,
            "CI_upper": np.nan,
            "r_squared": np.nan,
            "n_trials": len(temp)
        }

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
        "n_trials": int(model.nobs)
    }


def descriptive_stats(series):
    """返回一组描述性统计。"""
    s = pd.Series(series).dropna()

    if len(s) == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "min": np.nan,
            "max": np.nan
        }

    return {
        "n": len(s),
        "mean": s.mean(),
        "std": s.std(ddof=1),
        "median": s.median(),
        "q25": s.quantile(0.25),
        "q75": s.quantile(0.75),
        "min": s.min(),
        "max": s.max()
    }


def correlation_table(data, variables, method="pearson"):
    """
    计算变量两两相关，返回长格式表格。
    method 可选 pearson 或 spearman。
    """
    rows = []

    for i, var1 in enumerate(variables):
        for var2 in variables[i + 1:]:

            temp = data[[var1, var2]].dropna()

            if len(temp) < 3:
                r = np.nan
                p = np.nan
                n = len(temp)
            else:
                if method == "pearson":
                    r, p = stats.pearsonr(temp[var1], temp[var2])
                elif method == "spearman":
                    r, p = stats.spearmanr(temp[var1], temp[var2])
                else:
                    raise ValueError("method must be 'pearson' or 'spearman'")
                n = len(temp)

            rows.append({
                "var1": var1,
                "var2": var2,
                "method": method,
                "n": n,
                "r": r,
                "p": p
            })

    return pd.DataFrame(rows)


def safe_cv(series):
    """
    计算变异系数：SD / |Mean|。
    均值接近 0 时返回 NaN，避免将接近 0 的均值解释为极端 CV。
    """
    s = pd.Series(series).dropna().astype(float)
    if len(s) == 0:
        return np.nan
    mean_value = s.mean()
    if np.isclose(mean_value, 0.0, atol=1e-12):
        return np.nan
    return s.std(ddof=1) / abs(mean_value)


def describe_slope(series):
    """为参考论文式图表和结果表生成完整斜率描述统计。"""
    s = pd.Series(series).dropna().astype(float)
    n = len(s)
    if n == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "sd": np.nan,
            "cv_abs_mean": np.nan,
            "median": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "min": np.nan,
            "max": np.nan,
            "n_positive": 0,
            "pct_positive": np.nan,
            "n_negative": 0,
            "pct_negative": np.nan,
            "n_zero": 0,
            "pct_zero": np.nan
        }

    n_positive = int((s > 0).sum())
    n_negative = int((s < 0).sum())
    n_zero = int((s == 0).sum())
    return {
        "n": n,
        "mean": s.mean(),
        "sd": s.std(ddof=1),
        "cv_abs_mean": safe_cv(s),
        "median": s.median(),
        "q25": s.quantile(0.25),
        "q75": s.quantile(0.75),
        "min": s.min(),
        "max": s.max(),
        "n_positive": n_positive,
        "pct_positive": n_positive / n * 100,
        "n_negative": n_negative,
        "pct_negative": n_negative / n * 100,
        "n_zero": n_zero,
        "pct_zero": n_zero / n * 100
    }


def plot_individual_heterogeneity_figure(
    slope_df,
    metric_specs,
    output_path,
    seed=RANDOM_SEED
):
    """
    绘制个体难度响应斜率的概率密度函数（PDF）图。

    保留原有统计与输出功能，仅将原来的 3×2
    “直方图/KDE + 箱线图/个体散点”布局压缩为 3×1：
    - 每个指标对应一个概率密度分布面板；
    - KDE 曲线表示个体斜率的经验概率密度；
    - 横轴底部 rug 表示全部个体观测；
    - 灰色虚线表示零斜率；
    - 黑色实线表示样本平均斜率；
    - 面板内标注零点两侧个体比例及 n、M、SD。

    图内文字全部使用英文；Figure 编号和总标题由论文排版承载。
    """
    if len(metric_specs) != 3:
        raise ValueError("metric_specs 必须恰好包含 3 个指标。")

    metric_colors = ["#0072B2", "#009E73", "#D55E00"]
    metric_fills = ["#B9D7EA", "#BFE4D8", "#F2C4A8"]

    # 保留 seed 参数以兼容原函数调用；PDF 图不再需要散点 jitter。
    _ = seed

    with plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white"
    }):
        fig, axes = plt.subplots(3, 1, figsize=(7.15, 7.6))
        panel_letters = ["A", "B", "C"]

        for row_index, spec in enumerate(metric_specs):
            values = (
                slope_df[spec["column"]]
                .dropna()
                .astype(float)
                .to_numpy()
            )
            values = values[np.isfinite(values)]

            if len(values) == 0:
                raise ValueError(
                    f"指标 {spec['column']} 没有可用于绘图的有效斜率。"
                )

            ax = axes[row_index]
            color = metric_colors[row_index]
            fill = metric_fills[row_index]
            stat = describe_slope(values)

            if len(values) >= 3 and np.unique(values).size >= 2:
                try:
                    kde = stats.gaussian_kde(values)
                    data_min = min(values.min(), 0.0, stat["mean"])
                    data_max = max(values.max(), 0.0, stat["mean"])
                    span = data_max - data_min
                    pad = span * 0.10 if span > 0 else 0.1
                    x_grid = np.linspace(data_min - pad, data_max + pad, 500)
                    density = kde(x_grid)

                    ax.plot(
                        x_grid,
                        density,
                        color=color,
                        linewidth=1.8,
                        zorder=3
                    )
                    ax.fill_between(
                        x_grid,
                        0,
                        density,
                        color=fill,
                        alpha=0.60,
                        zorder=2
                    )
                    ymax = float(np.nanmax(density))
                except Exception:
                    bins = np.histogram_bin_edges(values, bins="fd")
                    if len(bins) < 5:
                        bins = np.histogram_bin_edges(
                            values,
                            bins=max(5, int(np.sqrt(len(values))))
                        )
                    counts, edges = np.histogram(values, bins=bins, density=True)
                    centers = (edges[:-1] + edges[1:]) / 2
                    widths = np.diff(edges)
                    ax.bar(
                        centers,
                        counts,
                        width=widths,
                        color=fill,
                        edgecolor=color,
                        linewidth=0.8,
                        alpha=0.80,
                        zorder=2
                    )
                    ymax = float(np.nanmax(counts)) if len(counts) else 1.0
            else:
                ymax = 1.0

            if not np.isfinite(ymax) or ymax <= 0:
                ymax = 1.0

            rug_height = ymax * 0.055
            ax.vlines(
                values,
                0,
                rug_height,
                color=color,
                linewidth=0.55,
                alpha=0.45,
                zorder=4
            )

            zero_line = ax.axvline(
                0,
                color="#666666",
                linestyle=(0, (4, 3)),
                linewidth=1.0,
                label="Zero slope",
                zorder=1
            )
            mean_line = ax.axvline(
                stat["mean"],
                color="#111111",
                linestyle="-",
                linewidth=1.3,
                label="Sample mean",
                zorder=5
            )

            ax.text(
                0.02,
                0.91,
                f"Negative: {stat['pct_negative']:.2f}%",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="#333333"
            )
            ax.text(
                0.98,
                0.91,
                f"Positive: {stat['pct_positive']:.2f}%",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="#333333"
            )

            stats_text = (
                rf"$n$ = {stat['n']}, "
                rf"$M$ = {stat['mean']:.3f}, "
                rf"$SD$ = {stat['sd']:.3f}"
            )
            ax.text(
                0.98,
                0.04,
                stats_text,
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=7.5,
                color="#333333"
            )

            ax.set_xlabel(spec["axis_label"])
            ax.set_ylabel("Probability density")
            ax.set_title(
                spec.get("panel_title", "Difficulty-response slope distribution"),
                loc="left",
                pad=5
            )
            ax.yaxis.grid(
                True,
                color="#E6E6E6",
                linewidth=0.55,
                zorder=0
            )
            ax.xaxis.grid(False)
            ax.set_ylim(bottom=0)

            if row_index == 0:
                ax.legend(
                    handles=[zero_line, mean_line],
                    loc="upper center",
                    bbox_to_anchor=(0.50, 1.02),
                    ncol=2,
                    handlelength=2,
                    borderaxespad=0.2
                )

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(0.8)
            ax.spines["bottom"].set_linewidth(0.8)
            ax.text(
                -0.10,
                1.06,
                panel_letters[row_index],
                transform=ax.transAxes,
                fontsize=10.5,
                fontweight="bold",
                ha="left",
                va="top"
            )

        fig.subplots_adjust(
            left=0.12,
            right=0.985,
            bottom=0.075,
            top=0.975,
            hspace=0.62
        )

        fig.savefig(
            output_path,
            dpi=600,
            bbox_inches="tight",
            facecolor="white"
        )

        pdf_path = os.path.splitext(output_path)[0] + ".pdf"
        fig.savefig(
            pdf_path,
            bbox_inches="tight",
            facecolor="white"
        )

        plt.close(fig)

    return output_path

# ============================================================
# 2. 读取数据
# ============================================================


def main():
    """Run the complete individual-heterogeneity analysis."""
    remove_deprecated_outputs()
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"找不到输入文件：{INPUT_FILE}\n"
            f"请确认 Excel 文件与当前 Python 脚本位于同一文件夹，"
            f"或将 INPUT_FILE 修改为完整文件路径。"
        )

    raw_df = pd.read_excel(INPUT_FILE)

    print("\n原始列名：")
    print(list(raw_df.columns))


    # ============================================================
    # 3. 输入 schema 映射（仅内存中使用，不修改原始 Excel）
    # ============================================================

    # 原始 Excel 列名保持中文；下面只为内部公式和数据处理建立英文别名。
    rename_map = {
        "被试": "subject",
        "试次": "trial",
        "新图位置": "new_position",
        "记忆表征能力": "memory_representation",
        "认知加工噪声水平": "processing_noise",
        "记忆提取速率": "retrieval_latency",
        "平均整体偏好": "mean_overall_preference",
        "difficulty": "difficulty"
    }

    df = raw_df.rename(columns=rename_map).copy()

    required_cols = [
        "subject",
        "trial",
        "memory_representation",
        "processing_noise",
        "retrieval_latency",
        "difficulty"
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(
            "数据中缺少以下必要列：\n"
            + "\n".join(missing_cols)
            + "\n\n请检查 Excel 表头是否与 rename_map 一致。"
        )


    # ============================================================
    # 4. 数据清洗与变量准备
    # ============================================================

    numeric_cols = [
        "memory_representation",
        "processing_noise",
        "retrieval_latency",
        "difficulty"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required_cols).copy()

    df["subject"] = df["subject"].astype(str)
    df["trial"] = df["trial"].astype(int)

    # difficulty 标准化，后续斜率解释为：
    # difficulty 每增加 1 个标准差，因变量变化多少。
    df["difficulty_z"] = zscore(df["difficulty"])

    # 用于绘图排序
    df = df.sort_values(["subject", "trial"]).reset_index(drop=True)


    # ============================================================
    # 5. 数据质控
    # ============================================================

    trial_counts = df.groupby("subject").size()

    qc_summary = pd.DataFrame({
        "item": [
            "valid_observations",
            "n_subjects",
            "n_trials",
            "min_trials_per_subject",
            "median_trials_per_subject",
            "max_trials_per_subject",
            "missing_values_in_key_columns",
            "difficulty_mean",
            "difficulty_std",
            "difficulty_min",
            "difficulty_max"
        ],
        "value": [
            len(df),
            df["subject"].nunique(),
            df["trial"].nunique(),
            trial_counts.min(),
            trial_counts.median(),
            trial_counts.max(),
            int(df[required_cols].isna().sum().sum()),
            df["difficulty"].mean(),
            df["difficulty"].std(ddof=1),
            df["difficulty"].min(),
            df["difficulty"].max()
        ]
    })

    save_csv(qc_summary, "QC_summary_2_2.csv")

    print("\n===== 数据质控 =====")
    print(qc_summary.to_string(index=False))


    # ============================================================
    # 6. 计算每名被试的静态均值
    # ============================================================

    static_features = (
        df
        .groupby("subject")
        .agg(
            n_trials=("trial", "count"),
            mean_difficulty=("difficulty", "mean"),
            mean_memory_representation=("memory_representation", "mean"),
            mean_processing_noise=("processing_noise", "mean"),
            mean_retrieval_latency=("retrieval_latency", "mean")
        )
        .reset_index()
    )

    # 方向统一后的静态指标：
    # 表征能力：越高越好
    # 加工稳定性：噪声越低越好，所以取反后标准化
    # 提取效率：如果 retrieval 是潜伏期，则越低越好，所以取反后标准化
    static_features["z_mean_memory_representation"] = zscore(
        static_features["mean_memory_representation"]
    )

    if PROCESSING_NOISE_HIGHER_IS_WORSE:
        static_features["z_mean_processing_stability"] = -zscore(
            static_features["mean_processing_noise"]
        )
    else:
        static_features["z_mean_processing_stability"] = zscore(
            static_features["mean_processing_noise"]
        )

    if RETRIEVAL_IS_LATENCY:
        static_features["z_mean_retrieval_efficiency"] = -zscore(
            static_features["mean_retrieval_latency"]
        )
    else:
        static_features["z_mean_retrieval_efficiency"] = zscore(
            static_features["mean_retrieval_latency"]
        )

    save_csv(static_features, "static_subject_features.csv")


    # ============================================================
    # 7. 为每名被试拟合三个个体水平回归，提取斜率
    # ============================================================

    individual_rows = []

    for subject, sub_df in df.groupby("subject"):

        rep_result = fit_subject_ols(
            sub_df,
            y_col="memory_representation",
            x_col="difficulty_z"
        )

        noise_result = fit_subject_ols(
            sub_df,
            y_col="processing_noise",
            x_col="difficulty_z"
        )

        latency_result = fit_subject_ols(
            sub_df,
            y_col="retrieval_latency",
            x_col="difficulty_z"
        )

        row = {
            "subject": subject,
            "n_trials": len(sub_df),

            # 记忆表征能力个体模型
            "rep_intercept": rep_result["intercept"],
            "rep_slope_raw": rep_result["slope"],
            "rep_slope_SE": rep_result["slope_SE"],
            "rep_slope_t": rep_result["t"],
            "rep_slope_p": rep_result["p"],
            "rep_slope_CI_lower": rep_result["CI_lower"],
            "rep_slope_CI_upper": rep_result["CI_upper"],
            "rep_r_squared": rep_result["r_squared"],

            # 加工噪声个体模型
            "noise_intercept": noise_result["intercept"],
            "noise_slope_raw": noise_result["slope"],
            "noise_slope_SE": noise_result["slope_SE"],
            "noise_slope_t": noise_result["t"],
            "noise_slope_p": noise_result["p"],
            "noise_slope_CI_lower": noise_result["CI_lower"],
            "noise_slope_CI_upper": noise_result["CI_upper"],
            "noise_r_squared": noise_result["r_squared"],

            # 提取潜伏期个体模型
            "latency_intercept": latency_result["intercept"],
            "latency_slope_raw": latency_result["slope"],
            "latency_slope_SE": latency_result["slope_SE"],
            "latency_slope_t": latency_result["t"],
            "latency_slope_p": latency_result["p"],
            "latency_slope_CI_lower": latency_result["CI_lower"],
            "latency_slope_CI_upper": latency_result["CI_upper"],
            "latency_r_squared": latency_result["r_squared"]
        }

        individual_rows.append(row)

    individual_slopes = pd.DataFrame(individual_rows)


    # ============================================================
    # 8. 构建方向统一后的动态难度响应斜率
    # ============================================================

    # 表征维持斜率：
    # 原始 slope 越高，表示难度越高时表征能力越强，方向为“越高越好”。
    individual_slopes["b_representation_maintenance"] = individual_slopes["rep_slope_raw"]

    # 加工稳定性维持斜率：
    # 原始 noise slope 越高，表示难度越高时噪声越大、稳定性越差。
    # 因此若噪声越高越差，则取负号，使其方向变成“越高越好”。
    if PROCESSING_NOISE_HIGHER_IS_WORSE:
        individual_slopes["b_processing_stability"] = -individual_slopes["noise_slope_raw"]
    else:
        individual_slopes["b_processing_stability"] = individual_slopes["noise_slope_raw"]

    # 提取效率保持斜率：
    # 如果原始 retrieval 是潜伏期，slope 越高表示难度越高时提取越慢。
    # 因此取负号，使其方向变成“越高越好”。
    if RETRIEVAL_IS_LATENCY:
        individual_slopes["b_retrieval_efficiency"] = -individual_slopes["latency_slope_raw"]
    else:
        individual_slopes["b_retrieval_efficiency"] = individual_slopes["latency_slope_raw"]


    # 标准化后的动态特征，便于后续聚类
    dynamic_good_cols = [
        "b_representation_maintenance",
        "b_processing_stability",
        "b_retrieval_efficiency"
    ]

    for col in dynamic_good_cols:
        individual_slopes[f"z_{col}"] = zscore(individual_slopes[col])

    save_csv(individual_slopes, "individual_difficulty_response_slopes.csv")


    # ============================================================
    # 9. 合并静态特征与动态斜率，形成 6 维个体表征
    # ============================================================

    subject_features = static_features.merge(
        individual_slopes[
            [
                "subject",
                "rep_slope_raw",
                "noise_slope_raw",
                "latency_slope_raw",
                "b_representation_maintenance",
                "b_processing_stability",
                "b_retrieval_efficiency",
                "z_b_representation_maintenance",
                "z_b_processing_stability",
                "z_b_retrieval_efficiency"
            ]
        ],
        on="subject",
        how="left"
    )

    # 6维标准化特征：
    # 1. z_mean_memory_representation
    # 2. z_mean_processing_stability
    # 3. z_mean_retrieval_efficiency
    # 4. z_b_representation_maintenance
    # 5. z_b_processing_stability
    # 6. z_b_retrieval_efficiency

    six_feature_cols = [
        "z_mean_memory_representation",
        "z_mean_processing_stability",
        "z_mean_retrieval_efficiency",
        "z_b_representation_maintenance",
        "z_b_processing_stability",
        "z_b_retrieval_efficiency"
    ]

    save_csv(subject_features, "subject_static_dynamic_6_features.csv")


    # ============================================================
    # 10. 个体斜率描述性统计
    # ============================================================

    slope_labels = {
        "rep_slope_raw": "Raw representation slope",
        "noise_slope_raw": "Raw processing noise slope",
        "latency_slope_raw": "Raw retrieval latency slope",
        "b_representation_maintenance": "Representation maintenance slope",
        "b_processing_stability": "Processing stability slope",
        "b_retrieval_efficiency": "Retrieval efficiency slope"
    }

    desc_rows = []

    for col, label in slope_labels.items():
        stat = descriptive_stats(individual_slopes[col])
        stat["variable"] = col
        stat["label"] = label
        desc_rows.append(stat)

    slope_descriptive = pd.DataFrame(desc_rows)[
        [
            "variable",
            "label",
            "n",
            "mean",
            "std",
            "median",
            "q25",
            "q75",
            "min",
            "max"
        ]
    ]

    save_csv(slope_descriptive, "slope_descriptive_statistics.csv")

    print("\n===== 个体难度响应斜率描述性统计 =====")
    print(slope_descriptive.round(4).to_string(index=False))


    # ============================================================
    # 11. 正负响应比例
    # ============================================================

    sign_rows = []

    for col, label in slope_labels.items():

        s = individual_slopes[col].dropna()
        n = len(s)

        n_positive = int((s > 0).sum())
        n_negative = int((s < 0).sum())
        n_zero = int((s == 0).sum())

        sign_rows.append({
            "variable": col,
            "label": label,
            "n_valid": n,
            "n_positive": n_positive,
            "pct_positive": n_positive / n * 100 if n > 0 else np.nan,
            "n_negative": n_negative,
            "pct_negative": n_negative / n * 100 if n > 0 else np.nan,
            "n_zero": n_zero,
            "pct_zero": n_zero / n * 100 if n > 0 else np.nan
        })

    slope_sign_summary = pd.DataFrame(sign_rows)

    save_csv(slope_sign_summary, "slope_sign_response_summary.csv")

    print("\n===== 个体斜率正负响应比例 =====")
    print(slope_sign_summary.round(2).to_string(index=False))


    # ============================================================
    # 12. 与群体混合效应模型固定效应对照
    # ============================================================

    print("\n开始拟合群体混合效应模型，用于与个体斜率均值对照...")

    mixed_rep = fit_mixedlm(
        "memory_representation ~ difficulty_z",
        data=df
    )

    mixed_noise = fit_mixedlm(
        "processing_noise ~ difficulty_z",
        data=df
    )

    mixed_latency = fit_mixedlm(
        "retrieval_latency ~ difficulty_z",
        data=df
    )

    mixed_compare = pd.DataFrame([
        {
            "outcome": "Memory representation",
            "mixed_model_beta": mixed_rep.params["difficulty_z"],
            "mixed_model_SE": mixed_rep.bse["difficulty_z"],
            "individual_slope_mean": individual_slopes["rep_slope_raw"].mean(),
            "individual_slope_SD": individual_slopes["rep_slope_raw"].std(ddof=1),
            "n_subjects": individual_slopes["rep_slope_raw"].notna().sum()
        },
        {
            "outcome": "Processing noise",
            "mixed_model_beta": mixed_noise.params["difficulty_z"],
            "mixed_model_SE": mixed_noise.bse["difficulty_z"],
            "individual_slope_mean": individual_slopes["noise_slope_raw"].mean(),
            "individual_slope_SD": individual_slopes["noise_slope_raw"].std(ddof=1),
            "n_subjects": individual_slopes["noise_slope_raw"].notna().sum()
        },
        {
            "outcome": "Retrieval latency",
            "mixed_model_beta": mixed_latency.params["difficulty_z"],
            "mixed_model_SE": mixed_latency.bse["difficulty_z"],
            "individual_slope_mean": individual_slopes["latency_slope_raw"].mean(),
            "individual_slope_SD": individual_slopes["latency_slope_raw"].std(ddof=1),
            "n_subjects": individual_slopes["latency_slope_raw"].notna().sum()
        }
    ])

    mixed_compare["difference_mean_minus_mixed_beta"] = (
        mixed_compare["individual_slope_mean"] - mixed_compare["mixed_model_beta"]
    )

    save_csv(mixed_compare, "mixed_model_vs_individual_slope_mean.csv")

    print("\n===== 群体混合模型固定效应 vs 个体斜率均值 =====")
    print(mixed_compare.round(4).to_string(index=False))


    # ============================================================
    # 12.1 新增：参考论文 5.20 对应的个体异质性统计表
    # ============================================================

    # 该表专门服务于论文中：
    # “基于上述个体水平回归模型，本研究进一步检验群体平均效应
    # 是否能够概括所有被试的适应模式。”
    # 主表保留原始斜率，以便与 3.2.1 的群体混合模型固定效应逐一对应；
    # 同时附加方向统一后的斜率，供后续静态-动态特征构建与聚类使用。
    heterogeneity_metric_specs = [
        {
            "outcome_id": "representation",
            "outcome": "Memory representation",
            "raw_slope_column": "rep_slope_raw",
            "adaptive_slope_column": "b_representation_maintenance",
            "mixed_model_beta": mixed_rep.params["difficulty_z"],
            "mixed_model_SE": mixed_rep.bse["difficulty_z"],
            "adaptive_multiplier": 1.0
        },
        {
            "outcome_id": "processing_noise",
            "outcome": "Processing noise",
            "raw_slope_column": "noise_slope_raw",
            "adaptive_slope_column": "b_processing_stability",
            "mixed_model_beta": mixed_noise.params["difficulty_z"],
            "mixed_model_SE": mixed_noise.bse["difficulty_z"],
            "adaptive_multiplier": -1.0 if PROCESSING_NOISE_HIGHER_IS_WORSE else 1.0
        },
        {
            "outcome_id": "retrieval_latency",
            "outcome": "Retrieval latency",
            "raw_slope_column": "latency_slope_raw",
            "adaptive_slope_column": "b_retrieval_efficiency",
            "mixed_model_beta": mixed_latency.params["difficulty_z"],
            "mixed_model_SE": mixed_latency.bse["difficulty_z"],
            "adaptive_multiplier": -1.0 if RETRIEVAL_IS_LATENCY else 1.0
        }
    ]

    heterogeneity_rows = []
    for spec in heterogeneity_metric_specs:
        raw_stats = describe_slope(individual_slopes[spec["raw_slope_column"]])
        adaptive_stats = describe_slope(individual_slopes[spec["adaptive_slope_column"]])
        mixed_beta_adaptive = spec["mixed_model_beta"] * spec["adaptive_multiplier"]

        heterogeneity_rows.append({
            "outcome_id": spec["outcome_id"],
            "outcome": spec["outcome"],
            "raw_slope_column": spec["raw_slope_column"],
            "adaptive_slope_column": spec["adaptive_slope_column"],
            "mixed_model_beta": spec["mixed_model_beta"],
            "mixed_model_SE": spec["mixed_model_SE"],
            "individual_slope_mean": raw_stats["mean"],
            "individual_slope_sd": raw_stats["sd"],
            "cv_abs_mean": raw_stats["cv_abs_mean"],
            "difference_individual_mean_minus_mixed_beta": raw_stats["mean"] - spec["mixed_model_beta"],
            "same_direction_as_mixed_beta": (
                np.sign(raw_stats["mean"]) == np.sign(spec["mixed_model_beta"])
                if not pd.isna(raw_stats["mean"]) and not pd.isna(spec["mixed_model_beta"])
                else np.nan
            ),
            "n_valid": raw_stats["n"],
            "n_positive": raw_stats["n_positive"],
            "pct_positive": raw_stats["pct_positive"],
            "n_negative": raw_stats["n_negative"],
            "pct_negative": raw_stats["pct_negative"],
            "n_zero": raw_stats["n_zero"],
            "pct_zero": raw_stats["pct_zero"],
            "adaptive_mixed_model_beta": mixed_beta_adaptive,
            "adaptive_slope_mean": adaptive_stats["mean"],
            "adaptive_slope_sd": adaptive_stats["sd"],
            "adaptive_n_positive": adaptive_stats["n_positive"],
            "adaptive_pct_positive": adaptive_stats["pct_positive"],
            "adaptive_n_negative": adaptive_stats["n_negative"],
            "adaptive_pct_negative": adaptive_stats["pct_negative"]
        })

    heterogeneity_summary = pd.DataFrame(heterogeneity_rows)
    save_csv(
        heterogeneity_summary,
        "individual_heterogeneity_summary_for_section_3_2_2.csv"
    )

    print("\n===== 3.2 个体异质性：论文用统计汇总 =====")
    print(heterogeneity_summary.round(6).to_string(index=False))


    # ============================================================
    # 13. 三个动态斜率之间的相关
    # ============================================================

    dynamic_corr_pearson = correlation_table(
        individual_slopes,
        variables=dynamic_good_cols,
        method="pearson"
    )

    dynamic_corr_spearman = correlation_table(
        individual_slopes,
        variables=dynamic_good_cols,
        method="spearman"
    )

    dynamic_corr = pd.concat(
        [dynamic_corr_pearson, dynamic_corr_spearman],
        ignore_index=True
    )

    save_csv(dynamic_corr, "correlation_among_dynamic_slopes.csv")

    print("\n===== 三个动态难度响应斜率之间的相关 =====")
    print(dynamic_corr.round(4).to_string(index=False))


    # ============================================================
    # 14. 静态基础能力与动态斜率之间的相关
    # ============================================================

    static_good_cols = [
        "z_mean_memory_representation",
        "z_mean_processing_stability",
        "z_mean_retrieval_efficiency"
    ]

    static_dynamic_corr_rows = []

    for s_col in static_good_cols:
        for d_col in dynamic_good_cols:

            temp = subject_features[[s_col, d_col]].dropna()

            if len(temp) < 3:
                r = np.nan
                p = np.nan
                n = len(temp)
            else:
                r, p = stats.pearsonr(temp[s_col], temp[d_col])
                n = len(temp)

            static_dynamic_corr_rows.append({
                "static_feature": s_col,
                "dynamic_slope": d_col,
                "method": "pearson",
                "n": n,
                "r": r,
                "p": p
            })

    static_dynamic_corr = pd.DataFrame(static_dynamic_corr_rows)

    save_csv(static_dynamic_corr, "correlation_static_features_with_dynamic_slopes.csv")

    print("\n===== 静态基础能力与动态难度响应斜率之间的相关 =====")
    print(static_dynamic_corr.round(4).to_string(index=False))


    # ============================================================
    # 15. 输出 6 维特征的相关矩阵
    # ============================================================

    six_feature_corr = subject_features[six_feature_cols].corr(method="pearson")

    six_feature_corr.to_csv(
        os.path.join(OUTPUT_DIR, "six_feature_correlation_matrix.csv"),
        encoding="utf-8-sig"
    )

    print("\n===== 6维静态-动态特征相关矩阵 =====")
    print(six_feature_corr.round(4).to_string())


    # ============================================================
    # 16. Figure 1：方向统一后的个体难度响应异质性
    # ============================================================

    # 方向统一后的动态适应斜率。
    # 三个维度均统一为“越高越好”，用于后续静态-动态特征分析。
    figure1_metric_specs = [
        {
            "column": "b_representation_maintenance",
            "axis_label": r"Memory representation ability slope",
            "panel_title": "Memory Representation Ability"
        },
        {
            "column": "b_processing_stability",
            "axis_label": r"Cognitive processing noise slope",
            "panel_title": "Cognitive Processing Noise Level"
        },
        {
            "column": "b_retrieval_efficiency",
            "axis_label": r"Memory retrieval latency slope",
            "panel_title": "Memory Retrieval Latency"
        }
    ]

    figure1_path = os.path.join(
        OUTPUT_DIR,
        "Figure1_individual_heterogeneity.png"
    )
    plot_individual_heterogeneity_figure(
        slope_df=individual_slopes,
        metric_specs=figure1_metric_specs,
        output_path=figure1_path
    )


    # ============================================================
    # 17. 保存模型摘要与文字结果
    # ============================================================

    summary_txt_path = os.path.join(
        OUTPUT_DIR,
        "analysis_summary_2_2.txt"
    )

    with open(summary_txt_path, "w", encoding="utf-8") as f:

        f.write("难度调节效应的个体异质性\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. 数据质控\n")
        f.write(qc_summary.to_string(index=False))
        f.write("\n\n")

        f.write("2. 个体斜率描述性统计\n")
        f.write(slope_descriptive.round(6).to_string(index=False))
        f.write("\n\n")

        f.write("3. 个体斜率正负响应比例\n")
        f.write(slope_sign_summary.round(6).to_string(index=False))
        f.write("\n\n")

        f.write("4. 群体混合模型固定效应 vs 个体斜率均值\n")
        f.write(mixed_compare.round(6).to_string(index=False))
        f.write("\n\n")

        f.write("4.1 论文 3.2 个体异质性专用统计表\n")
        f.write(heterogeneity_summary.round(6).to_string(index=False))
        f.write("\n\n")

        f.write("5. 三个动态难度响应斜率之间的相关\n")
        f.write(dynamic_corr.round(6).to_string(index=False))
        f.write("\n\n")

        f.write("6. 静态基础能力与动态难度响应斜率之间的相关\n")
        f.write(static_dynamic_corr.round(6).to_string(index=False))
        f.write("\n\n")

        f.write("7. 6维静态-动态特征相关矩阵\n")
        f.write(six_feature_corr.round(6).to_string())
        f.write("\n\n")

        f.write("8. 群体混合效应模型摘要\n\n")

        f.write("===== Mixed model: Memory representation =====\n")
        f.write(str(mixed_rep.summary()))
        f.write("\n\n")

        f.write("===== Mixed model: Processing noise =====\n")
        f.write(str(mixed_noise.summary()))
        f.write("\n\n")

        f.write("===== Mixed model: Retrieval latency =====\n")
        f.write(str(mixed_latency.summary()))
        f.write("\n\n")


    # ============================================================
    # 18. 打包所有输出文件
    # ============================================================

    with zipfile.ZipFile(ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(OUTPUT_DIR):
            for file in files:
                full_path = os.path.join(root, file)
                arcname = os.path.relpath(full_path, OUTPUT_DIR)
                zf.write(full_path, arcname=arcname)


    # ============================================================
    # 19. 控制台输出简短结论
    # ============================================================

    print("\n" + "=" * 80)
    print("VPC 2 个体异质性分析完成")
    print("=" * 80)

    print(f"\n结果文件夹：{OUTPUT_DIR}")
    print(f"压缩包：{ZIP_NAME}")

    print("\n核心输出文件：")
    print("1. QC_summary_2_2.csv")
    print("2. static_subject_features.csv")
    print("3. individual_difficulty_response_slopes.csv")
    print("4. subject_static_dynamic_6_features.csv")
    print("5. slope_descriptive_statistics.csv")
    print("6. slope_sign_response_summary.csv")
    print("7. mixed_model_vs_individual_slope_mean.csv")
    print("8. individual_heterogeneity_summary_for_section_3_2_2.csv")
    print("9. correlation_among_dynamic_slopes.csv")
    print("10. correlation_static_features_with_dynamic_slopes.csv")
    print("11. six_feature_correlation_matrix.csv")
    print("12. analysis_summary_2_2.txt")
    print("13. Figure1_individual_heterogeneity.png")
    print("14. Figure1_individual_heterogeneity.pdf")

    print("\n简要结果预览：")

    print("\n[个体斜率描述性统计]")
    print(slope_descriptive.round(4).to_string(index=False))

    print("\n[方向统一后的动态斜率正负响应比例]")
    good_sign_summary = slope_sign_summary[
        slope_sign_summary["variable"].isin(dynamic_good_cols)
    ]
    print(good_sign_summary.round(2).to_string(index=False))

    print("\n[动态斜率之间 Pearson/Spearman 相关]")
    print(dynamic_corr.round(4).to_string(index=False))

    print("\n全部分析已完成。")


if __name__ == "__main__":
    main()

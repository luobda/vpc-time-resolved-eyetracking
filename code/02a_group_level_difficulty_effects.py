# -*- coding: utf-8 -*-
"""
VPC 论文 2.1：
难度对视觉记忆核心指标的整体调节效应

功能：
1. 读取逐试次 VPC 指标数据；
2. 完成数据质控与描述性统计；
3. 分别建立三个线性混合效应模型：
   - 记忆表征能力 ~ difficulty
   - 认知加工噪声水平 ~ difficulty
   - 记忆提取速率/潜伏期 ~ difficulty
4. 建立综合标准化模型，检验 difficulty × 指标类型 交互；
5. 进行二次项稳健性检验；
6. 输出 CSV 结果表、模型摘要 TXT 和 ZIP 压缩包。

依赖：
pip install pandas numpy scipy statsmodels openpyxl
"""

import os
import sys
import zipfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")


class Tee:
    """将标准输出同时写入控制台和日志文件。"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()



# ============================================================
# 0. 参数设置
# ======================== 路径配置 ========================
# 以当前脚本所在的 code 文件夹为基准，自动定位同级的 data 文件夹
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = (SCRIPT_DIR / ".." / "data").resolve()

INPUT_FILE = DATA_DIR / "fitting_results_difficulty.xlsx"

OUTPUT_DIR = "Chapter2_Output_1"
ZIP_NAME = "Chapter2_Output_1.zip"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. 工具函数
# ============================================================

def zscore(series):
    """标准化为均值0、标准差1。"""
    return (series - series.mean()) / series.std(ddof=0)


def fit_mixedlm(formula, data, group_col="subject"):
    """
    拟合随机截距线性混合效应模型。
    若默认优化器失败，则自动尝试多个优化器。
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
            else:
                # 有时 result 不完全收敛但仍可返回；
                # 这里优先继续尝试其他优化器。
                last_error = RuntimeError(f"Model did not converge with {method}")
        except Exception as e:
            last_error = e

    raise RuntimeError(f"MixedLM failed for formula: {formula}\nLast error: {last_error}")


def get_fixed_effect_row(result, outcome_name, term="difficulty_z"):
    """提取混合效应模型中指定固定效应的结果。"""
    ci = result.conf_int()

    return {
        "outcome": outcome_name,
        "term": term,
        "beta": result.params[term],
        "SE": result.bse[term],
        "z": result.tvalues[term],
        "p": result.pvalues[term],
        "CI_lower": ci.loc[term, 0],
        "CI_upper": ci.loc[term, 1],
        "AIC": result.aic,
        "BIC": result.bic,
        "logLik": result.llf,
        "n_obs": int(result.nobs)
    }


def likelihood_ratio_test(full_model, reduced_model):
    """似然比检验。"""
    lr_stat = 2 * (full_model.llf - reduced_model.llf)
    df_diff = full_model.df_modelwc - reduced_model.df_modelwc
    p_value = stats.chi2.sf(lr_stat, df_diff)

    return lr_stat, df_diff, p_value


def save_csv(df, filename):
    """保存 CSV，使用 utf-8-sig 方便 Excel 打开中文。"""
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path

def run_analysis():
    """执行完整统计分析并保存非图形结果。"""
    # ============================================================
    # 2. 读取数据
    # ============================================================

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"找不到输入文件：{INPUT_FILE}\n"
            f"请确认 Excel 文件与当前 Python 脚本位于同一文件夹，"
            f"或修改 INPUT_FILE 为完整路径。"
        )

    raw_df = pd.read_excel(INPUT_FILE)

    print("\n原始列名：")
    print(list(raw_df.columns))


    # ============================================================
    # 3. 重命名列
    # ============================================================

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
            + "\n\n请检查 Excel 表头是否与代码中的 rename_map 一致。"
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

    # difficulty 标准化
    df["difficulty_z"] = zscore(df["difficulty"])
    df["difficulty_z2"] = df["difficulty_z"] ** 2

    # 三个指标方向统一：
    # 记忆表征能力：越高越好
    # 加工稳定性：噪声越低越好，所以取反
    # 提取效率：原始“记忆提取速率”是峰值出现时间/潜伏期，越低越快，所以取反
    df["memory_representation_z"] = zscore(df["memory_representation"])
    df["processing_stability_z"] = -zscore(df["processing_noise"])
    df["retrieval_efficiency_z"] = -zscore(df["retrieval_latency"])


    # ============================================================
    # 5. 数据质控与描述性统计
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
            "missing_values_in_key_columns"
        ],
        "value": [
            len(df),
            df["subject"].nunique(),
            df["trial"].nunique(),
            trial_counts.min(),
            trial_counts.median(),
            trial_counts.max(),
            int(df[required_cols].isna().sum().sum())
        ]
    })

    descriptive = (
        df[
            [
                "difficulty",
                "memory_representation",
                "processing_noise",
                "retrieval_latency"
            ]
        ]
        .describe()
        .T
    )

    descriptive["cv"] = descriptive["std"] / descriptive["mean"]

    qc_summary.to_csv(
        os.path.join(OUTPUT_DIR, "QC_summary.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    descriptive.to_csv(
        os.path.join(OUTPUT_DIR, "descriptive_statistics.csv"),
        encoding="utf-8-sig"
    )

    print("\n===== 数据质控 =====")
    print(qc_summary.to_string(index=False))

    print("\n===== 描述性统计 =====")
    print(descriptive.round(4).to_string())


    # ============================================================
    # 6. 主分析：三个线性混合效应模型
    # ============================================================

    print("\n开始拟合三个主模型...")

    model_rep = fit_mixedlm(
        "memory_representation ~ difficulty_z",
        data=df
    )

    model_noise = fit_mixedlm(
        "processing_noise ~ difficulty_z",
        data=df
    )

    model_latency = fit_mixedlm(
        "retrieval_latency ~ difficulty_z",
        data=df
    )

    main_model_table = pd.DataFrame([
        get_fixed_effect_row(
            model_rep,
            outcome_name="Memory representation",
            term="difficulty_z"
        ),
        get_fixed_effect_row(
            model_noise,
            outcome_name="Processing noise",
            term="difficulty_z"
        ),
        get_fixed_effect_row(
            model_latency,
            outcome_name="Retrieval latency",
            term="difficulty_z"
        )
    ])

    main_model_table["n_subjects"] = df["subject"].nunique()

    save_csv(main_model_table, "main_mixed_effect_models.csv")

    print("\n===== 2.1 主模型：difficulty 每增加 1 个标准差的固定效应 =====")
    print(main_model_table.round(4).to_string(index=False))


    # ============================================================
    # 7. 综合标准化模型：检验 difficulty × 指标类型
    # ============================================================

    print("\n开始拟合综合标准化模型...")

    long_df = pd.melt(
        df,
        id_vars=["subject", "trial", "difficulty", "difficulty_z"],
        value_vars=[
            "memory_representation_z",
            "processing_stability_z",
            "retrieval_efficiency_z"
        ],
        var_name="metric_type",
        value_name="performance_z"
    )

    metric_order = [
        "memory_representation_z",
        "processing_stability_z",
        "retrieval_efficiency_z"
    ]

    long_df["metric_type"] = pd.Categorical(
        long_df["metric_type"],
        categories=metric_order,
        ordered=True
    )

    full_model = fit_mixedlm(
        "performance_z ~ difficulty_z * C(metric_type)",
        data=long_df
    )

    reduced_model = fit_mixedlm(
        "performance_z ~ difficulty_z + C(metric_type)",
        data=long_df
    )

    lr_stat, lr_df, lr_p = likelihood_ratio_test(full_model, reduced_model)

    interaction_test = pd.DataFrame([{
        "comparison": "full model with difficulty × metric_type vs reduced model without interaction",
        "LR_chi2": lr_stat,
        "df": lr_df,
        "p": lr_p,
        "AIC_full": full_model.aic,
        "AIC_reduced": reduced_model.aic,
        "BIC_full": full_model.bic,
        "BIC_reduced": reduced_model.bic
    }])

    save_csv(interaction_test, "metric_type_interaction_test.csv")

    print("\n===== 指标类型差异：difficulty × metric_type 交互检验 =====")
    print(interaction_test.round(4).to_string(index=False))


    # ============================================================
    # 8. 提取综合模型中三个指标各自的标准化难度斜率
    # ============================================================

    metric_slopes = []

    base_slope = full_model.params["difficulty_z"]
    cov_matrix = full_model.cov_params()

    for metric in metric_order:

        if metric == "memory_representation_z":
            slope = base_slope
            se = full_model.bse["difficulty_z"]

        else:
            interaction_term = f"difficulty_z:C(metric_type)[T.{metric}]"
            slope = base_slope + full_model.params[interaction_term]

            var = (
                cov_matrix.loc["difficulty_z", "difficulty_z"]
                + cov_matrix.loc[interaction_term, interaction_term]
                + 2 * cov_matrix.loc["difficulty_z", interaction_term]
            )

            se = np.sqrt(var)

        z_value = slope / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_value)))

        metric_slopes.append({
            "metric": metric,
            "slope_per_1SD_difficulty": slope,
            "SE": se,
            "z": z_value,
            "p": p_value,
            "CI_lower": slope - 1.96 * se,
            "CI_upper": slope + 1.96 * se
        })

    metric_slope_table = pd.DataFrame(metric_slopes)

    metric_name_map = {
        "memory_representation_z": "Memory representation",
        "processing_stability_z": "Processing stability",
        "retrieval_efficiency_z": "Retrieval efficiency"
    }

    metric_slope_table["metric_label"] = metric_slope_table["metric"].map(metric_name_map)

    save_csv(metric_slope_table, "standardized_metric_slopes.csv")

    print("\n===== 标准化后各指标的难度斜率，方向统一为越高越好 =====")
    print(metric_slope_table.round(4).to_string(index=False))


    # ============================================================
    # 9. 二次项稳健性检验
    # ============================================================

    print("\n开始二次项稳健性检验...")

    quadratic_rows = []

    outcome_map = {
        "memory_representation": "Memory representation",
        "processing_noise": "Processing noise",
        "retrieval_latency": "Retrieval latency"
    }

    for outcome, label in outcome_map.items():

        linear_model = fit_mixedlm(
            f"{outcome} ~ difficulty_z",
            data=df
        )

        quadratic_model = fit_mixedlm(
            f"{outcome} ~ difficulty_z + difficulty_z2",
            data=df
        )

        lr, df_diff, p_value = likelihood_ratio_test(
            full_model=quadratic_model,
            reduced_model=linear_model
        )

        ci = quadratic_model.conf_int()

        quadratic_rows.append({
            "outcome": label,
            "LR_chi2_for_quadratic_term": lr,
            "df": df_diff,
            "p_LRT": p_value,
            "quadratic_beta": quadratic_model.params["difficulty_z2"],
            "quadratic_SE": quadratic_model.bse["difficulty_z2"],
            "quadratic_z": quadratic_model.tvalues["difficulty_z2"],
            "quadratic_p": quadratic_model.pvalues["difficulty_z2"],
            "quadratic_CI_lower": ci.loc["difficulty_z2", 0],
            "quadratic_CI_upper": ci.loc["difficulty_z2", 1],
            "AIC_linear": linear_model.aic,
            "AIC_quadratic": quadratic_model.aic,
            "BIC_linear": linear_model.bic,
            "BIC_quadratic": quadratic_model.bic
        })

    quadratic_test_table = pd.DataFrame(quadratic_rows)

    save_csv(quadratic_test_table, "quadratic_robustness_tests.csv")

    print("\n===== 二次项稳健性检验 =====")
    print(quadratic_test_table.round(4).to_string(index=False))


    # ============================================================
    # 10. 保存模型摘要为 txt
    # ============================================================



    with open(
        os.path.join(OUTPUT_DIR, "model_summaries.txt"),
        "w",
        encoding="utf-8"
    ) as f:
        f.write("===== Model 1: Memory representation =====\n")
        f.write(str(model_rep.summary()))
        f.write("\n\n")

        f.write("===== Model 2: Processing noise =====\n")
        f.write(str(model_noise.summary()))
        f.write("\n\n")

        f.write("===== Model 3: Retrieval latency =====\n")
        f.write(str(model_latency.summary()))
        f.write("\n\n")

        f.write("===== Full standardized model =====\n")
        f.write(str(full_model.summary()))
        f.write("\n\n")

        f.write("===== Reduced standardized model =====\n")
        f.write(str(reduced_model.summary()))
        f.write("\n\n")


    # ============================================================
    # 11. 打包所有输出文件
    # ============================================================

    with zipfile.ZipFile(ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(OUTPUT_DIR):
            for file in files:
                full_path = os.path.join(root, file)
                arcname = os.path.relpath(full_path, OUTPUT_DIR)
                zf.write(full_path, arcname=arcname)


    # ============================================================
    # 12. 输出简短结论
    # ============================================================

    print("\n" + "=" * 70)
    print("分析完成")
    print("=" * 70)

    print(f"\n结果文件夹：{OUTPUT_DIR}")
    print(f"压缩包：{ZIP_NAME}")

    print("\n已生成以下核心文件：")
    print("1. QC_summary.csv")
    print("2. descriptive_statistics.csv")
    print("3. main_mixed_effect_models.csv")
    print("4. metric_type_interaction_test.csv")
    print("5. standardized_metric_slopes.csv")
    print("6. quadratic_robustness_tests.csv")
    print("7. model_summaries.txt")
    print("8. console_output.txt")

    print("\n核心结果简述：")

    for _, row in main_model_table.iterrows():
        direction = "正向" if row["beta"] > 0 else "负向"
        sig = "显著" if row["p"] < 0.05 else "不显著"

        print(
            f"- {row['outcome']}: difficulty_z 的效应为 {direction}，"
            f"beta = {row['beta']:.4f}, "
            f"SE = {row['SE']:.4f}, "
            f"z = {row['z']:.2f}, "
            f"p = {row['p']:.4g}，{sig}。"
        )

    print(
        f"\n综合模型 difficulty × metric_type 交互："
        f"χ²({int(lr_df)}) = {lr_stat:.2f}, p = {lr_p:.4g}。"
    )

    print("\n全部分析已完成。")

def main():
    """运行分析，并将全部控制台输出同步保存到 TXT。"""
    log_path = os.path.join(OUTPUT_DIR, "console_output.txt")
    original_stdout = sys.stdout

    with open(log_path, "w", encoding="utf-8") as log_file:
        sys.stdout = Tee(original_stdout, log_file)
        try:
            run_analysis()
        finally:
            sys.stdout = original_stdout

    # run_analysis() 内部已按原逻辑生成 ZIP；
    # 此处重新打包一次，仅用于把完整 console_output.txt 一并纳入 ZIP。
    with zipfile.ZipFile(ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(OUTPUT_DIR):
            for file in files:
                full_path = os.path.join(root, file)
                arcname = os.path.relpath(full_path, OUTPUT_DIR)
                zf.write(full_path, arcname=arcname)


if __name__ == "__main__":
    main()

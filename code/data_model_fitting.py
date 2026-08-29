import pandas as pd
import numpy as np
import os
from pathlib import Path
from numpy.polynomial.chebyshev import chebfit, chebval
import warnings
from scipy.signal import find_peaks

warnings.filterwarnings("ignore")

# ======================== 配置项 ========================

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = (SCRIPT_DIR / ".." / "data").resolve()

# 质量控制后的数据根目录
FILTERED_DATA_ROOT = DATA_DIR / "filtered_data"

# 三个数据子文件夹与参数文件一一对应
DATASET_CONFIG = {
    "data1": DATA_DIR / "task_parameters_data1.csv",
    "data2": DATA_DIR / "task_parameters_data2.csv",
    "data3": DATA_DIR / "task_parameters_data3.csv",
}

# 切比雪夫拟合阶数：保持原程序不变
CHEBYSHEV_ORDER = 10

# 所有被试、所有试次结果汇总到一个 CSV
OUTPUT_FILE_PATH = DATA_DIR / "fitting_results.xlsx"
# =======================================================


def load_and_process_t1(eyetrack_path):
    """处理原始眼动数据，标记试次和阶段（与原版完全相同）"""
    df = pd.read_csv(eyetrack_path)[
        ['timestamp', 'bino_eye_gaze_position_x', 'bino_eye_gaze_position_y',
         'bino_eye_valid', 'trigger']]
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['trial'] = 0
    df['familiar_phase'] = 0
    df['forgetting_phase'] = 0
    df['test_phase'] = 0

    valid_triggers = []
    for t in df['trigger'].unique():
        if pd.isna(t): continue
        t_str = ''.join(filter(str.isdigit, str(t).strip()))
        if not t_str: continue
        t_num = int(t_str)
        if (t_num % 100) in [1, 2, 3, 4]:
            valid_triggers.append(t_num)
    valid_triggers = sorted(valid_triggers)

    if len(valid_triggers) < 4:
        raise ValueError(f"有效trigger不足，当前识别：{valid_triggers}")

    for i in range(0, len(valid_triggers), 4):
        try:
            trig_list = valid_triggers[i:i + 4]

            def get_time(trig):
                mask = df['trigger'].apply(
                    lambda x: int(''.join(filter(str.isdigit, str(x).strip()))) == trig)
                return df[mask]['timestamp'].iloc[0] if len(df[mask]) > 0 else None

            times = [get_time(t) for t in trig_list]
            if not all(times): continue

            t_fam_start, t_fam_end, t_forget_end, t_test_end = times
            current_trial = trig_list[0] // 100

            df.loc[(df['timestamp'] >= t_fam_start) & (df['timestamp'] < t_fam_end),
                   'trial'] = current_trial
            df.loc[(df['timestamp'] >= t_fam_start) & (df['timestamp'] < t_fam_end),
                   'familiar_phase'] = 1
            df.loc[(df['timestamp'] >= t_fam_end) & (df['timestamp'] < t_forget_end),
                   'trial'] = current_trial
            df.loc[(df['timestamp'] >= t_fam_end) & (df['timestamp'] < t_forget_end),
                   'forgetting_phase'] = 1
            df.loc[(df['timestamp'] >= t_forget_end) & (df['timestamp'] <= t_test_end),
                   'trial'] = current_trial
            df.loc[(df['timestamp'] >= t_forget_end) & (df['timestamp'] <= t_test_end),
                   'test_phase'] = 1
        except IndexError:
            continue

    df_valid = df[df['trial'] > 0].reset_index(drop=True)
    if len(df_valid) == 0:
        raise ValueError("无有效试次数据")
    return df_valid


def merge_params_and_gaze(t1_df, params_path):
    """合并参数文件并判断注视位置（与原版完全相同）"""
    try:
        params_df = pd.read_csv(params_path, encoding='gbk')
    except Exception:
        params_df = pd.read_csv(params_path, encoding='utf-8-sig')

    trial_map = {}
    for _, row in params_df.iterrows():
        trial   = int(row['轮次'])
        new_pic  = str(row['新图片']).strip()
        left_pic = str(row['左边图片']).strip()
        right_pic= str(row['右边图片']).strip()
        trial_map[trial] = (1 if left_pic == new_pic else 0,
                             1 if right_pic == new_pic else 0)

    t2_df = t1_df.copy()
    t2_df['left_new'] = 0
    t2_df['right_new'] = 0
    for trial in t2_df['trial'].unique():
        if trial in trial_map:
            l, r = trial_map[trial]
            t2_df.loc[t2_df['trial'] == trial, 'left_new'] = l
            t2_df.loc[t2_df['trial'] == trial, 'right_new'] = r

    def judge(x, y):
        if pd.isna(x) or pd.isna(y): return 0, 0, 1
        left  = 1 if (96  <= x <= 864)  and (270 <= y <= 810) else 0
        right = 1 if (1056 <= x <= 1824) and (270 <= y <= 810) else 0
        return left, right, 1 if (left == 0 and right == 0) else 0

    gaze = t2_df.apply(
        lambda row: judge(row['bino_eye_gaze_position_x'],
                          row['bino_eye_gaze_position_y']),
        axis=1, result_type='expand')
    t2_df[['fix_left_pic', 'fix_right_pic', 'fix_other']] = gaze
    return t2_df


def chebyshev_fit_smooth(t, y, order=5):
    """切比雪夫多项式拟合（与原版完全相同）"""
    t_norm = 2 * (t - t.min()) / (t.max() - t.min() + 1e-8) - 1
    coeffs = chebfit(t_norm, y, deg=order)
    z = chebval(t_norm, coeffs)
    return np.tanh(z), coeffs


def run_chebyshev_fit(t2_df, order=5):
    """切比雪夫拟合，返回拟合DataFrame（与原版完全相同）"""
    df = t2_df
    valid_trials = sorted(df[df.test_phase == 1]['trial'].unique())
    fit_rows = []
    for trial in valid_trials:
        d = df[(df.trial == trial) & (df.test_phase == 1)].sort_values('timestamp')
        if len(d) <= order: continue

        t     = d.timestamp.values / 1e9
        t_rel = t - t.min()
        gaze  = d.apply(
            lambda r: 1 if r.fix_left_pic else (-1 if r.fix_right_pic else 0),
            axis=1).values

        y_fit, _ = chebyshev_fit_smooth(t_rel, gaze, order=order)

        left_new  = d.left_new.iloc[0]
        right_new = d.right_new.iloc[0]
        pos = "左" if left_new == 1 else "右" if right_new == 1 else "无"

        for ti, yf in zip(t_rel, y_fit):
            fit_rows.append({
                "trial": trial, "time_s": round(ti, 4),
                "gaze_fit": round(yf, 4), "new_pos": pos
            })

    print(f"✅ 切比雪夫拟合完成，生成 {len(fit_rows)} 条拟合数据")
    return pd.DataFrame(fit_rows)


# ================================================================
# ▼▼▼  核心修改：calc_final_params_from_fit → 返回逐试次行  ▼▼▼
# ================================================================

def calc_trial_params_from_fit(fit_df, subject):
    """
    【改版】从拟合DataFrame计算每个试次的认知指标，返回 list[dict]。
    每个字典 = 一行（被试 × 试次），包含四项指标：
      - 记忆表征能力    (end_prob，4-5 s 段 P_t 均值)
      - 认知加工噪声水平 (prob_std，P_t 时间序列标准差)
      - 记忆提取速率    (time_to_peak，首个显著峰值时间，s)
      - 平均整体偏好    (mean_prob，0-5 s 全段 P_t 均值，辅助指标)
    无效试次（数据点 < 5）跳过，并在日志中打印跳过原因。
    """
    print(f"\n🔧 计算被试【{subject}】逐试次指标...")

    # ── 1. P_t 转换（与原版相同）──────────────────────────────────
    fit_df = fit_df.copy()
    fit_df["new_pos"] = fit_df["new_pos"].astype(str).str.strip().str.lower()
    fit_df["P_t"] = np.where(
        fit_df["new_pos"].isin(["左", "left", "1"]),
        np.clip( fit_df["gaze_fit"], 0, 1),
        np.clip(-fit_df["gaze_fit"], 0, 1)
    )
    # 截取 0-5 s 测试窗口
    fit_df = fit_df[(fit_df["time_s"] >= 0) & (fit_df["time_s"] <= 5)].copy()

    # ── 2. 逐试次计算 ─────────────────────────────────────────────
    trial_rows = []
    skipped = 0

    for trial, group in fit_df.groupby("trial"):
        group = group.sort_values("time_s").reset_index(drop=True)

        # 有效性检查：至少需要 5 个数据点
        if len(group) < 5:
            print(f"   ⚠️  试次 {trial} 数据点不足（{len(group)} 个），跳过")
            skipped += 1
            continue

        # ① 平均整体偏好（0-5 s 全段均值）
        mean_prob = round(float(group["P_t"].mean()), 6)

        # ② 记忆表征能力：4-5 s 段的 P_t 均值
        end_group = group[(group["time_s"] >= 4) & (group["time_s"] <= 5)]
        if len(end_group) > 0:
            end_prob = round(float(end_group["P_t"].mean()), 6)
        else:
            # 4-5 s 无数据时，用末尾 10% 时间段替代，并记录 flag
            tail_cut = group["time_s"].max() * 0.9
            tail_group = group[group["time_s"] >= tail_cut]
            end_prob = round(float(tail_group["P_t"].mean()), 6)
            print(f"   ⚠️  试次 {trial} 无 4-5s 数据，用末段均值替代")

        # ③ 认知加工噪声水平：P_t 序列的标准差（ddof=1）
        prob_std = round(float(group["P_t"].std(ddof=1)), 6)

        # ④ 记忆提取速率：首个显著峰值（height > 0.1）出现的时间
        pt_values = group["P_t"].values
        peaks, _ = find_peaks(pt_values, height=0.1)
        if len(peaks) > 0:
            # 取最高峰的时间（与原版逻辑相同）
            highest_idx = peaks[np.argmax(pt_values[peaks])]
            time_to_peak = round(float(group["time_s"].iloc[highest_idx]), 6)
        else:
            # 无显著峰值时，使用时窗中点 2.5 s 作为默认值
            time_to_peak = 2.5

        # 新图位置（取该试次第一行）
        new_pos_raw = group["new_pos"].iloc[0]

        trial_rows.append({
            "被试":          subject,
            "试次":          int(trial),
            "新图位置":      new_pos_raw,
            "记忆表征能力":   end_prob,
            "认知加工噪声水平": prob_std,
            "记忆提取速率":   time_to_peak,
            "平均整体偏好":   mean_prob,
        })

    if not trial_rows:
        raise ValueError(f"被试【{subject}】无任何有效试次，请检查原始数据")

    # ── 3. 打印该被试汇总 ─────────────────────────────────────────
    n_valid = len(trial_rows)
    arr_end   = np.array([r["记忆表征能力"]    for r in trial_rows])
    arr_noise = np.array([r["认知加工噪声水平"] for r in trial_rows])
    arr_speed = np.array([r["记忆提取速率"]    for r in trial_rows])

    print(f"   ✅ 有效试次：{n_valid} 个（跳过 {skipped} 个）")
    print(f"   📊 记忆表征能力    均值={arr_end.mean():.4f}  SD={arr_end.std():.4f}")
    print(f"   📊 认知加工噪声水平 均值={arr_noise.mean():.4f}  SD={arr_noise.std():.4f}")
    print(f"   📊 记忆提取速率    均值={arr_speed.mean():.4f}  SD={arr_speed.std():.4f}")

    return trial_rows   # ← 返回 list，每元素对应一个试次（一行）


# ================================================================
def get_csv_files_recursively(folder_path):
    """
    递归获取指定文件夹及其所有子文件夹中的 CSV 文件。
    """
    return sorted(
        path
        for path in folder_path.rglob("*")
        if path.is_file() and path.suffix.lower() == ".csv"
    )


def main():
    print("=" * 70)
    print("【逐试次模式】递归批量处理 filtered_data 中所有被试")
    print(f"filtered_data 根目录：{FILTERED_DATA_ROOT}")
    print(f"输出文件：{OUTPUT_FILE_PATH}")
    print("输出格式：每行 = 1个被试 × 1个试次")
    print("=" * 70)

    # 1. 检查 filtered_data 根目录
    if not FILTERED_DATA_ROOT.exists():
        raise FileNotFoundError(
            f"filtered_data 文件夹不存在：{FILTERED_DATA_ROOT}"
        )

    if not FILTERED_DATA_ROOT.is_dir():
        raise NotADirectoryError(
            f"FILTERED_DATA_ROOT 不是文件夹：{FILTERED_DATA_ROOT}"
        )

    # 2. 检查 data1/data2/data3 及对应参数文件
    for dataset_name, param_path in DATASET_CONFIG.items():
        dataset_dir = FILTERED_DATA_ROOT / dataset_name

        if not dataset_dir.exists():
            raise FileNotFoundError(
                f"数据子文件夹不存在：{dataset_dir}"
            )

        if not dataset_dir.is_dir():
            raise NotADirectoryError(
                f"数据路径不是文件夹：{dataset_dir}"
            )

        if not param_path.exists():
            raise FileNotFoundError(
                f"{dataset_name} 对应的参数文件不存在：{param_path}"
            )

    all_rows = []
    success_count = 0
    fail_count = 0
    total_file_count = 0

    dataset_summary = {}

    # 3. 分别处理 data1 / data2 / data3
    for dataset_name, param_path in DATASET_CONFIG.items():
        dataset_dir = FILTERED_DATA_ROOT / dataset_name
        csv_files = get_csv_files_recursively(dataset_dir)

        dataset_summary[dataset_name] = {
            "total": len(csv_files),
            "success": 0,
            "fail": 0,
        }

        print("\n" + "=" * 70)
        print(f"开始处理数据组：{dataset_name}")
        print(f"数据目录：{dataset_dir}")
        print(f"参数文件：{param_path}")
        print(f"发现 CSV：{len(csv_files)} 个")
        print("=" * 70)

        if not csv_files:
            print(f"⚠️ {dataset_name} 中没有找到 CSV 文件，跳过。")
            continue

        total_file_count += len(csv_files)

        for eye_file in csv_files:
            # 沿用原程序：被试编号 = 文件名去掉扩展名
            subj_name = eye_file.stem
            relative_path = eye_file.relative_to(FILTERED_DATA_ROOT)

            print(
                f"\n{'=' * 20} "
                f"处理被试：{subj_name} "
                f"[{relative_path}] "
                f"{'=' * 20}"
            )

            try:
                print("[1/4] 加载数据并标记试次...")
                df_t1 = load_and_process_t1(str(eye_file))

                print("[2/4] 合并对应参数并判断注视位置...")
                df_t2 = merge_params_and_gaze(
                    df_t1,
                    str(param_path)
                )

                print("[3/4] 切比雪夫拟合...")
                df_fit = run_chebyshev_fit(
                    df_t2,
                    CHEBYSHEV_ORDER
                )

                print("[4/4] 计算逐试次认知指标...")
                trial_rows = calc_trial_params_from_fit(
                    df_fit,
                    subj_name
                )

                # 增加来源数据组，便于合并后追溯
                for row in trial_rows:
                    row["数据组"] = dataset_name

                all_rows.extend(trial_rows)

                success_count += 1
                dataset_summary[dataset_name]["success"] += 1

            except Exception as e:
                print(
                    f"❌ 被试【{subj_name}】处理失败：{e}，跳过"
                )
                fail_count += 1
                dataset_summary[dataset_name]["fail"] += 1
                continue

    # 4. 汇总三个数据组，保存为一个 CSV
    if not all_rows:
        print("\n❌ 无任何有效数据，未生成输出文件")
        return

    final_df = pd.DataFrame(
        all_rows,
        columns=[
            "被试",
            "试次",
            "新图位置",
            "记忆表征能力",
            "认知加工噪声水平",
            "记忆提取速率",
            "平均整体偏好",
            "数据组",
        ],
    )

    # 先按被试、试次排序；与原程序逻辑一致
    final_df = final_df.sort_values(
        ["被试", "试次"]
    ).reset_index(drop=True)

    OUTPUT_FILE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_df = final_df.drop(
        columns=["新图位置", "数据组"],
        errors="ignore"
    )

    output_df.to_excel(
        OUTPUT_FILE_PATH,
        index=False
    )

    # 5. 汇总打印
    print("\n" + "=" * 70)
    print("🎉 全部数据处理完成")
    print("=" * 70)

    for dataset_name in DATASET_CONFIG:
        info = dataset_summary[dataset_name]
        print(
            f"{dataset_name}: "
            f"总文件 {info['total']}，"
            f"成功 {info['success']}，"
            f"失败 {info['fail']}"
        )

    print("-" * 70)
    print(f"总 CSV 文件数：{total_file_count}")
    print(f"成功被试数：{success_count}")
    print(f"失败被试数：{fail_count}")
    print(f"总行数（被试×试次）：{len(final_df)}")
    print(f"涉及被试数：{final_df['被试'].nunique()}")

    if final_df["被试"].nunique() > 0:
        print(
            "每被试平均试次数："
            f"{len(final_df) / final_df['被试'].nunique():.1f}"
        )

    print(f"💾 输出文件：{OUTPUT_FILE_PATH}")
    print("=" * 70)

    print("\n📋 前 10 行预览：")
    print(final_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

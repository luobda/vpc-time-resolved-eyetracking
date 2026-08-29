import shutil
from pathlib import Path

import pandas as pd


# ============================================================
# 路径配置
# ============================================================
# 项目结构默认如下：
#
# project/
# ├─ code/
# │  └─ data_quality_control.py
# └─ data/
#    └─ raw_data/
#
# 因此 raw_data 相对于本脚本的路径为 ../data/raw_data
SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = (SCRIPT_DIR / ".." / "data" / "raw_data").resolve()

# 筛选后的数据保存在 raw_data 同级目录：
# data/filtered_data/
FILTERED_DATA_DIR = RAW_DATA_DIR.parent / "filtered_data"

# 判断阈值
THRESHOLD = 70.0


def read_csv_with_fallback(file_path):
    """
    读取 CSV。
    优先使用 UTF-8-SIG；如果失败则尝试 GBK。
    与原程序读取逻辑保持一致。
    """
    try:
        return pd.read_csv(file_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding="gbk")


def calculate_valid_percentage(file_path):
    """
    计算 bino_eye_valid == 1 的记录占全部记录的百分比。
    如果不存在 bino_eye_valid 列，则返回 None。
    """
    df = read_csv_with_fallback(file_path)

    if "bino_eye_valid" not in df.columns:
        print(
            f"警告：{file_path.name} 中不存在 "
            f"'bino_eye_valid' 列，已跳过。"
        )
        return None

    total_count = len(df)

    if total_count == 0:
        percentage = 0.0
    else:
        bino_eye_valid = pd.to_numeric(
            df["bino_eye_valid"],
            errors="coerce"
        )

        valid_count = (bino_eye_valid == 1).sum()
        percentage = valid_count / total_count * 100

    return percentage


def copy_raw_data():
    """
    将整个 raw_data 文件夹完整复制为 filtered_data。

    如果 filtered_data 已经存在，则停止运行，
    避免覆盖以前已经筛选好的数据。
    """
    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(
            f"找不到 raw_data 文件夹：\n{RAW_DATA_DIR}"
        )

    if not RAW_DATA_DIR.is_dir():
        raise NotADirectoryError(
            f"RAW_DATA_DIR 不是文件夹：\n{RAW_DATA_DIR}"
        )

    if FILTERED_DATA_DIR.exists():
        raise FileExistsError(
            "目标文件夹 filtered_data 已经存在。\n"
            "为了防止覆盖已有筛选结果，程序已停止。\n"
            f"目标路径：{FILTERED_DATA_DIR}\n"
            "请先确认并手动删除或重命名该文件夹，然后重新运行。"
        )

    print("=" * 70)
    print("第一步：复制原始数据")
    print("=" * 70)
    print(f"原始数据：{RAW_DATA_DIR}")
    print(f"复制到：  {FILTERED_DATA_DIR}")

    shutil.copytree(RAW_DATA_DIR, FILTERED_DATA_DIR)

    print("raw_data 已完整复制。\n")


def quality_control_and_delete():
    """
    递归遍历 filtered_data 下所有层级的 CSV 文件。

    判断规则与原程序保持一致：
    1. 读取 bino_eye_valid；
    2. 统计 bino_eye_valid == 1 的占比；
    3. 占比低于 THRESHOLD 时删除该 CSV；
    4. 缺少 bino_eye_valid 列时跳过，不删除。
    """
    csv_files = sorted(
        path
        for path in FILTERED_DATA_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() == ".csv"
    )

    if not csv_files:
        print("filtered_data 及其子文件夹中没有找到 CSV 文件。")
        return

    results = []
    deleted_files = []
    skipped_files = []
    error_files = []

    print("=" * 70)
    print("第二步：递归进行数据质量控制")
    print("=" * 70)
    print(f"共找到 {len(csv_files)} 个 CSV 文件。\n")

    for file_path in csv_files:
        relative_path = file_path.relative_to(FILTERED_DATA_DIR)

        try:
            percentage = calculate_valid_percentage(file_path)

            if percentage is None:
                skipped_files.append(str(relative_path))
                continue

            results.append((str(relative_path), percentage))

            if percentage < THRESHOLD:
                file_path.unlink()
                deleted_files.append((str(relative_path), percentage))
                print(
                    f"[删除] {relative_path}: "
                    f"{percentage:.2f}% < {THRESHOLD:.0f}%"
                )
            else:
                print(
                    f"[保留] {relative_path}: "
                    f"{percentage:.2f}%"
                )

        except Exception as e:
            error_files.append((str(relative_path), str(e)))
            print(
                f"[错误] 读取 {relative_path} 时发生错误：{e}"
            )

    print("\n" + "=" * 70)
    print("质量控制结果汇总")
    print("=" * 70)
    print(f"检查 CSV 数量：{len(csv_files)}")
    print(f"成功计算质量：{len(results)}")
    print(f"删除文件数量：{len(deleted_files)}")
    print(f"跳过文件数量：{len(skipped_files)}")
    print(f"读取错误数量：{len(error_files)}")

    print("\n" + "-" * 70)
    print(f"占比低于 {THRESHOLD:.0f}%、已删除的 CSV")
    print("-" * 70)

    if deleted_files:
        for relative_path, percentage in deleted_files:
            print(f"{relative_path}: {percentage:.2f}%")
    else:
        print(
            f"没有占比低于 {THRESHOLD:.0f}% 的 CSV 文件。"
        )

    if skipped_files:
        print("\n" + "-" * 70)
        print("因缺少 bino_eye_valid 列而跳过的 CSV")
        print("-" * 70)
        for relative_path in skipped_files:
            print(relative_path)

    if error_files:
        print("\n" + "-" * 70)
        print("读取时发生错误的 CSV")
        print("-" * 70)
        for relative_path, error_message in error_files:
            print(f"{relative_path}: {error_message}")

    print("\n" + "=" * 70)
    print("处理完成")
    print("=" * 70)
    print(f"原始数据保持不变：{RAW_DATA_DIR}")
    print(f"筛选后数据位于：  {FILTERED_DATA_DIR}")


def main():
    # 第一步：复制整个 raw_data
    copy_raw_data()

    # 第二步：只在副本中进行质量控制和删除
    quality_control_and_delete()


if __name__ == "__main__":
    main()

"""
将trial_difficulty.csv中的难度值匹配到年轻人逐试次指标.xlsx的完整代码
功能：根据试次编号（pair_id/试次）将difficulty值添加到Excel文件中
"""

import pandas as pd
import numpy as np
from pathlib import Path


def add_difficulty_to_trial_data(difficulty_file_path, trial_data_file_path, output_file_path):
    """
    将难度值添加到试次数据中

    参数:
    difficulty_file_path: str, trial_difficulty.csv文件路径
    trial_data_file_path: str, 年轻人逐试次指标.xlsx文件路径
    output_file_path: str, 输出文件路径
    """

    # 1. 读取难度数据文件
    print("正在读取难度数据文件...")
    difficulty_df = pd.read_csv(difficulty_file_path, encoding = 'gbk')

    # 检查必要的列是否存在
    required_columns_diff = ['pair_id', 'difficulty']
    for col in required_columns_diff:
        if col not in difficulty_df.columns:
            raise ValueError(f"难度数据文件缺少必要的列: {col}")

    # 创建难度字典，用于快速查找（pair_id -> difficulty）
    difficulty_dict = dict(zip(difficulty_df['pair_id'], difficulty_df['difficulty']))
    print(f"成功读取难度数据，共包含 {len(difficulty_dict)} 个试次的难度值")
    print(f"试次范围: {min(difficulty_dict.keys())} - {max(difficulty_dict.keys())}")

    # 2. 读取试次数据文件
    print("\n正在读取试次数据文件...")
    trial_df = pd.read_excel(trial_data_file_path)

    # 检查必要的列是否存在
    required_columns_trial = ['被试', '试次']
    for col in required_columns_trial:
        if col not in trial_df.columns:
            raise ValueError(f"试次数据文件缺少必要的列: {col}")

    print(f"成功读取试次数据，共包含 {len(trial_df)} 条记录")
    print(f"被试数量: {trial_df['被试'].nunique()}")
    print(f"试次范围: {trial_df['试次'].min()} - {trial_df['试次'].max()}")

    # 3. 匹配难度值
    print("\n正在匹配难度值...")
    # 根据试次列匹配难度值
    trial_df['difficulty'] = trial_df['试次'].map(difficulty_dict)

    # 检查是否有匹配失败的记录（试次不在难度字典中）
    missing_difficulty = trial_df['difficulty'].isnull().sum()
    if missing_difficulty > 0:
        print(f"警告: 有 {missing_difficulty} 条记录未找到对应的难度值")
        # 显示未匹配的试次编号
        missing_trials = trial_df[trial_df['difficulty'].isnull()]['试次'].unique()
        print(f"未匹配的试次编号: {sorted(missing_trials)}")
    else:
        print("所有记录都成功匹配到难度值")

    # 4. 统计每个被试的试次数量
    trial_count_per_subject = trial_df.groupby('被试')['试次'].count()
    subjects_with_less_30 = (trial_count_per_subject < 30).sum()
    print(f"\n试次数量不足30轮的被试数量: {subjects_with_less_30}")
    print(f"试次数量分布统计:")
    print(trial_count_per_subject.describe())

    # 5. 保存结果到新文件
    print(f"\n正在保存结果到: {output_file_path}")
    trial_df.to_excel(output_file_path, index=False)
    print("文件保存成功！")

    # 返回处理后的数据
    return trial_df


# ------------------- 主程序执行 -------------------
if __name__ == "__main__":
    # 配置文件路径
    # 以当前脚本所在的 code 文件夹为基准，自动定位同级的 data 文件夹
    SCRIPT_DIR = Path(__file__).resolve().parent
    DATA_DIR = (SCRIPT_DIR / ".." / "data").resolve()

    difficulty_FILE = DATA_DIR / "trial_difficulty.csv"
    TRIAL_DATA_FILE = DATA_DIR / "fitting_results.xlsx"
    OUTPUT_FILE = DATA_DIR / "fitting_results_difficulty.xlsx"

    try:
        # 执行难度值添加操作
        result_df = add_difficulty_to_trial_data(difficulty_FILE, TRIAL_DATA_FILE, OUTPUT_FILE)

        # 显示处理结果摘要
        print("\n" + "=" * 50)
        print("处理完成！结果摘要:")
        print(f"原始记录数: {len(result_df)}")
        print(f"包含被试数: {result_df['被试'].nunique()}")
        print(f"难度值范围: {result_df['difficulty'].min():.6f} - {result_df['difficulty'].max():.6f}")
        print(f"难度值平均值: {result_df['difficulty'].mean():.6f}")
        print("=" * 50)

    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")
        print("请检查文件路径和文件格式是否正确")

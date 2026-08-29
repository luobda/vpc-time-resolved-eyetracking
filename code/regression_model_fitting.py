import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
from pathlib import Path

# ======================== 路径配置 ========================
# 项目结构：
# project/
# ├─ code/
# │  └─ regression_model_fitting.py
# └─ data/
#    └─ trial_difficulty.csv
#
# 以当前脚本所在的 code 文件夹为基准，自动定位同级的 data 文件夹
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = (SCRIPT_DIR / ".." / "data").resolve()

score_FILE = DATA_DIR / "trial_difficulty.csv"
# ========================================================
df = pd.read_csv(score_FILE, encoding = 'gbk')

# ---------------------- 2. difficulty → z‑score（正态分位数变换） ----------------------
# 秩次
r = df['经验难度']
n = len(df)
# 累积概率
p = r
# 正态反函数 → z
df['z_difficulty'] = norm.ppf(p)

# ---------------------- 3. 生成交互项：T1 × score ----------------------
df['T1_score'] = df['T1'] * df['score']

# ---------------------- 4. 回归：z_difficulty ~ T1 + score + T1×score ----------------------
X = df[['T1', 'T2', 'score', 'T1_score']]
X = sm.add_constant(X)  # 加常数项
y = df['z_difficulty']

model = sm.OLS(y, X).fit()

# ---------------------- 5. 输出结果：系数、显著性 ----------------------
print("=" * 60)
print("回归结果：z_difficulty ~ T1 + score + T1*score")
print("=" * 60)
print(model.summary())

# 提取关键信息：系数、p值、显著性标记
print("\n=== 回归系数与显著性汇总 ===")
res = pd.DataFrame({
    '变量': model.params.index,
    '系数': model.params.values,
    '标准误': model.bse.values,
    't值': model.tvalues.values,
    'p值': model.pvalues.values
})
# 显著性标记
res['显著性'] = res['p值'].apply(lambda p:
                                 '***' if p < 0.001 else
                                 '**' if p < 0.01 else
                                 '*' if p < 0.05 else 'ns')
print(res)
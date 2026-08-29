# 代码说明（Code Description）

本项目代码用于完成 Visual Paired-Comparison（VPC）任务中的逐试次眼动指标验证、任务难度效应分析、个体差异分析、PCA、人群聚类以及问卷关联分析。

建议按照代码编号顺序运行。

---

# 1. `01_trial_level_validation.py`

用于验证逐试次任务难度指标及三个核心眼动指标的基本有效性。

主要分析指标包括：

* Memory Representation Ability
* Cognitive Processing Noise Level
* Memory Retrieval Latency

主要输入：

```text
data/fitting_results_difficulty.xlsx
```

主要输出：

```text
Chapter1_Output/
```

---

# 2. `02a_group_level_difficulty_effects.py`

用于分析任务难度对三个核心眼动指标的总体群体效应。

主要采用线性混合效应模型。

主要输入：

```text
data/fitting_results_difficulty.xlsx
```

主要输出：

```text
Chapter2_Output_1/
```

---

# 3. `02b_individual_difficulty_response.py`

用于分析不同参与者对任务难度增加的个体响应差异。

程序计算每名被试三个核心眼动指标随任务难度变化的个体 slope，并分析静态表现与动态难度响应之间的关系。

主要输入：

```text
data/fitting_results_difficulty.xlsx
```

主要输出：

```text
Chapter2_Output_2/
```

---

# 4. `03_six_feature_pca_analysis.py`

用于构建被试层面的六维眼动特征，并进行主成分分析（PCA）。

六个特征包括：

* Memory Representation Ability
* Cognitive Processing Noise Level
* Memory Retrieval Latency
* Memory Representation Ability slope
* Cognitive Processing Noise slope
* Memory Retrieval Latency slope

程序同时进行相关分析、KMO、Bartlett 检验，并输出 PCA loading 和 component scores。

主要输入：

```text
data/fitting_results_difficulty.xlsx
```

主要输出：

```text
Chapter3_Output/
Chapter3_Output.zip
```

其中后续分析主要使用：

```text
Table_S3_3_subject_six_features_standardized.csv
Table_S3_3_PCA_component_scores.csv
```

---

# 5. `04_subtype_clustering_analysis.py`

用于根据眼动 PCA 表征进行参与者亚型识别。

程序首先筛选低表征参与者，再对其余参与者基于 PC1–PC5 进行 K-means 聚类。

主要输入：

```text
Chapter3_Output.zip
```

主要输出：

```text
Chapter4_Output/
Chapter4_Output.zip
```

主要结果包括：

```text
Table_S4_subject_classification_and_features.csv
Table_S4_subject_group_membership_probabilities.csv
```

---

# 6. `05a_questionnaire_subtype_comparison.py`

用于比较不同 VPC 人群亚型在六个问卷总分上的差异。

主要采用：

* Welch ANOVA
* Pairwise Welch t-test
* Effect size
* FDR correction

主要输出：

```text
Chapter5_Output/
```

---

# 7. `05b_questionnaire_subtype_probability_correlations.py`

用于分析不同亚型 membership probability 与六个问卷总分之间的 Spearman 相关。

主要输出：

```text
Chapter5_Output1/
```

---

# 8. `05c_questionnaire_pca_correlations.py`

用于分析眼动 PC1–PC5 与六个问卷总分之间的 Spearman 相关。

主要输入：

```text
Chapter3_Output/
└── Table_S3_3_PCA_component_scores.csv
```

主要输出：

```text
Chapter5_Output2/
```

---

# 9. `05d_questionnaire_eye_feature_correlations.py`

用于分析六个标准化眼动特征与六个问卷总分之间的 Spearman 相关。

主要输入：

```text
Chapter3_Output/
└── Table_S3_3_subject_six_features_standardized.csv
```

主要输出：

```text
Chapter5_Output3/
```

---

# 10. 推荐执行顺序

```bash
python 01_trial_level_validation.py
python 02a_group_level_difficulty_effects.py
python 02b_individual_difficulty_response.py
python 03_six_feature_pca_analysis.py
python 04_subtype_clustering_analysis.py
python 05a_questionnaire_subtype_comparison.py
python 05b_questionnaire_subtype_probability_correlations.py
python 05c_questionnaire_pca_correlations.py
python 05d_questionnaire_eye_feature_correlations.py
```

其中 `05a–05d` 为后续并行分析，没有严格的相互依赖顺序。



# 数据说明（Data Description）

=======
# README

# 数据说明（Data Description）


## 1. 数据目录结构

```
data/
├── raw_data/
│   ├── data1/
│   ├── data2/
│   └── data3/
│
├── task_parameters_data1.csv
├── task_parameters_data2.csv
├── task_parameters_data3.csv
├── trial_difficulty.csv
└── fitting_results.xlsx
```

其中，`data1`、`data2` 和 `data3` 表示不同批次或来源的数据集合。三个数据集合使用相同的分析流程处理，并在后续分析中合并。

---

## 2. `raw_data/`

`raw_data/` 保存质量控制之前的原始眼动记录。

```
raw_data/
├── data1/
├── data2/
└── data3/
```

每个 CSV 文件对应一名匿名化被试。

文件名使用匿名化被试编号，例如：

```
1.csv
2.csv
3.csv
...
```

原始眼动数据中包含用于后续分析的时间戳、双眼注视位置、眼动有效性标记以及实验 trigger 等信息。

分析程序主要使用以下字段：

- `timestamp`：眼动采样时间戳；
- `bino_eye_gaze_position_x`：双眼合并后的水平注视位置；
- `bino_eye_gaze_position_y`：双眼合并后的垂直注视位置；
- `bino_eye_valid`：双眼数据有效性标记；
- `trigger`：实验程序发送的事件标记。

原始数据仅用于数据质量控制及后续眼动指标计算，不应直接修改。

---

## 3. 任务参数文件

三个任务参数文件分别对应三个数据集合：

```
task_parameters_data1.csv
task_parameters_data2.csv
task_parameters_data3.csv
```

这些文件记录各试次的实验刺激配置，包括新图片及左右图片位置等信息。

眼动分析程序根据任务参数判断每个试次中新刺激位于屏幕左侧还是右侧，并据此将注视轨迹转换为对新刺激的视觉偏好指标。

三个参数文件分别对应：

```
filtered_data/data1/
filtered_data/data2/
filtered_data/data3/
```

---

## 4. `fitting_results.xlsx`

`fitting_results.xlsx` 是由质量控制后的眼动数据经过逐试次时间序列拟合后得到的主要分析数据。

每一行表示：

```
1 名被试 × 1 个试次
```

目前保存的主要字段包括：

| 字段 | 含义 |
| --- | --- |
| `被试` | 匿名化被试编号 |
| `试次` | VPC 任务试次编号 |
| `记忆表征能力` | 测试阶段后期对新刺激的视觉偏好，用于表示视觉识别记忆表征水平 |
| `认知加工噪声水平` | 试次内新刺激偏好时间序列的波动程度 |
| `记忆提取速率` | 新刺激偏好显著峰值出现的时间指标 |

其中，当前程序中的“记忆提取速率”实际上采用峰值出现时间进行操作化，因此数值越大表示显著偏好峰值出现得越晚，即记忆提取潜伏期越长。

该文件是后续任务难度分析、个体特征构建、PCA 和聚类分析的重要输入文件。

---

## 5. `trial_difficulty.csv`

`trial_difficulty.csv` 保存每个试次对应的任务难度指标。

该文件用于将任务难度信息添加到逐试次眼动指标中，从而研究随着任务难度增加，视觉记忆加工指标如何发生变化。

后续分析主要关注以下三类指标与任务难度之间的关系：

- 记忆表征能力；
- 认知加工噪声水平；
- 记忆提取潜伏期 / 记忆提取速率指标。

---

## 6. 问卷数据

项目后续分析还使用六个问卷或量表总分，包括：

- 注意控制量表；
- 心力自评量表；
- 抑郁情绪自评量表；
- 焦虑情绪自评量表；
- 失眠严重程度指数；
- 简易精神状态评价量表（MMSE）。

量表分析仅纳入完成全部六个量表且能够与眼动数据成功匹配的被试，以保证不同量表分析具有统一的样本基础。

量表均使用原始总分，不进行 PCA 或 z 标准化。

其中：

- 注意控制、心力和 MMSE：总分越高通常表示状态越好；
- 抑郁、焦虑和失眠：总分越高通常表示症状越严重。
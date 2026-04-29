# gaze-toolkit · 多模态认知状态感知平台

面向移动 / 穿戴产品的 AI 驱动人因分析平台原型。以眼动数据为核心，融合生理多模态信号，输出可解释的认知状态推断与 UX 改进建议。

## 分析链

```text
产品界面截图
    ↓ [基线层]  物理显著性（OpenCV）+ DeepGaze → 注意力基线地图
    ↓ [采集层]  眼动信号（CSV / EyeLink）+ 心率等生理数据
    ↓ [特征层]  注视 / 扫视 / 瞳孔 / 眨眼特征（40+ 维）
    ↓ [AI 层]   认知负荷估计 + 意图预测 + LLM 特征异常解读
    ↓ [洞察层]  认知状态假说 + UX 改进建议
```

## 功能模块

| 分析链层 | 模块 | 说明 |
| --- | --- | --- |
| 基线层 | 物理显著性 + DeepGaze | 产品截图 → 注意力基线地图，无需眼动数据 |
| 采集层 | 眼动 + 心率多模态 | CSV / EyeLink 接入，采样率 30–1000 Hz |
| 特征层 | 注视 / 扫视 / 瞳孔 / 眨眼 | 40+ 维特征，AOI 兴趣区联动分析 |
| AI 层 | 认知负荷 + 意图预测 | RF / GBDT / SVM + LLM 特征异常解读 |
| 洞察层 | 报告生成 + UX 建议 | 认知状态假说 + 可操作界面改进建议 |
| 批量层 | 多文件批量分析 | 跨会话对比 + 统计检验 + 效应量估计 |

## 快速开始

```bash
pip install -e .
pip install -e .[dashboard]
```

启用 DeepGaze 认知显著性后端（可选）：

```powershell
scripts\setup-deepgaze-runtime.cmd
```

命令行演示：

```bash
gaze-toolkit simulate --output demo.csv
gaze-toolkit features --input demo.csv
gaze-toolkit train-demo --sessions 24
```

启动 Dashboard：

```bash
gaze-toolkit-ui
# 或
streamlit run src/gaze_toolkit/dashboard.py
```

## 典型演示路径

### 1. 三层注意力分析

上传产品界面截图，在 Dashboard 侧边栏勾选任意层组合叠加显示：

- **物理显著性**：用户应该看哪里（底层视觉吸引力）
- **DeepGaze 预测**：人类注视先验预测（认知期望）
- **真实眼动热力图**：用户实际看了哪里（真实观测）

三层偏差 = 界面设计优化的直接证据。

### 2. LLM 特征异常解读

在「单次会话分析」Tab 底部，自动检测眼动特征异常（warning / critical 双级），
可选接入 LLM 生成认知状态假说和 UX 建议。

### 3. 认知负荷分类

```python
from gaze_toolkit.cognitive_load import simulate_cognitive_load_dataset, run_cognitive_load_experiment

df = simulate_cognitive_load_dataset(num_sessions=80)
report = run_cognitive_load_experiment(df, target="style", task="classification")
print(report.result.metrics)
```

参见 [notebooks/07-真实数据集认知负荷预测.ipynb](notebooks/07-真实数据集认知负荷预测.ipynb)：

- EyeLink 真实数据管线验证：LogisticRegression 70% 准确率
- 80 被试双条件 LOO 交叉验证：RandomForest 100% 准确率
- 扩展到 GazeBase（322 名被试）的完整代码模板

### 4. 多模态扩展

```python
from gaze_toolkit import compare_modalities
comparison = compare_modalities(num_sessions=32)
```

### 5. 研究方法验证

在「意图建模实验台」Tab 使用 pymovements ToyDataset 对照：
原生阈值法 vs pymovements I-VT vs I-DT，自动生成 Markdown 方法论摘要。

## 核心 API

```python
from gaze_toolkit import analyze_recording
from gaze_toolkit.datasets import simulate_gaze_recording
from gaze_toolkit.report_generator import explain_feature_anomalies, generate_insight_report

recording = simulate_gaze_recording(style="careful", seed=42)
analysis = analyze_recording(recording)

# LLM 特征异常解读
explanation = explain_feature_anomalies(analysis.features, use_llm=False)
print(explanation.cognitive_state_hypothesis)

# 生成洞察报告（含异常解读 section）
report = generate_insight_report(analysis.features, explain_anomalies=True)
```

## 教程 Notebook

| # | 主题 |
| --- | --- |
| 01 | 数据加载与预处理 |
| 02 | 特征提取与探索 |
| 03 | 构建意图分类器（传统方法） |
| 04 | 深度学习时序建模 |
| 05 | 多模态融合示例（眼动+心率） |
| 06 | 模型解释与可视化 |
| 07 | **真实数据集认知负荷预测**（new） |

## 项目结构

```text
src/gaze_toolkit/      核心模块（pipeline / features / cognitive_load / report_generator / dashboard）
notebooks/             7 个教程 notebook
tests/                 128 个单元测试（pytest）
examples/              演示脚本
scripts/               环境搭建脚本（DeepGaze runtime）
```

## 运行测试

```bash
pytest
```

## 注意事项

- 心率信号为模拟数据，不作为真实实验结论引用。
- 认知负荷分类结果为方法展示，不替代正式研究结论。
- 真实厂商设备导出数据字段名可能不同，需传入 `column_map` 适配。

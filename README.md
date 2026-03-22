# Human Factors AI Lab

`gaze-toolkit` 是一个面向作品集展示的 Python 工具包与研究演示台，目标受众是华为的人因研究专家和技术面试官。它不是单纯的眼动可视化脚本，而是一个从原始眼动数据到特征自动提取、状态/意图建模、多模态融合、可视化操作界面的最小完整闭环。

## 这个项目要证明什么

这个项目重点证明你具备下面这条能力链：

- 能处理多模态用户数据，至少覆盖眼动，并为生理信号留出统一接口。
- 能把原始时序数据转换成可用于研究和建模的结构化特征。
- 能构建基础评测模型，而不是停留在数据清洗和画图层面。
- 能把状态/意图预测做成可复用工具，而不是一次性分析脚本。
- 能把研究型逻辑封装成别人可操作的可视化界面。

## 当前 MVP 范围

输入：

- `CSV` 文件
- 基础 `EyeLink .asc`
- 内置模拟眼动数据
- 内置模拟心率信号，用于默认多模态演示

处理流程：

1. 数据标准化加载
2. 缺失值插值
3. 平滑滤波
4. 坐标归一化
5. I-VT 事件检测
6. 特征提取
7. 基线意图分类
8. 多模态融合对比
9. 可视化展示与交互操作

输出：

- 扫描路径图
- 热力图
- 信号时序总览
- 事件表
- 特征摘要
- 分类指标
- 混淆矩阵
- 特征重要性
- 眼动 vs 眼动+心率 基线对比

## 为什么这版设计适合作品集

它故意没有把所有研究级复杂性都一次性塞进去，而是优先保证一条完整、可信、能运行的闭环。对于面试展示，这比堆很多未闭环的“计划能力”更有效。

默认保留的假设：

- 演示主要在本地运行。
- 面试演示以 `CSV` 和模拟数据为主。
- 第二模态默认用心率做轻量演示，后续可替换为 EDA、EEG、鼠标轨迹、键盘行为等。

当前未默认打包但已保留扩展点：

- `EDF` 原生解析
- SHAP / LIME 完整解释链路
- LSTM / TCN / Transformer 深度时序模型
- 在线流式真实接入

## 项目结构

```text
configs/
docs/plans/
examples/
notebooks/
src/gaze_toolkit/
tests/
```

## 快速开始

安装：

```bash
pip install -e .
pip install -e .[dashboard]
```

如果你要启用基于 PyTorch + PySaliency + DeepGaze 的认知显著性后端，推荐直接运行仓库内的一键脚本：

```powershell
scripts\setup-deepgaze-runtime.cmd
```

如果你希望安装后顺手把完整 DeepGaze 推理链路也验掉，可以加上：

```powershell
scripts\setup-deepgaze-runtime.cmd -RunFullValidation
```

命令行演示：

```bash
gaze-toolkit simulate --output demo.csv
gaze-toolkit features --input demo.csv
gaze-toolkit train-demo --sessions 24
```

启动可视化研究界面：

```bash
gaze-toolkit-ui
```

如果你更习惯直接用 Streamlit：

```bash
streamlit run src/gaze_toolkit/dashboard.py
```

## 典型作品集演示路径

### 1. 单条眼动记录分析

- 上传 `CSV`
- 调整平滑窗口、扫视阈值、最小注视时长
- 查看扫描路径、热力图、信号曲线、事件表和关键特征

### 2. 阅读意图基线建模

- 生成模拟的 `careful vs skim` 数据集
- 训练随机森林 / SVM / 梯度提升模型
- 展示准确率、F1、混淆矩阵、特征重要性

### 3. 多模态扩展能力展示

- 在同一界面展示心率信号
- 比较眼动单模态与眼动+心率多模态结果
- 说明这条架构如何扩展到真实人因研究中的更多传感器

## 核心 API

```python
from gaze_toolkit import analyze_recording, compare_modalities, run_intent_experiment
from gaze_toolkit.datasets import simulate_gaze_recording

recording = simulate_gaze_recording(style="careful", seed=42)
analysis = analyze_recording(recording)

report = run_intent_experiment(num_sessions=32, model_name="random_forest")
comparison = compare_modalities(num_sessions=32)
```

## 示例与教程

示例脚本：

- [examples/demo_pipeline.py](examples/demo_pipeline.py)
- [examples/demo_multimodal.py](examples/demo_multimodal.py)

教程 notebook：

- [notebooks/01-数据加载与预处理.ipynb](notebooks/01-数据加载与预处理.ipynb)
- [notebooks/02-特征提取与探索.ipynb](notebooks/02-特征提取与探索.ipynb)
- [notebooks/03-构建意图分类器（传统方法）.ipynb](notebooks/03-构建意图分类器（传统方法）.ipynb)
- [notebooks/04-使用深度学习进行时序建模.ipynb](notebooks/04-使用深度学习进行时序建模.ipynb)
- [notebooks/05-多模态融合示例（眼动+心率）.ipynb](notebooks/05-多模态融合示例（眼动+心率）.ipynb)
- [notebooks/06-模型解释与可视化.ipynb](notebooks/06-模型解释与可视化.ipynb)

## 验证建议

```bash
pytest
```

## 未验证前提

- 真实厂商导出数据的字段名可能与默认列映射不同，此时需要传入 `column_map` 或补一个定制 loader。
- 本仓库默认多模态演示中的心率信号是模拟数据，不应被表述为真实实验结果。
- 当前意图分类任务是用于展示建模与工程能力的可运行基线，不应直接替代正式研究结论。

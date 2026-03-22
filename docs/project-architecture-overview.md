# 眼动项目架构、设计与能力说明

## 1. 项目定位

这个项目不是单一的“眼动可视化脚本”，而是一个围绕眼动数据构建的最小完整研究闭环，目标是把以下几类能力串起来：

1. 数据接入与标准化
2. 预处理与事件识别
3. 特征工程与基线建模
4. 多模态扩展与在线预测
5. 可视化展示与交互分析
6. 基于刺激图内容的注意分布建模

从实际代码状态看，它已经同时具备：

- 面向研究流程的核心库能力
- 可重复运行的 CLI 能力
- 可演示的 Streamlit Dashboard 能力
- 面向图像刺激的快速显著性与认知显著性能力

---

## 2. 总体架构

当前项目可以理解为一个“核心分析内核 + 多个交付界面 + 一个独立 DeepGaze 运行时”的结构。

### 2.1 模块分层

```text
输入层
  ├─ 文件加载 / DataFrame 转换 / 模拟数据生成
  └─ 配置加载

数据处理层
  ├─ 缺失值处理
  ├─ 平滑与归一化
  ├─ 事件识别
  └─ 分段切片

表征层
  ├─ 单记录特征提取
  ├─ 多记录特征表构建
  ├─ 多模态对齐与融合
  └─ 图像注意分布建模

建模层
  ├─ 分类/回归基线模型
  ├─ 特征重要性与 SHAP
  └─ 在线滑窗预测

交付层
  ├─ CLI
  ├─ Streamlit Dashboard
  ├─ Notebook / 示例脚本
  └─ 图表与热力图可视化

独立运行时层
  └─ PyTorch + PySaliency + DeepGaze 子进程运行时
```

### 2.2 核心设计原则

- 核心分析逻辑与 UI 解耦。`src/gaze_toolkit/` 中的大部分模块都可以脱离 Dashboard 单独调用。
- 数据以统一领域对象 `GazeRecording` / `EyeEvent` 为核心，而不是把 CSV/DataFrame 直接在各模块之间裸传。
- 预处理、事件识别、特征提取、分段、建模是可组合管线，而不是写死在单一入口里。
- 可扩展点通过 registry 暴露，包括自定义 loader、自定义 feature、自定义 model。
- 可选依赖分层管理，例如 Savitzky-Golay、SHAP、DeepGaze 都不是强制绑定在基础安装里的。
- 对复杂的 DeepGaze 认知模型使用独立 Python 运行时，避免把主环境和 PyTorch/Windows DLL 风险耦死。

---

## 3. 代码层面的架构分工

### 3.1 领域模型层

核心对象位于 `types.py`：

- `GazeRecording`
  - 表示一条标准化后的眼动记录
  - 内含 `samples`、`sampling_rate_hz`、`metadata`、`events`
- `EyeEvent`
  - 表示 fixation / saccade / blink / smooth_pursuit 等事件
  - 包含时间边界、幅度、峰值速度、质心等信息

这一层的意义是把“眼动研究对象”从具体文件格式里抽象出来，为后续处理建立统一输入。

### 3.2 接入层

接入相关逻辑主要位于：

- `io.py`
- `config.py`
- `datasets.py`

当前支持的输入方式：

- CSV / TSV / TXT 表格文件
- 基础 EyeLink `.asc`
- Pandas DataFrame 直接构造
- 内置模拟眼动记录
- 内置模拟意图分类数据
- 内置模拟心率信号

接入层做的不是简单读取，而是：

- 列名归一化
- 时间戳补齐
- 多种别名映射
- 非法格式拦截

这让后续处理模块几乎不需要关心“原始文件长什么样”。

### 3.3 数据处理层

处理逻辑分布在：

- `preprocess.py`
- `events.py`
- `segmentation.py`

职责分别是：

- `preprocess.py`
  - 缺失值处理
  - 平滑
  - 坐标归一化
- `events.py`
  - 速度计算
  - 基于原始标签或阈值规则的事件检测
- `segmentation.py`
  - 整条记录
  - 按时间窗切段
  - 按 marker 窗口切段
  - 按 start/end marker 成对切段

这层是整个项目的“研究前处理骨架”。

### 3.4 表征与建模层

表征与建模相关模块主要是：

- `features.py`
- `pipeline.py`
- `modeling.py`
- `multimodal.py`
- `streaming.py`
- `saliency.py`
- `src/deepgaze_worker.py`

它们分别承担：

- 从单条记录中提取统计/行为特征
- 把多条记录转换成 feature dataset
- 训练和评估基线模型
- 做多模态对齐与融合
- 做在线滑动窗口预测
- 做图像级注意分布建模

### 3.5 交付层

用户实际直接使用的入口主要有：

- `cli.py`
- `dashboard.py`
- `dashboard_launcher.py`
- `visualization.py`

其中：

- CLI 面向命令式批处理
- Dashboard 面向交互式研究演示
- visualization 封装了 scanpath、heatmap、signal overview、feature importance、confusion matrix 等图形输出

---

## 4. 现在已经能够实现的功能

下面列的是当前代码中已经落地并可运行的能力，而不是规划中的方向。

### 4.1 数据加载与标准化

已经支持：

- 从 CSV / TSV / TXT 加载眼动数据
- 从基础 EyeLink ASC 文本解析样本
- 从内存 DataFrame 直接构造
- 缺少时间戳时通过 `sampling_rate_hz` 合成时间轴
- 自动识别常见列别名，如 `x/y`、`gaze_x/gaze_y`、`event_label`、`marker`

### 4.2 单记录分析

已经支持：

- 缺失样本插值或剔除
- 移动平均平滑
- 可选 Savitzky-Golay 平滑
- 坐标归一化
- fixation / saccade / blink 检测
- 事件表输出
- 速度剖面计算
- 质量摘要输出
- 单记录特征摘要输出

### 4.3 分段分析

已经支持：

- 整体分析
- 时间范围切段
- marker 前后窗口切段
- 起止 marker 成对切段
- 每个 segment 独立进行 preprocess -> event -> feature 的分析链路

这使得项目不只适合“整段会话分析”，也适合刺激呈现、试次、阶段性任务片段分析。

### 4.4 特征工程

当前已经实现的特征类型包括：

- 基础时长和样本数特征
- 有效率/无效率
- 轨迹长度
- 速度均值/峰值
- fixation 数量、均值时长、总时长、密度
- saccade 数量、幅度均值、峰值速度均值、潜伏期
- blink 数量、频率、均值时长
- `x / y / pupil` 的均值、标准差、分位数、偏度、峰度
- 滚动均值/滚动标准差统计
- 近似熵 `Approximate Entropy`
- pupil baseline 和变化率

同时支持通过 `register_feature()` 注册自定义特征。

### 4.5 基线建模

当前已经支持：

- 分类任务
- 回归任务
- 数据集切分与 holdout 评估
- 输出 accuracy / F1 / ROC-AUC / confusion trace / MAE / RMSE / R2
- 输出 holdout prediction 明细

内置模型包括：

- Random Forest
- Gradient Boosting
- SVM
- Logistic Regression

可选模型包括：

- XGBoost
- LightGBM

解释能力包括：

- permutation feature importance
- SHAP（可选依赖）

### 4.6 多模态能力

当前多模态能力不是“空接口”，而是已经有一套可跑通的基础结构：

- `MultiModalData` 支持按时间戳对齐多个模态
- 支持 nearest-neighbor 的 `merge_asof` 对齐
- 支持 early fusion 的对齐后补齐
- 支持 feature-level concat
- 支持 late fusion 的加权平均

默认示范模态是：

- 眼动
- 模拟心率

当前心率信号是模拟的，适合作为架构演示，不应表述为真实生理实验结果。

### 4.7 在线预测

`SlidingWindowPredictor` 已经实现：

- 滑动时间窗缓存
- 按 step 间隔触发预测
- 每次在线提取特征
- 调用已训练模型输出分类结果
- 当模型支持 `predict_proba` 时输出各类别分数

这代表项目已经具备向实时人机交互场景扩展的基础骨架。

### 4.8 可视化与交互

当前已经支持：

- 静态 scanpath
- 静态 heatmap
- 交互式 scanpath
- 交互式 heatmap
- 信号总览
- 特征相关图
- 模型指标图
- 特征重要性图
- 混淆矩阵图
- 图像显著性热力图

Dashboard 已经组织成几个完整能力区块：

- 单次会话分析链路
- 意图建模实验台
- 多模态融合演示
- 项目解读

### 4.9 图像注意分布建模

这是最近新增并已经打通的能力，当前支持两类后端：

#### A. OpenCV 快速显著性

特点：

- 不依赖真实眼动
- 直接对上传图片做底层视觉显著性估计
- 适合做刺激图先验注意分布

#### B. PyTorch + PySaliency + DeepGaze 认知显著性

特点：

- 独立运行时执行
- 无 fixation history 时使用 `DeepGazeIIE`
- 有 fixation history 时使用 `DeepGazeIII`
- 输出 saliency map 的同时计算 `NSS / SIM / KL divergence`

这部分已经接入 Dashboard 和统一 API，不再是占位接口。

---

## 5. 背后的核心算法

### 5.1 预处理算法

当前预处理主要是经典规则与统计方法：

- 缺失值插值：`linear` 插值为主
- 缺失处理策略：`interpolate / drop / keep`
- 平滑：
  - Moving Average
  - Savitzky-Golay（可选依赖）
- 坐标归一化：
  - 0-1 归一化
  - 或进一步映射到 -1 到 1

### 5.2 事件识别算法

当前核心是 I-VT 思路：

- 先计算点对点 gaze velocity
- 再以速度阈值把样本粗分为 fixation / saccade
- `valid=False` 的片段归为 blink
- 通过连续相同标签聚合成事件段
- 再施加最小时长过滤

此外，项目也支持优先使用原始设备导出的 event label，而不是强制重算。

### 5.3 特征工程算法

特征工程主要基于：

- 事件统计
- 描述统计
- 滑动窗口统计
- 序列复杂度指标

其中复杂度特征使用了：

- Approximate Entropy

这让项目不仅能表达“平均水平”，还能表达眼动轨迹的动态复杂性。

### 5.4 建模算法

当前建模策略以“稳妥、可解释、易演示”的传统机器学习基线为主：

- Random Forest
- Gradient Boosting
- SVM
- Logistic Regression

项目并没有默认直接走深度学习时序模型，而是先把最小闭环做扎实，这符合 MVP 和作品集展示的工程策略。

### 5.5 多模态融合算法

当前融合方法主要有三种：

1. 时间对齐
   - `merge_asof`
   - nearest neighbor + tolerance
2. 特征级拼接
   - 各模态特征向量加前缀后拼接
3. 决策级融合
   - 对不同模态的概率输出做加权平均

这是一个典型的、可逐步演进的多模态工程骨架。

### 5.6 OpenCV 快速显著性算法

快速显著性后端并不是直接调用黑盒 API，而是组合了几个经典底层视觉信号：

- LAB 颜色对比
- 局部反差
- Sobel 边缘强度
- 轻量中心偏置
- 高斯平滑与稳健归一化

因此它本质上是一个 bottom-up saliency 近似器，适合快速预估“图像哪里更容易吸引初级视觉注意”。

### 5.7 DeepGaze 认知显著性算法

认知后端的核心流程是：

1. 读取图片
2. 加载中心偏置模板
3. 构造 image tensor 和 centerbias tensor
4. 根据是否有 fixation history 选择：
   - `DeepGazeIIE`
   - `DeepGazeIII`
5. 得到 log density
6. 转为归一化 saliency density
7. 使用 PySaliency 指标和真实 fixation 对比

当前已经计算的认知评价指标包括：

- NSS
- SIM
- image-based KL divergence

也就是说，这部分不只是“出一张热力图”，而是已经带有一定研究评价链路。

---

## 6. 开发思路与分层逻辑

这个项目的开发思路不是“先搭 UI，再把算法往里塞”，而是相反：

### 第 1 层：统一数据对象

先定义 `GazeRecording` / `EyeEvent`，让数据、事件、元信息有统一容器。

### 第 2 层：最小研究流水线

先打通：

`load -> preprocess -> detect events -> extract features`

这一层保证“输入一条记录，能产出研究可用摘要”。

### 第 3 层：数据集与模型

在最小流水线之上再扩展：

- 多条 recording 转 feature table
- 训练 baseline model
- 输出评估指标

这一层把项目从“分析工具”推进到“建模工具”。

### 第 4 层：多模态与在线能力

当单模态链路稳定后，再加：

- 模态对齐
- 融合方式
- 滑窗在线预测

这一层体现架构可扩展性。

### 第 5 层：交互界面与演示

最后再把能力封装进：

- CLI
- Dashboard
- Notebook / 示例脚本

这样 UI 只是能力的展示壳，不是业务逻辑的唯一承载点。

### 第 6 层：高级注意建模

在原有眼动分析之外，新增：

- 刺激图先验注意分布
- DeepGaze 认知显著性

这一层把项目从“看已有眼动”扩展为“在没有真实眼动时也能对刺激图做注意建模”。

---

## 7. 当前项目的层级总结

可以把整个项目概括成 6 个工程层级：

1. 领域对象层
   - `GazeRecording` / `EyeEvent`
2. 数据操作层
   - IO / preprocess / events / segmentation
3. 表征层
   - features / saliency / multimodal alignment
4. 模型层
   - modeling / streaming
5. 编排层
   - analysis / pipeline
6. 交付层
   - CLI / Dashboard / visualization

这套层级说明项目已经不仅是“功能堆叠”，而是有比较清晰的可维护结构。

---

## 8. 当前边界与未默认覆盖部分

为了避免误判项目成熟度，下面这些点需要明确：

- EDF 原生解析没有内置实现
- 默认多模态中的心率信号仍是模拟信号
- SHAP 依赖是可选的，不是默认安装
- DeepGaze 认知后端依赖独立 Python 运行时
- DeepGaze 运行时在 Windows 上依赖正确版本的 VC++ runtime
- 当前深度时序模型（LSTM / TCN / Transformer）仍处于预留扩展方向，而不是主链默认能力

---

## 9. 一句话总结

这不是一个“只会画眼动热力图”的项目，而是一个围绕眼动研究构建的分层式工程骨架：它已经打通了数据接入、预处理、事件识别、特征工程、基线建模、多模态扩展、在线预测、可视化交付，以及刺激图注意分布建模这几条真实可运行的能力链路。

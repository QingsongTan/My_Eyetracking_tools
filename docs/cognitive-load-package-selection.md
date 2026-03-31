# 眼动认知负荷识别技术选型表

## 1. 文档目标

这份文档不是泛泛地罗列眼动相关工具，而是专门回答一个问题：

**在当前这个项目的架构上，哪些外部包最值得接入，用来做“基于眼动数据识别认知负荷”，应该接在哪一层，优先级如何排。**

当前项目已经具备这些基础能力：

- 数据接入与标准化：`io.py`
- 预处理与事件识别：`preprocess.py`、`events.py`
- 特征工程：`features.py`
- 建模：`modeling.py`
- 多模态对齐：`multimodal.py`
- Dashboard 展示：`dashboard.py`

因此，这里的选型标准不是“功能最多”，而是：

1. 能否和现有 `GazeRecording -> features -> modeling -> dashboard` 链路自然接上
2. 能否直接提升“认知负荷识别”而不是只增加普通可视化
3. 能否增强你的作品集对“AI + 多模态人因 + 工程落地”的证明力

---

## 2. 总体结论

如果目标是尽快把当前项目升级成一个更像“认知负荷识别平台”的作品集，最推荐的组合是：

- `pymovements`
- `pypillometry`
- `scikit-learn / XGBoost`（项目中已具备基线建模入口）

这三者的组合最适合当前项目，因为它们分别补足了：

- 更稳的眼动数据处理底座
- 更专业的瞳孔负荷信号处理
- 可解释、可展示、适合面试的认知负荷分类/回归基线

如果要继续往“AI 驱动人因”方向加深，下一步再考虑：

- `uneye`
- `EyeBench`

如果手头没有原始眼部视频，而只有导出的 gaze / pupil 数值数据，则暂时**不要优先投入**：

- `PyPupilEXT`
- `Open-PupilEXT`

---

## 3. 技术选型总表

| 组件 | 定位 | 当前项目建议接入点 | 适合当前项目的用途 | 接入成本 | 推荐级别 |
| --- | --- | --- | --- | --- | --- |
| `pymovements` | 现代眼动数据处理底座 | `io.py`、`preprocess.py`、`events.py` | 提升数据加载、公共数据集对接、事件处理一致性 | 中 | 高 |
| `pypillometry` | 瞳孔信号预处理与建模 | 新增 `pupil_preprocess.py`，并接到 `features.py` | 为认知负荷识别补最关键的 pupil 特征链路 | 中 | 高 |
| `uneye` | 深度学习事件检测 | `events.py` 的可选 backend | 把 fixation/saccade 检测从规则法升级成学习法 | 中 | 中高 |
| `EyeBench` | 眼动预测 benchmark/评估协议 | `notebooks/` 或 `experiments/` | 用统一 protocol 验证“眼动能预测什么” | 低到中 | 中高 |
| `deep_em_classifier` | 研究型深度事件分类器 | `events.py` 的实验 backend | 对比 `uneye` 或作为研究增强项 | 中到高 | 中 |
| `PyTrack` | 传统眼动分析工具箱 | 仅参考其统计与数据组织方式 | 可借鉴分析流程，但不建议作为主干依赖 | 低 | 中低 |
| `PyGazeAnalyser` | 老牌高层分析工具 | 仅参考，不建议深接入 | 适合读思路，不适合作为当前项目主干 | 低 | 低 |
| `PyPupilEXT` / `Open-PupilEXT` | 从眼部视频提 pupil | 新增独立视频预处理入口 | 只有你拿到眼部视频时才有价值 | 高 | 条件性推荐 |

---

## 4. 逐项接入建议

### 4.1 `pymovements`

来源：

- https://github.com/pymovements/pymovements

它最适合补的是**眼动数据工程底座**，不是直接做认知负荷分类。

适合当前项目的接法：

- 在 `io.py` 增加一个 `pymovements` adapter
- 先把支持的输入格式映射成当前项目的 `GazeRecording`
- 只复用它的数据读取、标准化、公共数据集接口，不强行替换现有全链路

推荐原因：

- 当前项目已经有自己的统一领域对象 `GazeRecording`
- 所以最合理的方式不是“全量改造”，而是让 `pymovements` 变成一个**输入与处理增强层**
- 这样能保留你现有 Dashboard、特征工程和建模链路

最直接收益：

- 更容易接入公开眼动数据集
- 更容易做“方法对比”与“复现实验”
- 让项目更像严肃研究工具，而不是只处理自定义 CSV

结论：

- **应该接**
- 但应该以 adapter 方式接，不要重写主干

### 4.2 `pypillometry`

来源：

- https://github.com/ihrke/pypillometry
- 文档：https://ihrke.github.io/pypillometry/

这是当前项目最值得补的一块，因为认知负荷识别里，**瞳孔反应往往比单纯 gaze path 更直接**。

适合当前项目的接法：

- 新增 `pupil_preprocess.py`
- 在 `analyze_recording()` 或 `extract_features()` 前增加 pupil-only 预处理链
- 输出基线校正后的 pupil 序列，以及额外 pupil 特征

建议增加的特征：

- baseline-corrected pupil mean
- peak pupil dilation
- pupil dilation latency
- pupil change rate
- blink-corrected pupil variance
- trial/window 内 tonic / phasic pupil summary

为什么优先级高：

- 当前项目虽然已经有 `pupil_baseline`、`pupil_change_rate`
- 但还没有真正专业的 pupillometry 处理流程
- 对“认知负荷识别”来说，这一层的增益会明显高于继续堆普通 gaze 统计量

结论：

- **这是当前项目最优先的新增包之一**

### 4.3 `uneye`

来源：

- https://github.com/berenslab/uneye

它的价值不在于直接预测 cognitive load，而在于**先把事件检测质量提高**。

适合当前项目的接法：

- 在 `events.py` 中增加 `backend="uneye"` 选项
- 保留现有阈值法作为默认稳定版
- 新 backend 只作为增强模式或实验模式

为什么值得加：

- 认知负荷模型很多特征都依赖 fixation / saccade 分割质量
- 你现在已有规则法事件检测
- `uneye` 可以作为“AI 驱动事件层”的很好展示点

对作品集的意义：

- 能展示你不是只会做统计特征
- 还会把深度学习接入到人因信号处理前端

结论：

- **推荐作为第二阶段接入**

### 4.4 `EyeBench`

来源：

- https://github.com/EyeBench/eyebench
- https://eyebench.github.io/eyebench/

它不是一个“业务依赖包”，更像是**实验评价框架**。

适合当前项目的接法：

- 不建议直接塞进主产品依赖
- 建议放到 `notebooks/`、`experiments/` 或单独的 benchmark 脚本里

适合用途：

- 比较不同特征集对预测任务的效果
- 比较不同建模方式
- 给你的项目增加“标准化评估 protocol”这一层

对面试的价值很高，因为它能让你讲：

- 你不仅做了模型
- 还知道怎么做 protocol、benchmark、generalization evaluation

结论：

- **推荐接入，但定位为实验评估层，不是生产依赖**

### 4.5 `deep_em_classifier`

来源：

- https://github.com/MikhailStartsev/deep_em_classifier

它和 `uneye` 类似，都是为了提升事件层，但更偏研究代码。

适合当前项目的接法：

- 作为 `events.py` 的实验 backend
- 与规则法、`uneye` 做效果对比

结论：

- 可以接
- 但优先级低于 `uneye`

### 4.6 `PyTrack`

来源：

- https://github.com/titoghose/PyTrack

它更像一个传统分析工具箱，适合参考：

- 数据组织方式
- 统计分析习惯
- notebook 级工作流

但不太适合当前项目深度接入，因为：

- 你自己的架构已经比它更统一
- 直接混进来容易变成“双套数据结构并存”

结论：

- **建议参考，不建议深接入**

### 4.7 `PyGazeAnalyser`

来源：

- https://github.com/esdalmaijer/PyGazeAnalyser

它适合做：

- 方法参考
- 经典思路回顾

不适合做：

- 你当前项目的核心依赖

原因很简单：

- 仓库偏老
- 当前项目的工程结构已经比它更贴近你的作品集目标

结论：

- **只参考，不接入**

### 4.8 `PyPupilEXT` / `Open-PupilEXT`

来源：

- https://github.com/openPupil/PyPupilEXT
- https://github.com/openPupil/Open-PupilEXT

只有在一种情况下建议优先考虑：

- 你手上拿到的是眼部视频，而不是已经导出的 pupil 数值

对当前项目来说，它们不应该直接插入主分析链，而应该作为：

- 独立前置步骤
- “视频 -> pupil diameter -> CSV/Parquet -> 当前项目”的前处理桥

结论：

- **不是当前项目主线**
- 仅在数据源发生变化时再接入

---

## 5. 当前项目最优实施顺序

### Phase 1：最小可落地闭环

目标：

- 尽快让项目具备“认知负荷识别”而不是只有“眼动分析”

推荐：

1. 接 `pypillometry`
2. 扩展 `features.py`，增加 workload-oriented pupil features
3. 在 `modeling.py` 增加 cognitive load 分类/回归实验入口
4. 在 Dashboard 中增加“认知负荷实验台”

这是最短路径，因为：

- 不破坏现有项目结构
- 技术风险最低
- 对作品集增益最大

### Phase 2：增强事件层

推荐：

1. 接 `uneye`
2. 把 `events.py` 改成可切换 backend
3. 比较规则法与学习法对下游 workload prediction 的影响

这样你就能在面试里明确讲出：

- AI 不只用于最终分类
- 也用于前端信号结构化

### Phase 3：标准化 benchmark 与泛化验证

推荐：

1. 引入 `EyeBench` 风格的评估 protocol
2. 增加 LOSO / cross-task / cross-condition 验证
3. 输出 benchmark 报告

这一步最适合强化“研究严谨性”和“工程化可信度”。

---

## 6. 最终推荐方案

如果只能选一条最合理的路线，建议如下：

| 层级 | 推荐方案 |
| --- | --- |
| 数据与预处理层 | `pymovements` + 当前项目已有 `io/preprocess` |
| 瞳孔负荷层 | `pypillometry` |
| 事件层增强 | `uneye` |
| 建模层 | 当前项目 `modeling.py` + `XGBoost/SVM/RandomForest` |
| 多模态层 | 当前项目 `multimodal.py`，后续接 HR/GSR |
| 评估层 | `EyeBench` 风格 benchmark |

一句话总结：

**当前项目最值得优先接入的是 `pypillometry` 和 `pymovements`；最值得作为 AI 强化项接入的是 `uneye`；最值得作为研究展示层接入的是 `EyeBench`。**

---

## 7. 不建议现在做的事

为了避免项目失焦，当前不建议：

- 同时接入太多老工具箱
- 为了“包多”而引入双套事件/数据结构
- 先做 raw eye video pipeline
- 先追求深度模型端到端，而跳过可解释的基线特征链路

对这个项目来说，最重要的不是“依赖多”，而是：

**能清楚展示你如何把眼动信号转成认知负荷指标、模型和人因解释。**

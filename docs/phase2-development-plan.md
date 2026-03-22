# Phase 2 开发计划：作品集增强迭代

**版本**: v3.0 (Claude + Codex 双方定稿)
**日期**: 2026-03-22
**目标**: 围绕华为人因岗位 JD 匹配度，系统性增强项目的研究深度和 AI 能力展示
**状态**: PLAN_AGREED — 已通过 Phase 2 方案讨论阶段

---

## 1. 开发总览

### 1.1 核心原则

所有功能开发服务于同一目标：**更强地证明你是一个"懂人因研究、懂多模态数据、会用 AI、还能把方法做成系统"的候选人。**

### 1.2 关键架构决策（Phase 2 讨论共识）

以下决策由 Claude（架构师）和 Codex（主程序员/技术顾问）在 Phase 2 讨论阶段达成共识：

1. **前插 foundation 阶段**：在实现业务功能前，先补一层薄的公共分析接口层，防止 AOI、统计、批量模块各自围绕不同 DataFrame 形态长出不兼容的实现。
2. **AOI 基于 fixation 而非 sample**：性能与领域正确性的双重要求。AOI 指标（TTFF、驻留时长、回视次数）的分析单元是 fixation event。
3. **统计检验粒度约束**：`compare_conditions()` 只接受 subject x condition 粒度的汇总表，禁止 sample/fixation 级输入，防止 p 值虚高。
4. **生理数据走独立 loader**：`load_physio_csv()` 作为独立时序加载器，最终进入 `MultiModalData`，不混入 gaze `io.py`。
5. **scipy 保持 optional**：统计模块做 lazy import，核心 pipeline 不引入重依赖。
6. **quality_grade P0 只展示不过滤**：v0 仅基于 tracking_ratio 分级，不作为自动剔除依据。
7. **列契约测试**：关键公共函数的输出列在 docstring + test 中锁死，防止后续漂移。
8. **显著性指标展示已完成**：NSS/SIM/KL 已在 Dashboard 中打通，从 P1 任务列表移除。

### 1.3 阶段划分

```
Phase 2a.foundation (半天)        Phase 2a.feature (4-5天)
━━━━━━━━━━━━━━━━━━━━             ━━━━━━━━━━━━━━━━━━━━
┌──────────────────┐             ┌──────────────────┐
│ 1. fixation_table│             │ 1. AOI 分析模块   │
│ 2. 实验设计列传递 │             │ 2. 统计分析模块   │
│ 3. quality_grade │             │ 3. 产品场景模板   │
│ 4. ApEn性能修复  │             │                  │
│ 5. 列契约测试    │             │                  │
└──────────────────┘             └──────────────────┘

Phase 2b (4天)                    Phase 2c (5天)
━━━━━━━━━━━━━━━━━━━━             ━━━━━━━━━━━━━━━━━━━━
┌──────────────────┐             ┌──────────────────┐
│ 1. 批量分析+报告  │             │ 1. Scanpath回放   │
│ 2. 数据质量面板UI │             │ 2. I-DT算法扩展   │
│ 3. 生理CSV→Multi │             │ 3. LLM自动报告    │
└──────────────────┘             └──────────────────┘
```

---

## 2. Phase 2a.foundation：公共分析接口层

**目标**：在最小工作量（半天）内建立后续所有模块共享的数据契约，修复已知性能隐患。

### 2.1 公共 fixation_table() 函数

**位置**: `src/gaze_toolkit/events.py` 或新建 `src/gaze_toolkit/tables.py`

```python
def fixation_table(recording: GazeRecording) -> pd.DataFrame:
    """
    从 GazeRecording 的 events 中提取标准化注视事件表。

    这是 AOI 模块、统计模块、批量分析的公共输入层。

    Returns
    -------
    DataFrame
        列契约（锁死，不可省略）:
        - event_index: int       # 事件在 events 列表中的索引
        - start_time_ms: float   # 注视开始时间
        - end_time_ms: float     # 注视结束时间
        - duration_ms: float     # 注视持续时长
        - centroid_x: float      # 注视点质心 X
        - centroid_y: float      # 注视点质心 Y
        - session_id: str        # 会话标识（可选，来自 metadata）
        - subject_id: str        # 被试标识（可选，来自 metadata）
        - condition: str         # 实验条件（可选，来自 metadata）
        - trial: str             # 试次标识（可选，来自 metadata）
    """
    ...
```

### 2.2 build_feature_dataset 补实验设计列

修改 `pipeline.py` 中的 `build_feature_dataset()`，从 `recording.metadata` 中传递：
- `subject_id`
- `condition`
- `trial`
- `quality_grade`（由 2.3 提供）
- `segment_name`（如果来自分段分析）

### 2.3 quality_grade 基础计算

从 `analysis.py` 已有的 `quality_summary` 升级，提取为独立函数：

```python
def compute_quality_grade(recording: GazeRecording) -> str:
    """
    基于 tracking_ratio 的轻量级质量分级。

    Returns: "优" / "良" / "可用" / "建议剔除"

    注意：v0 仅用于展示标签，不作为自动过滤依据。
    后续版本将补充 max_gap_duration 和 interpolated_ratio。
    """
    ...
```

### 2.4 approximate_entropy 性能修复

**P0 blocker**：当前 `features.py:143-148` 的 `approximate_entropy()` 显式构造 O(n^2) 距离矩阵，120Hz x 5min = 36,000 点时内存约 10GB，会直接 OOM。

修复方案：
- 对超过 `max_samples`（默认 2000）的信号自动降采样后再计算
- `batch_analyze` 路径默认 `include_complexity=False`
- 保持对短记录的精确计算不变

### 2.5 列契约测试

为以下函数编写专项测试，锁死输出列名和类型：
- `fixation_table()` — 必须包含上述 10 列
- `compare_conditions()` — 运行时 assert 输入必须是 subject x condition 粒度
- `load_physio_csv()` — 输出必须包含时间列，可直接进入 `MultiModalData.add_modality()`

测试文件: `tests/test_foundation.py`

---

## 3. Phase 2a.feature：P0 功能详细设计

### 3.1 AOI 分析模块

**为什么最优先**: AOI 是人因眼动研究的核心语言。没有 AOI，面试官会觉得你只会做信号处理，不会做人因解释。

#### 3.1.1 功能清单

| 子功能 | 说明 | 面试价值 |
|-------|------|---------|
| AOI 定义 | 在 Dashboard 上通过坐标或交互方式定义矩形/多边形 AOI | 展示你理解 AOI 概念 |
| AOI 归属判定 | 每个 **fixation** 判定属于哪个 AOI（基于 fixation_table） | 基础分析能力 |
| 首次注视到达时间 (TTFF) | 从试次开始到首次注视落入 AOI 的时间 | 衡量信息可发现性 |
| AOI 总驻留时长 | 在 AOI 内的总注视时间 | 衡量信息加工深度 |
| AOI 注视占比 | AOI 驻留时长 / 总记录时长 | 注意力分配 |
| AOI 访问次数 | 进入 AOI 的次数 | 视觉搜索模式 |
| 回视次数 | 离开 AOI 后再次返回的次数 | 信息理解难度 |
| AOI 转移矩阵 | AOI 之间的注视转移概率热力图（基于 fixation 序列 crosstab） | 信息流分析 |

#### 3.1.2 代码设计

**新增文件**: `src/gaze_toolkit/aoi.py`

```python
"""AOI (Area of Interest) 分析模块"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class AOI:
    """兴趣区域定义"""
    name: str
    # 矩形: (x_min, y_min, x_max, y_max)
    # 多边形: [(x1,y1), (x2,y2), ...]
    region: object
    region_type: str = "rectangle"  # "rectangle" | "polygon"


@dataclass
class AOIMetrics:
    """单个 AOI 的分析指标"""
    aoi_name: str
    first_fixation_time: Optional[float]  # TTFF (ms)
    total_dwell_time: float               # 总驻留时长 (ms)
    dwell_proportion: float               # 驻留占比
    fixation_count: int                   # 注视次数
    visit_count: int                      # 访问次数 (进入次数)
    revisit_count: int                    # 回视次数
    mean_fixation_duration: float         # 平均注视时长 (ms)


def define_aoi(name: str, x_min: float, y_min: float,
               x_max: float, y_max: float) -> AOI:
    """定义矩形 AOI"""
    return AOI(name=name, region=(x_min, y_min, x_max, y_max),
               region_type="rectangle")


def assign_fixations_to_aoi(
    fixation_df: pd.DataFrame,
    aois: List[AOI],
) -> pd.DataFrame:
    """
    将 fixation_table() 输出的注视事件分配到 AOI。

    重要：输入必须是 fixation 级别（来自 fixation_table()），
    不是 sample 级别。AOI 碰撞检测基于 centroid_x / centroid_y。

    Parameters
    ----------
    fixation_df : DataFrame
        fixation_table() 的输出，至少包含 centroid_x, centroid_y, duration_ms
    aois : list of AOI
        AOI 定义列表

    Returns
    -------
    DataFrame
        添加了 aoi_name 列的注视事件表（未命中任何 AOI 的行值为 None）
    """
    ...


def compute_aoi_metrics(
    fixations_with_aoi: pd.DataFrame,
    aois: List[AOI],
    total_duration: float,
) -> Dict[str, AOIMetrics]:
    """计算每个 AOI 的完整指标集"""
    ...


def compute_transition_matrix(
    fixations_with_aoi: pd.DataFrame,
    aoi_names: List[str],
) -> pd.DataFrame:
    """
    计算 AOI 转移概率矩阵。

    实现要求：基于 fixation 序列的 aoi_name 列做
    pd.crosstab + 行归一化，不使用 Python 双重循环。

    Returns
    -------
    DataFrame
        行=来源AOI, 列=目标AOI, 值=转移概率
    """
    ...
```

#### 3.1.3 Dashboard 集成

在"单次会话分析"标签页中新增一个区块：

```
+-----------------------------------------------------+
|  兴趣区域 (AOI) 分析                                  |
|                                                       |
|  +--------------------+  +------------------------+  |
|  |  AOI 定义区         |  |  AOI 指标汇总表          |  |
|  |  [添加 AOI] 按钮     |  |  TTFF | 驻留 | 回视 ... |  |
|  |  AOI 名称/坐标输入   |  |                        |  |
|  |  scanpath+AOI叠加图  |  |  +------------------+  |  |
|  |                    |  |  |  AOI 转移矩阵热力图 |  |  |
|  |                    |  |  +------------------+  |  |
|  +--------------------+  +------------------------+  |
+-----------------------------------------------------+
```

新增 Dashboard 区块建议写成独立的 renderer 函数（可在 dashboard.py 内，但函数边界清晰），不做文件级拆分。

#### 3.1.4 面试话术

"这里是 AOI 分析。在人因研究里，我们不只关心注视轨迹本身，更关心用户看了哪些功能区域、看了多久、什么顺序看的。比如评测手机设置页面时，我会把导航栏、搜索框、各个设置分组定义为 AOI，然后自动计算首次到达时间、驻留时长、回视次数和转移矩阵，来判断信息架构是否合理。"

---

### 3.2 统计分析模块

**为什么优先**: 作为心理学博士，缺少统计分析模块会削弱你的专业可信度。

#### 3.2.1 功能清单

| 子功能 | 说明 | 适用场景 |
|-------|------|---------|
| 独立样本 t-test | 两组被试间差异 | A/B 界面设计对比 |
| 配对样本 t-test | 同一被试前后差异 | 被试内实验设计 |
| Wilcoxon 符号秩检验 | 非参数配对检验 | 小样本 / 非正态 |
| Mann-Whitney U | 非参数独立检验 | 小样本组间比较 |
| 单因素重复测量 ANOVA | 3+ 条件差异 | 多条件实验 |
| 效应量 (Cohen's d, eta-sq) | 实际差异大小 | 结果解释 |
| 描述性统计表 | 均值/SD/CI | 基础报告 |

#### 3.2.2 代码设计

**新增文件**: `src/gaze_toolkit/statistics.py`

**依赖**: `scipy.stats`（保持 optional，lazy import + 清晰报错）

```python
"""人因研究统计分析模块

依赖: scipy (optional)。导入时 lazy check，缺失时给出安装提示。
"""

from dataclasses import dataclass
from typing import Optional, List
import numpy as np
import pandas as pd


@dataclass
class StatTestResult:
    """统计检验结果"""
    test_name: str           # 检验名称
    statistic: float         # 检验统计量
    p_value: float           # p 值
    effect_size: float       # 效应量
    effect_size_name: str    # 效应量名称 (Cohen's d / eta-sq / r)
    ci_lower: float          # 95% CI 下界
    ci_upper: float          # 95% CI 上界
    n: int                   # 样本量
    conclusion: str          # 中文结论 (如 "差异显著, p < .05, d = 0.72")

    def to_apa_string(self) -> str:
        """输出 APA 格式的报告字符串"""
        ...


def independent_t_test(group1: np.ndarray, group2: np.ndarray,
                       var_name: str = "指标") -> StatTestResult:
    """独立样本 t 检验 + Cohen's d"""
    ...


def paired_t_test(before: np.ndarray, after: np.ndarray,
                  var_name: str = "指标") -> StatTestResult:
    """配对样本 t 检验 + Cohen's d"""
    ...


def wilcoxon_test(before: np.ndarray, after: np.ndarray,
                  var_name: str = "指标") -> StatTestResult:
    """Wilcoxon 符号秩检验 + r 效应量"""
    ...


def mann_whitney_test(group1: np.ndarray, group2: np.ndarray,
                      var_name: str = "指标") -> StatTestResult:
    """Mann-Whitney U 检验 + r 效应量"""
    ...


def repeated_measures_anova(data: pd.DataFrame,
                            dv: str, within: str,
                            subject: str) -> StatTestResult:
    """单因素重复测量方差分析 + eta-sq"""
    ...


def descriptive_table(data: pd.DataFrame,
                      group_col: str,
                      value_cols: List[str]) -> pd.DataFrame:
    """
    生成分组描述性统计表。

    Returns
    -------
    DataFrame
        包含 mean, sd, median, min, max, n, 95% CI 的汇总表
    """
    ...


def compare_conditions(
    summary_df: pd.DataFrame,
    condition_col: str,
    metric_cols: List[str],
    paired: bool = False,
    subject_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    批量对多个指标进行条件间比较。

    粒度约束（运行时 assert）：
    summary_df 必须是 subject x condition 粒度的汇总表，
    即每个 (subject_col, condition_col) 组合最多一行。
    不接受 sample 级或 fixation 级输入。

    自动选择参数/非参数检验（基于正态性检验），
    输出每个指标的检验结果汇总表。
    """
    ...
```

#### 3.2.3 Dashboard 集成

新增一个标签页"统计分析" 或嵌入"意图建模实验台"中：

```
+-----------------------------------------------------+
|  统计分析                                             |
|                                                       |
|  条件选择: [careful] vs [skim]                        |
|  指标选择: [注视次数] [平均注视时长] [扫视幅度] ...     |
|  检验类型: [自动选择] / [t-test] / [Wilcoxon] / ...   |
|                                                       |
|  +------------------------------------------------+  |
|  |  描述性统计表                                    |  |
|  |  指标      | careful M(SD) | skim M(SD) | p    |  |
|  |  注视次数   | 45.2 (8.3)   | 28.1 (6.7) | .001 |  |
|  |  ...                                           |  |
|  +------------------------------------------------+  |
|                                                       |
|  +-----------------+  +--------------------------+  |
|  |  箱线图对比       |  |  效应量森林图              |  |
|  +-----------------+  +--------------------------+  |
+-----------------------------------------------------+
```

#### 3.2.4 面试话术

"这里是统计分析模块。做人因研究不能只看模型准确率，还需要用经典的假设检验来验证条件间差异是否具有统计显著性。系统会根据数据正态性自动选择参数或非参数检验，输出 p 值、效应量和 APA 格式的结论。比如在 careful vs skim 任务里，注视次数的 Cohen's d 达到了大效应量，说明两种阅读策略在眼动行为上有本质差异。"

---

### 3.3 产品场景研究模板

**为什么优先**: JD 第一条明确要求"针对手机/耳机/穿戴/平板等产品制定研究计划"。

#### 3.3.1 功能清单

新增一个"研究场景模板"模块，至少包含一个完整可演示的场景。

**推荐首选场景**: 手机设置页信息架构评测

```yaml
# configs/scenarios/phone_settings_evaluation.yaml

scenario:
  name: "手机设置页信息架构评测"
  product: "手机"
  channel: "视觉 + 触觉"

research_design:
  type: "被试内设计 (within-subject)"
  iv: "设置页方案 (A: 当前版 vs B: 优化版)"
  dv:
    eye_tracking:
      - "目标设置项首次注视到达时间 (TTFF)"
      - "搜索路径最优比"
      - "注视熵 (Gaze Entropy)"
      - "任务期间 TEPR (认知负荷指标)"
    behavior:
      - "任务完成时间"
      - "错误率"
    subjective:
      - "SUS 可用性量表"
      - "NASA-TLX 任务负荷指数"
  sample_size: "N=30 (G*Power: a=0.05, d=0.5, power=0.8)"
  counterbalancing: "拉丁方设计"

tasks:
  - id: "T1"
    description: "找到并打开 Wi-Fi 设置"
    aoi_regions:
      - name: "导航栏"
        region: [0, 0, 1920, 120]
      - name: "搜索框"
        region: [100, 130, 1820, 210]
      - name: "网络分组"
        region: [100, 220, 1820, 500]
      - name: "显示分组"
        region: [100, 510, 1820, 790]
  - id: "T2"
    description: "找到蓝牙开关并切换状态"
  - id: "T3"
    description: "调整屏幕亮度到 70%"

analysis_plan:
  primary:
    - "配对 t-test / Wilcoxon: 方案A vs 方案B 各指标差异"
    - "重复测量 ANOVA: 任务 x 方案 交互效应"
  secondary:
    - "AOI 转移矩阵对比"
    - "ML模型: 根据眼动特征预测用户是否找到目标"
```

#### 3.3.2 在 Dashboard 中展示

新增标签页或子区块"产品评测场景"：

- 展示研究方案 YAML（证明你会做研究设计）
- 预定义的 AOI 配置（证明你理解产品评测流程）
- 演示数据 + 自动分析结果
- 统计分析 + ML 建模 联合输出

---

## 4. Phase 2b：P1 功能设计

### 4.1 批量分析 + 报告导出

#### 功能设计

```python
# src/gaze_toolkit/batch.py

def batch_analyze(
    file_paths: List[str],
    config: Optional[dict] = None,
) -> pd.DataFrame:
    """
    批量分析多个眼动记录文件。

    基于 foundation 层的公共接口（fixation_table, quality_grade,
    build_feature_dataset），复用统一的分析流程。

    Returns
    -------
    DataFrame
        每行一个记录，列为所有提取的特征 + 实验设计列 + quality_grade
    """
    ...


def export_report(
    analysis_results: dict,
    format: str = "html",  # "html" | "markdown" | "pdf"
    output_path: str = "report.html",
    include_plots: bool = True,
) -> str:
    """
    导出分析报告。

    报告结构:
    1. 实验概览 (场景、被试数、数据质量)
    2. 描述性统计表
    3. 关键可视化 (scanpath, heatmap, 箱线图)
    4. 统计检验结果
    5. ML建模结果
    6. 结论与建议
    """
    ...
```

#### 面试价值

"我做了批量分析功能，一次传入多个记录文件就能自动跑完整个分析流程，最后生成可分享的 HTML 报告。这个对提升研究效率非常关键，也是 JD 里强调的'运用 AI 技术提升研究效率'的直接体现。"

---

### 4.2 数据质量控制面板 UI

#### 功能设计

```python
# src/gaze_toolkit/quality.py

@dataclass
class QualityReport:
    """数据质量报告（完整版，P1 阶段实现）"""
    tracking_ratio: float          # 追踪率 (有效样本占比)
    total_samples: int             # 总样本数
    valid_samples: int             # 有效样本数
    missing_segments: int          # 缺失段数
    max_gap_duration_ms: float     # 最大缺失段时长
    interpolated_ratio: float      # 插值填补比例
    blink_count: int               # 眨眼次数
    recording_duration_s: float    # 记录时长
    sampling_rate_actual: float    # 实际采样率
    quality_grade: str             # 质量等级: "优" / "良" / "可用" / "建议剔除"


def assess_quality(recording) -> QualityReport:
    """评估眼动记录数据质量（完整版）"""
    ...
```

#### Dashboard 展示

在"单次会话分析"顶部增加一行质量指标卡片：

```
+----------+ +----------+ +----------+ +----------+ +----------+
| 追踪率    | | 采样率    | | 记录时长   | | 缺失段    | | 质量等级  |
| 94.2%    | | 119.8 Hz | | 45.3 s   | | 3 段     | | 良       |
+----------+ +----------+ +----------+ +----------+ +----------+
```

---

### 4.3 真实生理数据 CSV 导入

#### 功能设计

**重要**：独立的时序加载器，不混入 gaze `io.py`。最终进入 `MultiModalData`。

```python
# src/gaze_toolkit/physio_loader.py（独立文件）

def load_physio_csv(
    file_path: str,
    timestamp_col: str = "timestamp",
    signal_cols: Optional[List[str]] = None,
    sampling_rate: Optional[float] = None,
) -> pd.DataFrame:
    """
    加载外部生理信号 CSV。

    列契约：输出必须包含 timestamp_ms 列（统一为毫秒），
    可直接通过 MultiModalData.add_modality() 注册。

    支持:
    - HR (心率)
    - GSR / EDA (皮电)
    - 呼吸率
    - 加速度计数据

    自动处理时间戳格式转换和采样率推断。
    """
    ...
```

---

## 5. Phase 2c：P2 功能设计

### 5.1 Scanpath 时间回放

在 Dashboard 中增加一个回放控件，按时间顺序动态展示注视点移动：

- 播放/暂停/速度控制
- 当前时间指示器
- 与信号总览图的时间同步

### 5.2 I-DT 事件检测算法扩展

在 `events.py` 中增加 I-DT (Identification by Dispersion-Threshold) 算法作为第二后端：

- 基于空间散布度而非速度
- 可在 Dashboard 中对比 I-VT vs I-DT 结果
- 展示你理解经典眼动方法的深度
- 注意：AOI 和统计模块已通过 fixation_table() 解耦，不依赖特定事件检测后端

### 5.3 LLM 驱动的自动报告生成

基于模板 + LLM 的混合方案（建议先做模板式，再考虑自由生成）：

```python
# src/gaze_toolkit/report_generator.py

def generate_insight_report(
    analysis_results: dict,
    scenario_context: dict,
    use_llm: bool = False,
) -> str:
    """
    生成人因分析洞察报告。

    模板模式: 基于规则填充预定义结构
    LLM模式: 调用 Claude API 生成自由文本洞察
    """
    ...
```

---

## 6. 技术实现注意事项

### 6.1 依赖管理

- P0 foundation 只需已有依赖（numpy, pandas），无需引入新包
- 统计分析模块使用 `scipy.stats`（保持 optional extra，lazy import）
- 重复测量 ANOVA 可选用 `pingouin`（加到 optional deps）

### 6.2 测试策略

```
tests/test_foundation.py     # 列契约测试 (fixation_table, compare_conditions 粒度)
tests/test_aoi.py            # AOI 模块测试
tests/test_statistics.py     # 统计分析测试
tests/test_batch.py          # 批量分析测试
tests/test_quality.py        # 质量控制测试
```

### 6.3 Dashboard 整合顺序

1. 先实现核心计算逻辑 + 测试
2. 再在 Dashboard 中增加展示区块（写成独立 renderer 函数）
3. 最后调整样式和交互细节

### 6.4 代码质量

- 保持中文 docstring + 英文变量名的现有风格
- 所有新增 public API 都需要在 `__init__.py` 中导出
- 保持现有的 `register_*` 扩展点模式
- scipy 等可选依赖使用 lazy import pattern，缺失时给出清晰安装提示

---

## 7. 对现有问题的修复建议

### 7.1 visualization.py 乱码修复

`visualization.py:466` 存在 mojibake 字符串，需要修复为正确的中文。

### 7.2 需求文档与实际代码对齐

`docs/眼动分析系统_开发架构与需求文档_claude版本.md` 中描述了大量未实现功能。建议：

- 要么在文档中明确标注"已实现"/"规划中"状态
- 要么将该文档重命名为 `architecture-vision.md`（愿景文档），与 `project-architecture-overview.md`（实际架构文档）区分

### 7.3 README 结构优化

当前 README 已经很好，建议在 Phase 2 功能完成后更新：

- 在"当前 MVP 范围"中增加 AOI 分析和统计分析
- 在"典型作品集演示路径"中增加 AOI + 统计 的演示步骤

---

## 8. 开发优先级排序（最终执行顺序）

| 序号 | 阶段 | 任务 | 预计耗时 | 依赖 |
|-----|------|------|---------|------|
| 0a | foundation | fixation_table() + 列契约测试 | 2h | 无 |
| 0b | foundation | build_feature_dataset 补实验设计列 | 1h | 无 |
| 0c | foundation | quality_grade 基础计算 | 1h | 无 |
| 0d | foundation | approximate_entropy 降采样修复 | 1h | 无 |
| 1 | feature | AOI 核心计算模块 (`aoi.py`) | 1天 | #0a |
| 2 | feature | AOI Dashboard 集成 | 1天 | #1 |
| 3 | feature | 统计分析核心模块 (`statistics.py`) | 1天 | #0a, #0b |
| 4 | feature | 统计分析 Dashboard 集成 | 0.5天 | #3 |
| 5 | feature | 产品场景 YAML 模板 | 0.5天 | #1, #3 |
| 6 | feature | 产品场景 Dashboard 展示 | 1天 | #5 |
| 7 | bugfix | 修复 visualization.py 乱码 | 0.5h | 无 |
| 8 | P1 | 批量分析 + 报告导出 | 2天 | #0-6 |
| 9 | P1 | 数据质量面板 UI | 1天 | #0c |
| 10 | P1 | 真实生理 CSV 导入 → MultiModalData | 1天 | 无 |
| 11 | P2 | Scanpath 回放 | 2天 | 无 |
| 12 | P2 | I-DT 算法扩展 | 1天 | 无 |
| 13 | P2 | LLM 自动报告 | 2天 | #8 |
| 14 | doc | 更新 README 和文档 | 0.5天 | 全部 |

---

## 9. 完成后的项目 JD 匹配度预估

| JD 要求 | 当前匹配度 | Phase 2a 后 | Phase 2b 后 |
|---------|----------|------------|------------|
| 针对终端产品制定人因研究方案 | 中 | **高** | **很高** |
| 视/听/触多通道研究 | 中 | 中高 | 高 |
| AI 提升研究效率/信度/效度 | 高 | **很高** | **很高** |
| 人因工程方法论理解 | 高 | **很高** | **很高** |
| 从生理/心理/行为角度拆解变量 | 中高 | 高 | **很高** |
| Python 数据分析能力 | 高 | 高 | 高 |
| AI驱动人因应用（JD第4点） | 高 | **很高** | **很高** |

---

*完成 Phase 2a（foundation + feature）后，你的项目将全面覆盖 JD 的所有核心要求，尤其在"AI 驱动人因应用"和"独立完成人因研究全流程"上的证明力会显著增强。*

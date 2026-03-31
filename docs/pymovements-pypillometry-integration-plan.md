# `pymovements` + `pypillometry` 具体接入实施方案

## 1. 目标

这份方案只解决两个明确目标：

1. 把 `pymovements` 接入当前项目，增强眼动数据读取、标准化和公共数据集接入能力
2. 把 `pypillometry` 接入当前项目，补齐“认知负荷识别”最关键的瞳孔预处理与 pupil 特征链路

这份方案默认遵循当前项目已有架构，不重写主干，不引入第二套并行系统。

---

## 2. 当前项目的可接入位置

从现有代码看，接入点已经很清楚：

- 数据接入入口：`src/gaze_toolkit/io.py`
- 统一领域对象：`src/gaze_toolkit/types.py`
- 默认预处理链：`src/gaze_toolkit/preprocess.py`
- 事件检测：`src/gaze_toolkit/events.py`
- 特征提取：`src/gaze_toolkit/features.py`
- 单记录分析编排：`src/gaze_toolkit/analysis.py`
- 多记录特征表：`src/gaze_toolkit/pipeline.py`
- 交互界面：`src/gaze_toolkit/dashboard.py`

这意味着最合理的接法不是替换现有链路，而是：

```text
外部包
  -> adapter / wrapper
  -> 当前项目统一对象 GazeRecording
  -> preprocess / events / features / modeling / dashboard
```

这也是这份方案的核心原则。

---

## 3. 关键假设

下面这些前提是当前方案的基础，如果后面真实数据不满足，需要再调整：

- 当前主要输入仍然是样本级眼动表格，而不是原始眼部视频
- 当前主分析对象仍然是单眼或已汇总后的 `x/y/pupil/timestamp_ms`
- 当前项目暂不引入 binocular 左右眼双通道复杂建模
- 当前认知负荷识别优先采用“可解释特征 + 基线模型”，而不是先做端到端深度学习
- 当前 Dashboard 仍然以 Streamlit 为主，不新增第二套 Web 前端

这些假设和你当前项目定位是一致的，也能最大化作品集收益。

---

## 4. 总体实施顺序

建议顺序不是先接两个包一起大改，而是：

### Phase A：先接 `pypillometry`

原因：

- 它直接提升认知负荷识别能力
- 改动面较集中
- 对作品集叙事最强

### Phase B：再接 `pymovements`

原因：

- 它更偏底座增强
- 更适合在 `io/preprocess/events` 层做 adapter 化扩展
- 不会立即改变负荷识别效果，但会提升项目研究工具属性

---

## 5. `pypillometry` 接入实施方案

## 5.1 接入目标

补齐当前项目缺少的 pupillometry 专业链路，包括：

- blink-aware pupil 清洗
- baseline correction
- pupil trend / response summary
- 更贴近认知负荷的 pupil 特征

当前 `features.py` 已有：

- `pupil_baseline`
- `pupil_change_rate`

但这还不够。它们更像一般统计量，不是完整的 cognitive load pupil pipeline。

---

## 5.2 最小正确方案

推荐新增一个单独模块：

- `src/gaze_toolkit/pupil_preprocess.py`

不要把 `pypillometry` 逻辑直接塞进 `preprocess.py`，因为：

- 当前 `preprocess.py` 面向一般 gaze 信号
- pupillometry 是更专业、依赖更多参数的一条子链
- 单独拆模块更利于在 Dashboard 中做可视化和可选开关

### 推荐新增对象

```python
@dataclass
class PupilProcessingResult:
    cleaned_pupil: pd.Series
    baseline_corrected_pupil: pd.Series
    blink_mask: pd.Series
    metadata: dict[str, Any]
```

### 推荐新增函数

```python
def preprocess_pupil_signal(
    recording: GazeRecording,
    *,
    baseline_mode: str = "trial_start",
    baseline_window_ms: float = 500.0,
    zscore_within_recording: bool = False,
) -> PupilProcessingResult:
    ...

def extract_pupil_load_features(
    recording: GazeRecording,
    pupil_result: PupilProcessingResult | None = None,
    *,
    window_ms: float = 1000.0,
) -> dict[str, float]:
    ...
```

这里的关键设计是：

- `preprocess_pupil_signal()` 只做 pupil 链路
- `extract_pupil_load_features()` 专门负责认知负荷相关 pupil 特征
- 不强迫所有地方都立刻依赖 `pypillometry`

---

## 5.3 对现有代码的具体改动点

### 5.3.1 `features.py`

改动目标：

- 在现有 `extract_features()` 中接入 pupil 专项特征

建议方式：

- 增加参数：

```python
def extract_features(
    recording: GazeRecording,
    ...,
    include_pupil_load_features: bool = False,
) -> dict[str, float]:
```

- 当 `include_pupil_load_features=True` 且项目环境装了 `pypillometry` 时：
  - 调用 `preprocess_pupil_signal()`
  - 调用 `extract_pupil_load_features()`
  - 把结果 merge 到当前 feature map

建议新增的 pupil 特征包括：

- `pupil_bc_mean`
- `pupil_bc_std`
- `pupil_bc_peak`
- `pupil_bc_q75`
- `pupil_dilation_latency_ms`
- `pupil_tonic_level`
- `pupil_phasic_mean`
- `pupil_phasic_peak`
- `pupil_blink_corrected_variance`

### 5.3.2 `analysis.py`

改动目标：

- 在 `RecordingAnalysis` 中挂出 pupil 分析结果

建议新增字段：

```python
pupil_summary: dict[str, float]
pupil_trace: pd.DataFrame
```

不要一开始就把完整 `PupilProcessingResult` 暴露到 UI 层，先转成更稳定的 summary / trace。

### 5.3.3 `pipeline.py`

改动目标：

- 支持构建用于认知负荷建模的 feature dataset

建议方式：

- `build_feature_dataset()` 增加参数：

```python
include_pupil_load_features: bool = False
```

- 后续可以直接用：

```text
gaze-only features
vs
gaze + pupil-load features
```

做效果对比，这对作品集非常有价值。

### 5.3.4 `dashboard.py`

建议新增一个最小 UI 区块，而不是立刻做大页重构：

- 放在“多模态融合”或“单次会话分析”里
- 先增加一个 `Pupil / Cognitive Load` 小节

第一阶段只展示：

- 清洗前后 pupil 曲线
- baseline corrected pupil 曲线
- pupil load 特征摘要

第二阶段再增加：

- 基于 pupil 的负荷等级预测
- 被试 / 条件 / trial 对比

---

## 5.4 输入、处理、状态、输出链路检查

### 输入

- `GazeRecording.samples`
- 必须包含：
  - `timestamp_ms`
  - `pupil`
- 最好包含：
  - `valid`
  - `marker` / `trial`

### 处理流程

1. 从 `recording.samples["pupil"]` 取 pupil 序列
2. 根据 `valid` 和 blink 段构造缺失/污染 mask
3. 走 `pypillometry` 做清洗和 blink handling
4. 做 baseline correction
5. 提取认知负荷相关 pupil 特征

### 状态变化

- 原始 `GazeRecording` 不直接改写
- pupil 处理结果作为独立对象或 summary 进入分析输出

### 输出

- 特征字典
- trace DataFrame
- 可供 Dashboard 画图的 pupil summary

### 上下游影响

- 上游：要求输入数据里 pupil 列质量更稳定
- 下游：可以直接接 `modeling.py` 做认知负荷分类/回归

---

## 5.5 测试方案

建议新增：

- `tests/test_pupil_preprocess.py`

至少覆盖：

1. 缺失 pupil 时能否优雅退化
2. baseline correction 是否产生合理输出
3. pupil 特征是否进入 `extract_features()`
4. Dashboard 相关 summary 是否可渲染

验收标准：

- 不装 `pypillometry` 时给出清晰提示，不拖垮主程序
- 装了 `pypillometry` 时，feature map 能稳定增加 pupil load 特征

---

## 6. `pymovements` 接入实施方案

## 6.1 接入目标

`pymovements` 不是用来替代当前分析主干，而是补这三件事：

1. 扩展数据源接入
2. 接公开眼动数据集更方便
3. 为后续方法对比提供标准化底层接口

---

## 6.2 最小正确方案

推荐新增：

- `src/gaze_toolkit/pymovements_adapter.py`

不要直接在 `io.py` 里堆大段第三方对象转换逻辑，应该把所有 `pymovements` 相关的依赖隔离到 adapter 里。

### 推荐新增函数

```python
def load_with_pymovements(
    source: str | Path,
    *,
    dataset: str | None = None,
    trial_id: str | int | None = None,
    subject_id: str | int | None = None,
) -> GazeRecording:
    ...

def pymovements_to_recording(obj: Any) -> GazeRecording:
    ...
```

这里的核心任务不是“保留 `pymovements` 原对象”，而是：

- 把它们转换成当前项目统一的 `GazeRecording`

这样后面 `preprocess / events / features / dashboard` 都不用知道 `pymovements` 的内部结构。

---

## 6.3 对现有代码的具体改动点

### 6.3.1 `io.py`

建议新增两种接入方式：

#### 方式 A：注册自定义 loader

利用现有 `register_loader()` 机制：

```python
register_loader("pymovements", load_with_pymovements)
```

这条路径最符合当前项目设计。

#### 方式 B：新增显式入口

```python
def load_dataset(
    dataset_name: str,
    ...,
) -> GazeRecording:
```

如果你后面想直接从公开数据集做作品集演示，这会更友好。

### 6.3.2 `types.py`

这一层尽量不改结构，只补 metadata 约定即可。

建议在 `metadata` 中统一保留：

- `dataset_name`
- `subject_id`
- `trial`
- `stimulus_id`
- `condition`

这样后续建模和统计都更顺。

### 6.3.3 `events.py`

第一阶段不建议强绑定 `pymovements` 事件检测逻辑。

最稳的做法是：

- 先继续用你当前 `detect_events()` 主链路
- 如果 `pymovements` 自带更规范的事件标记或事件表，再把它映射成 `event_label`

也就是说，第一阶段 `pymovements` 的角色主要是：

- 数据读入器
- 数据集访问器

而不是直接替换事件后端。

### 6.3.4 `dashboard.py`

建议只做一个轻量增强：

- 在上传区之外增加“公开数据集示例导入”入口

例如：

- 选择 dataset
- 选择 subject
- 选择 trial
- 一键转为当前 `GazeRecording`

这样非常适合作品集展示，因为你可以现场演示：

“这个平台不只支持我自己的 CSV，也能接标准眼动数据集。”

---

## 6.4 输入、处理、状态、输出链路检查

### 输入

- 文件路径或 `pymovements` dataset identifier

### 处理流程

1. 调用 `pymovements` 读取数据
2. 把字段映射成当前项目标准列：
   - `timestamp_ms`
   - `x`
   - `y`
   - `pupil`
   - `valid`
   - `marker`
   - `trial`
3. 构造成 `GazeRecording`
4. 进入现有 `preprocess -> events -> features -> modeling`

### 状态变化

- 不改变当前领域模型
- 只增加一种新的输入来源

### 输出

- 标准 `GazeRecording`

### 上下游影响

- 上游：扩展数据源
- 下游：完全复用现有分析主干

---

## 6.5 测试方案

建议新增：

- `tests/test_pymovements_adapter.py`

至少覆盖：

1. adapter 是否能输出合法 `GazeRecording`
2. 时间列是否正确转成 `timestamp_ms`
3. 坐标列是否正确映射到 `x/y`
4. metadata 是否保留 `dataset_name/subject_id/trial`

验收标准：

- 不装 `pymovements` 时给出清晰可理解提示
- 装了 `pymovements` 时，接入结果能无缝走通 `analyze_recording()`

---

## 7. 推荐的最小改动清单

如果按“最小完整方案”落地，建议的文件级改动如下：

### 新增文件

- `src/gaze_toolkit/pupil_preprocess.py`
- `src/gaze_toolkit/pymovements_adapter.py`
- `tests/test_pupil_preprocess.py`
- `tests/test_pymovements_adapter.py`

### 修改文件

- `src/gaze_toolkit/features.py`
- `src/gaze_toolkit/analysis.py`
- `src/gaze_toolkit/pipeline.py`
- `src/gaze_toolkit/io.py`
- `src/gaze_toolkit/dashboard.py`
- `pyproject.toml`

---

## 8. 依赖管理建议

不建议把两个包都塞进基础安装。

建议用 extra：

```toml
[project.optional-dependencies]
pupil = ["pypillometry"]
datasets = ["pymovements"]
humanfactors = ["pypillometry", "pymovements"]
```

这样更适合当前项目，也更利于作品集展示时说明：

- 基础版平台可以独立运行
- 高级人因/认知负荷能力按需启用

---

## 9. 实施优先级建议

### 第一优先级

- `pypillometry`

理由：

- 它直接增强认知负荷识别能力
- 更贴近华为人因岗位 JD 中“基于多模态数据提取特征、构建评测模型”的要求

### 第二优先级

- `pymovements`

理由：

- 它增强平台的研究工具属性
- 对作品集可信度和开放数据复现能力很有帮助

---

## 10. 最终建议

如果你希望这两个包的接入既服务项目能力，也服务求职叙事，最合理的落地方式是：

### 对外讲法

- `pypillometry` 负责“认知负荷相关 pupil 信号建模”
- `pymovements` 负责“标准化眼动数据接入与公共数据集复现”

### 对内实现

- `pypillometry` 进入 `features / analysis / dashboard`
- `pymovements` 进入 `io / dataset adapter`

### 不该做的事

- 不要让第三方库对象直接穿透整个项目
- 不要为了接包而推翻当前 `GazeRecording` 主干
- 不要同时大改事件检测、数据读取、建模接口

一句话收敛：

**`pypillometry` 应该作为“认知负荷识别增强层”接入，`pymovements` 应该作为“输入与数据集 adapter 层”接入。**

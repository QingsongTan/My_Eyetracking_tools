# AOI 鼠标绘制需求文档（可落地最小版）

**版本**: v2.0  
**日期**: 2026-03-22  
**状态**: 可开发  
**目标阶段**: V1 最小完整交付  
**前置依赖**: AOI 核心模块已完成（`src/gaze_toolkit/aoi.py`），当前 Dashboard 已有 AOI 手动输入与结果展示能力

---

## 1. 背景与目标

当前 AOI 定义方式依赖手动输入 `x_min / y_min / x_max / y_max`，对演示和研究操作都不够友好。

本期目标不是一次性做完“完整图形编辑器”，而是交付一个**能稳定落地的最小完整版本**：

- 用户可以在刺激图或空白画布上直接**鼠标拖拽绘制矩形 AOI**
- 系统可以把画布坐标映射为 `screen_size` 坐标
- 绘制结果可以进入现有 AOI 分析链路，自动刷新指标表和转移矩阵
- 手动输入模式继续保留，不被破坏

一句话定义本期范围：

> 在现有 Streamlit Dashboard 中增加“矩形 AOI 鼠标绘制 + 显式应用到分析”的能力，不承诺本期支持多边形、画布内编辑闭环、双向同步。

---

## 2. 本期结论

### 2.1 推荐实现

**V1 采用 `streamlit-drawable-canvas` 作为可选增强组件，但只交付“矩形绘制”路径。**

推荐理由：

- 对当前目标来说，矩形绘制已经足够覆盖大多数 AOI 演示场景
- 现有 `aoi.py` 已支持矩形 AOI，接入成本最低
- 当前 Dashboard 已有 AOI 分析结果区，新增能力只需要把“画布输出”转换为 `AOI` 列表
- 相比多边形、变换编辑、画布与手动双向同步，矩形版本更容易在一次开发内做闭合

### 2.2 本期不采用的能力

以下能力不进入本期交付：

- 多边形绘制与解析
- 画布内移动、缩放、旋转、单个删除
- 现有 AOI 自动回填到画布进行继续编辑
- 手动模式和画布模式之间的双向同步
- 与 Plotly scanpath 图完全共享同一图层

原因不是这些能力没价值，而是它们会显著增加状态同步和图形解析复杂度，不符合“最小完整且稳”的目标。

### 2.3 依赖风险的明确结论

`streamlit-drawable-canvas` 上游已归档，不能按“长期稳定基础设施”来描述。

因此本期要求是：

- 将其视为 **dashboard optional dependency**
- 仅在导入成功时启用鼠标绘制功能
- 导入失败或运行时不兼容时，**只保留手动输入模式，不做复杂降级**

这意味着本期交付的是：

> “有条件启用的增强交互”，不是“强依赖、全环境保证可用”的基础能力。

---

## 3. 范围与非目标

### 3.1 In Scope

- AOI 区块新增模式切换：`手动输入` / `鼠标绘制`
- 鼠标绘制模式下展示矩形画布
- 画布支持刺激图背景；无刺激图时展示空白背景
- 用户绘制一个或多个矩形后，点击按钮将当前画布结果**覆盖**为当前 AOI 列表
- 生成的 AOI 自动命名为 `AOI 1 / AOI 2 / ...`
- 用户可以在应用后对 AOI 名称做文本重命名
- AOI 分析结果区继续复用现有 `assign_fixations_to_aoi` / `compute_aoi_metrics` / `compute_transition_matrix`
- 保留“示例 AOI”“清空 AOI”“手动输入 AOI”现有能力

### 3.2 Out of Scope

- 多边形 AOI
- 自由绘制 / 折线 / 圆形
- 选中已有 AOI 后在画布内继续编辑
- 画布草稿与结果区 AOI 的实时双向绑定
- 针对移动端或窄屏做特殊适配

---

## 4. 假设与未验证前提

### 4.1 当前假设

- Dashboard 的主要使用场景是桌面浏览器下的演示和研究操作
- `screen_size` 仍然是 AOI 与 scanpath 的统一坐标系
- 鼠标绘制功能主要服务于“创建矩形 AOI”，不是完整标注平台

### 4.2 未验证前提

- `streamlit-drawable-canvas` 在本项目当前 Streamlit 版本下能稳定导入并运行
- 组件在当前环境下显示背景图时不会出现不可接受的兼容问题

这两点在文档阶段无法证明，必须在开发开始时先做一次本地验证。  
如果验证失败，本期功能不应继续扩展实现，而应保留手动模式并单独记录兼容性问题。

---

## 5. 输入 / 处理 / 状态 / 输出链路

### 5.1 输入

- `recording`: 当前单次会话记录，且已具备 fixation 事件
- `stimulus_image`: 可选，来自上传图片
- `screen_size`: 当前刺激图坐标系
- `canvas_result.json_data`: 画布输出的原始矩形列表

### 5.2 处理流程

1. 用户进入 AOI 分析区块
2. 选择 `鼠标绘制` 模式
3. 在画布上绘制一个或多个矩形
4. 点击“用画布覆盖当前 AOI”按钮
5. 系统把画布矩形坐标按比例映射为 `screen_size` 坐标
6. 生成 `AOI` dataclass 列表并写入 AOI 分析状态
7. 右侧 AOI 指标表和转移矩阵基于新的 AOI 列表重新计算

### 5.3 状态变化

本期必须区分两类状态：

- **画布草稿状态**：只是当前 canvas 的原始 JSON，不直接驱动分析
- **分析 AOI 状态**：真正用于指标计算的 `st.session_state[AOI_STATE_KEY]`

关键规则：

- 切换到鼠标绘制模式时，**不自动覆盖**现有 AOI
- 只有用户点击“用画布覆盖当前 AOI”后，才更新 `AOI_STATE_KEY`
- 切换模式本身不清空 `AOI_STATE_KEY`
- “清空 AOI”按钮只清空分析 AOI 状态；如需清空画布，使用画布自身工具栏或单独的“清空画布”按钮

### 5.4 输出

- AOI 定义表
- AOI 叠加 scanpath 图
- AOI 指标汇总表
- AOI 转移矩阵热力图

### 5.5 上下游影响

- 上游：不修改 `recording`、事件识别、fixation 生成逻辑
- 下游：继续复用现有 AOI 分析流程，不改 `aoi.py` 核心计算

---

## 6. 交互设计

### 6.1 模式切换

AOI 区块新增：

```python
st.radio("AOI 定义方式", ["手动输入", "鼠标绘制"])
```

两种模式的职责：

- `手动输入`：保持当前实现
- `鼠标绘制`：只负责从画布生成矩形 AOI

### 6.2 鼠标绘制模式交互

```
用户进入 AOI 分析区块
  ├─ 选择“鼠标绘制”
  ├─ 系统显示画布（背景为刺激图或空白背景）
  ├─ 用户拖拽绘制 1~N 个矩形
  ├─ 点击“用画布覆盖当前 AOI”
  ├─ 系统解析矩形 -> AOI 列表
  ├─ AOI 列表进入现有分析状态
  ├─ 用户可在下方重命名 AOI
  └─ 右侧结果区自动刷新
```

### 6.3 本期不承诺的交互

以下行为不得写入本期验收标准：

- 在画布中选中 AOI 后直接 Delete 删除
- 在画布中移动 / 缩放后结果与分析状态始终实时同步
- 手动模式新增的 AOI 自动回显到 canvas
- 示例按钮生成的 AOI 自动回显到 canvas

这些都是后续迭代项，不属于本期最小闭环。

---

## 7. 技术方案

### 7.1 方案选择

本期采用：

**`streamlit-drawable-canvas` + 画布结果显式应用到现有 AOI 状态**

不采用 Plotly drawing 作为本期主方案，原因如下：

- 当前项目中 Plotly 主要用于展示，未验证 shape drawing 事件能在现有 Streamlit 组合下稳定回传
- 本期目标是“尽快稳态落地鼠标绘制”，不增加新的前端事件链调试成本

### 7.2 依赖管理

`pyproject.toml`：

```toml
[project.optional-dependencies]
dashboard = [
  "plotly>=5.20",
  "streamlit>=1.33",
  "streamlit-drawable-canvas==0.9.3",
]
```

说明：

- 使用固定版本，避免开放式 `>=0.9` 带来的不确定性
- 该组件只进入 `dashboard` extra，不进入核心依赖

导入保护建议：

```python
try:
    from streamlit_drawable_canvas import st_canvas
    _HAS_AOI_CANVAS = True
except Exception:
    st_canvas = None
    _HAS_AOI_CANVAS = False
```

这里使用 `except Exception`，不是只捕获 `ImportError`，因为组件可能在导入阶段就因上游 API 变更而报错。

### 7.3 画布尺寸策略

本期不做“自动撑满列宽”。

固定策略：

```python
canvas_width = 640
canvas_height = round(canvas_width * screen_size[1] / screen_size[0])
```

理由：

- 减少自适应尺寸带来的背景图和坐标映射不稳定
- 演示场景下固定宽度更容易复现和测试

### 7.4 画布配置

```python
canvas_result = st_canvas(
    fill_color="rgba(0, 243, 255, 0.14)",
    stroke_width=2,
    stroke_color="rgba(0, 243, 255, 0.90)",
    background_image=canvas_background_pil,
    background_color="#0A1628" if canvas_background_pil is None else "",
    width=canvas_width,
    height=canvas_height,
    drawing_mode="rect",
    display_toolbar=True,
    update_streamlit=True,
    initial_drawing=st.session_state.get(AOI_CANVAS_DRAFT_KEY),
    key="aoi-canvas-v1",
)
```

本期只允许：

- `drawing_mode="rect"`

不在 UI 中暴露：

- `polygon`
- `transform`
- `freedraw`

### 7.5 背景图处理

当前项目中的 `stimulus_image` 可能是 `UploadedFile`、`Path` 或其他 file-like 对象。  
而 `st_canvas` 需要的是 `PIL.Image`。

因此需要新增辅助函数：

```python
def _load_canvas_background_image(stimulus_image: str | Path | Any | None) -> Image.Image | None:
    """将 Dashboard 当前刺激图输入转换为 st_canvas 所需的 PIL.Image。"""
```

处理规则：

- `stimulus_image is None` -> 返回 `None`
- `Path` / `str` -> `Image.open(...)`
- file-like 对象 -> 先 `seek(0)` 再 `Image.open(...)`
- 最终统一 `convert("RGBA")`

### 7.6 坐标映射

画布坐标到 `screen_size` 坐标的映射规则：

```python
scale_x = screen_size[0] / canvas_width
scale_y = screen_size[1] / canvas_height

x_min = obj["left"] * scale_x
y_min = obj["top"] * scale_y
x_max = (obj["left"] + obj["width"] * obj.get("scaleX", 1.0)) * scale_x
y_max = (obj["top"] + obj["height"] * obj.get("scaleY", 1.0)) * scale_y
```

本期只解析 `type == "rect"`。

### 7.7 解析策略

新增辅助函数：

```python
def _parse_canvas_rectangles_to_aois(
    json_data: dict | None,
    canvas_width: int,
    canvas_height: int,
    screen_size: tuple[int, int],
) -> list[AOI]:
    """将 drawable canvas 的矩形对象解析为 AOI 列表。"""
```

解析规则：

- `json_data is None` -> 返回空列表
- `objects` 为空 -> 返回空列表
- 只处理 `type == "rect"`
- 忽略宽或高小于 `8px` 的矩形，视为误触
- 如果对象存在 `angle != 0`，本期直接忽略并提示“暂不支持旋转矩形”
- 结果统一调用 `define_aoi(...)`

### 7.8 命名与编辑

本期命名规则分两段：

1. **应用时自动命名**
   - `AOI 1`, `AOI 2`, ...
2. **应用后可重命名**
   - 在 AOI 定义表下方用 `st.text_input` 列表编辑名称

关键约束：

- 名称编辑针对的是 `AOI_STATE_KEY` 中的已应用 AOI
- 如果用户重新点击“用画布覆盖当前 AOI”，则当前 AOI 列表被新结果整体替换，名称会重新生成

这条约束必须明确写给用户，避免产生“画布更新后名称为什么没保留”的歧义。

### 7.9 示例 AOI 与手动模式的关系

本期规定：

- 示例按钮仍然保留，继续直接写入 `AOI_STATE_KEY`
- 手动模式仍然继续直接写入 `AOI_STATE_KEY`
- 这些 AOI **不要求**自动同步进画布

也就是说：

- 画布是一个“新建矩形 AOI 的独立输入器”
- 结果区是统一的 AOI 分析出口

---

## 8. 状态设计

建议新增状态键：

```python
AOI_STATE_KEY = "dashboard_aois"              # 已应用到分析的 AOI 列表
AOI_CANVAS_DRAFT_KEY = "dashboard_aoi_canvas" # 当前画布草稿 JSON
```

状态规则：

- `AOI_STATE_KEY` 是唯一的分析真值
- `AOI_CANVAS_DRAFT_KEY` 仅用于画布重绘和保留草稿
- 任何结果图和结果表只读 `AOI_STATE_KEY`

---

## 9. 文件变更清单

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `pyproject.toml` | 修改 | `dashboard` extra 增加 `streamlit-drawable-canvas==0.9.3` |
| `src/gaze_toolkit/dashboard.py` | 修改 | AOI 区块新增鼠标绘制模式、画布草稿状态、矩形解析与应用逻辑 |
| `src/gaze_toolkit/aoi.py` | 不变 | 复用现有矩形 AOI 定义与指标计算 |
| `tests/test_dashboard_helpers.py` | 修改 | 为画布解析与状态辅助逻辑补测试 |

---

## 10. `dashboard.py` 具体实现要求

### 10.1 `_render_aoi_section()` 结构调整

在左侧 AOI 定义区增加模式切换：

1. `手动输入`
   - 保持现有逻辑
2. `鼠标绘制`
   - 若 `_HAS_AOI_CANVAS == False`，显示 warning 并提示继续使用手动输入
   - 若 `_HAS_AOI_CANVAS == True`，展示画布和“用画布覆盖当前 AOI”按钮

### 10.2 鼠标绘制模式的最小行为

- 展示固定宽度矩形画布
- 画布下方显示说明：
  - 仅支持矩形
  - 绘制后需点击“用画布覆盖当前 AOI”才会进入分析
- 点击应用按钮后：
  - 解析草稿
  - 生成 AOI 列表
  - 覆盖 `st.session_state[AOI_STATE_KEY]`

### 10.3 新增辅助函数

```python
def _load_canvas_background_image(stimulus_image: str | Path | Any | None) -> Image.Image | None:
    ...

def _parse_canvas_rectangles_to_aois(
    json_data: dict | None,
    canvas_width: int,
    canvas_height: int,
    screen_size: tuple[int, int],
) -> list[AOI]:
    ...
```

如需减少 `_render_aoi_section()` 复杂度，可以再拆一个：

```python
def _apply_canvas_aois(
    json_data: dict | None,
    *,
    canvas_width: int,
    canvas_height: int,
    screen_size: tuple[int, int],
) -> list[AOI]:
    ...
```

---

## 11. 边界情况处理

| 场景 | 处理方式 |
|------|---------|
| `streamlit-drawable-canvas` 导入失败 | 鼠标绘制模式显示 warning，不进入复杂 fallback，用户继续用手动输入 |
| 无刺激图 | 使用纯色背景画布 |
| 画布为空 | 点击应用时提示“请先绘制至少一个矩形 AOI” |
| 矩形过小 | 忽略该对象 |
| 画布对象不是矩形 | 忽略 |
| 旋转矩形 | 忽略并提示本期不支持 |
| 切换模式 | 不自动修改 `AOI_STATE_KEY` |
| 再次应用画布 | 直接覆盖当前 AOI 列表 |

---

## 12. 验收标准

- [ ] 手动输入模式的现有行为不变
- [ ] 鼠标绘制模式可在画布上拖拽创建矩形
- [ ] 点击“用画布覆盖当前 AOI”后，矩形能正确映射到 `screen_size`
- [ ] 应用后的 AOI 能驱动右侧指标表和转移矩阵刷新
- [ ] 有刺激图时，画布背景正确显示刺激图
- [ ] 无刺激图时，画布仍可工作
- [ ] 已应用 AOI 可重命名
- [ ] “清空 AOI”按钮正常工作
- [ ] 组件不可用时页面不崩溃，仍可手动输入 AOI
- [ ] 新增的 Dashboard 辅助逻辑有单元测试覆盖
- [ ] `pytest tests/test_aoi.py tests/test_dashboard_helpers.py` 通过

---

## 13. 延后项

以下内容明确延后，不得在本期实现过程中顺手扩展：

- 多边形 AOI
- 画布内选择后拖动 / 缩放 / 删除
- 示例 AOI 回显到 canvas
- 画布与 AOI 结果表实时联动
- Plotly 同层绘制 AOI

后续如果需要做 V2，应单独立项。

---

## 14. 工作量估算

| 子任务 | 估算 |
|--------|------|
| 依赖接入与导入保护 | 0.5h |
| AOI 区块模式切换与画布 UI | 1.5h |
| 背景图转换与矩形解析 | 1.5h |
| AOI 应用 / 重命名 / 清空链路联调 | 1h |
| 测试补充 | 1h |
| **合计** | **4.5 - 5.5h** |

---

## 15. 最终交付定义

本期交付完成的标准不是“能画各种 AOI”，而是：

> 用户可以在 Dashboard 中通过鼠标拖拽创建矩形 AOI，并把结果稳定送入现有 AOI 分析流程，且不破坏手动输入模式。

只要这条主链路闭合，本期就算完成。

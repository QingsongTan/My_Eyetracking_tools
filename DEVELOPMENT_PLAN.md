# 平台开发计划

**定位**：面向移动/穿戴产品的多模态认知状态感知与意图预测平台

**核心分析链**：
```
产品界面截图
    ↓ [基线层]  物理显著性 + DeepGaze → 注意力基线地图
    ↓ [采集层]  眼动信号 + 心率等生理数据
    ↓ [特征层]  注视/扫视/瞳孔特征 + 注意力偏差特征
    ↓ [AI 层]   认知负荷估计 + 意图预测
    ↓ [洞察层]  认知状态解读 + UX 改进建议
```

---

## P0 · LLM 特征异常解读（RFC-001）

**目标**：让 LLM 从特征向量中推断认知状态，替代当前「仅写散文」角色

**估时**：2 天

### U1 · 数据结构定义
- 文件：`report_generator.py`
- 新增 `FeatureAnomaly` dataclass：`feature / value / direction / severity / threshold_ref`
- 新增 `FeatureAnomalyExplanation` dataclass：`anomalies / cognitive_state_hypothesis / ux_recommendations / explanation_mode / model_used`

### U2 · 规则引擎异常检测
- 文件：`report_generator.py`
- 新增 `_detect_feature_anomalies(features) -> list[FeatureAnomaly]`
- 扩展 `_THRESHOLDS`，覆盖 8 个核心特征，支持 warning / critical 双级
- 验收：全正常输入 → 空列表；`fixation_duration_mean=50` → critical_low

### U3 · LLM Prompt 构建器
- 文件：`report_generator.py`
- 新增 `_build_anomaly_prompt(features, anomalies, context) -> str`
- 结构化指令：要求 LLM 输出合规 JSON，包含 `cognitive_state` + `ux_recommendations[]`

### U4 · 公开函数 `explain_feature_anomalies()`
- 文件：`report_generator.py`
- 签名：`explain_feature_anomalies(features, *, context, use_llm, llm_backend, llm_model, api_key) -> FeatureAnomalyExplanation`
- 降级链：LLM 失败 → 规则引擎 → 空结果，全程不抛异常
- 验收：`use_llm=False` 不触发任何网络请求

### U5 · 接入报告生成器
- 文件：`report_generator.py`
- `generate_insight_report()` 新增 `explain_anomalies: bool = False`
- 默认 False，零破坏现有调用
- 新增 section_type `"anomaly_explanation"`，插入 features 摘要之后

### U6 · Dashboard 面板
- 文件：`dashboard.py`
- 单会话分析 Tab 新增「特征异常解读」展开区
- 内容：异常特征彩色表格（warning=黄/critical=红）+ 认知状态假说文本 + UX 建议列表

### U7 · 测试套件
- 文件：`tests/test_anomaly_explanation.py`（新建）
- 用例：全正常特征/单一异常/`use_llm=False` 不触网络/报告 section 数量不变/新 section 存在/LLM 失败降级

---

## P1 · 真实数据集验证

**目标**：用公开真实数据跑通认知负荷预测，得到可报告的准确率数字

**估时**：1-2 天

### Step 1 · 数据接入
- 使用 `pymovements` 内置公开数据集（GazeBase / ToyDataset）
- 无需手动下载，直接 API 获取

### Step 2 · Pipeline 复用
- 复用 `pipeline.py → modeling.py → cognitive_load.py` 现有链路
- 仅替换数据源，不改动业务逻辑

### Step 3 · 结果产出
- 文件：`notebooks/07-真实数据集认知负荷预测.ipynb`（新建）
- 输出：准确率、F1、混淆矩阵、特征重要性排名
- 目标：面试时能说「模型准确率 XX%，基于 N 名被试真实眼动数据」

---

## P2 · 叙事与界面对齐

**目标**：README 和 Dashboard 视觉语言与新定位一致

**估时**：半天

### Step 1 · README 重写
- 结构：「平台定位 → 分析链图示 → 功能模块 → 快速开始」
- 首段直接点明定位，不以功能列表开头
- 显著性功能归入「基线层」章节，不再列为「高级功能」

### Step 2 · Dashboard 首页 Overview
- 把当前功能列表替换为「信号 → 意图 → 建议」流程说明文字
- 说明三层可视化的含义（基线/预测/实际）

### Step 3 · 三层可视化 Toggle
- 文件：`dashboard.py` 显著性模块
- 三个独立 checkbox：`物理显著性` / `DeepGaze 预测注视` / `真实眼动热力图`
- 可单独显示任意层，也可叠加全部三层

---

## P3 · 视觉大模型热力图自动解读

**目标**：上传界面截图 + 热力图，调用视觉大模型自动识别注意力模式，输出 UX 建议

**估时**：2-3 天

### Step 1 · 接口设计
- 文件：新建 `vision_analyst.py`
- 核心函数：
```python
def analyze_attention_heatmap(
    interface_image: PIL.Image,
    heatmap_image: PIL.Image,
    saliency_image: PIL.Image | None = None,
    *,
    api_key: str | None = None,
    backend: str = "openai",   # "openai" | "anthropic"
) -> AttentionAnalysis
```
- 输出结构 `AttentionAnalysis`：`focus_regions / ignored_regions / saliency_deviation / ux_recommendations / raw_response`

### Step 2 · 图像构建
- 将界面截图、热力图、显著性图拼接或分别传入
- 对 OpenAI：base64 编码后传入 vision 消息
- 对 Anthropic：使用 Claude Vision 多图输入

### Step 3 · Prompt 设计
系统指令：「你是人因工程研究员，分析用户在产品界面上的真实注意力分布」

用户指令结构：
1. 这是产品界面截图
2. 这是用户真实眼动热力图（红色=高注意力）
3. （可选）这是 AI 预测的显著性图
4. 请识别：主要关注区域 / 被忽略的关键区域 / 与显著性预测的偏差 / 具体 UX 改进建议

### Step 4 · Dashboard 接入
- 文件：`dashboard.py` 可视化模块
- 在三层可视化区域下方新增「生成 AI 解读」按钮
- 输出面板：关注区域标注 + 偏差说明 + UX 建议列表
- 无 API Key 时按钮置灰并提示

### Step 5 · 测试
- 文件：`tests/test_vision_analyst.py`（新建）
- Mock Vision API，验证图像编码、prompt 构建、输出解析

---

## 依赖关系

```
P0 ──────────────── 无外部依赖，立即可开始
P1 ──────────────── pymovements 已在依赖中，立即可开始（与 P0 并行）
P2 ── 依赖 P0 ────── 叙事需与新功能对齐后再写
P3 ── 依赖 P2 ────── 需要 Vision API Key；Dashboard 叙事先对齐
```

## 风险登记

| 风险 | 等级 | 缓解方案 |
|---|---|---|
| LLM 输出非合规 JSON（P0） | 中 | 解析失败自动降级规则引擎 |
| GazeBase 无认知负荷标签（P1） | 中 | 改用 pymovements ToyDataset 或 SEED 公开数据集 |
| Vision API 图像理解精度不稳定（P3） | 中 | Prompt 迭代；加结构化输出约束 |
| `dashboard.py` Toggle 渲染冲突（P2） | 低 | 独立 st.session_state key |
| 报告生成器签名变更（P0） | 低 | 新参数全部默认 False，零破坏 |

## 时间估算

| 阶段 | 估时 | 并行可能性 |
|---|---|---|
| P0 | 2 天 | — |
| P1 | 1-2 天 | 可与 P0 并行 |
| P2 | 0.5 天 | P0 完成后 |
| P3 | 2-3 天 | P2 完成后 |
| **合计** | **5.5-7.5 天** | — |

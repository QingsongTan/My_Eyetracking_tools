# 多模态人因分析平台 — 眼动分析子系统

## 开发架构与需求文档

**版本**: v1.0
**日期**: 2026-03-21
**作者**: [姓名]
**适用产品线**: 手机 / 耳机 / 穿戴设备 / 平板PC

---

## 1. 项目背景与目标

### 1.1 背景

在消费电子产品（手机、耳机、穿戴设备、平板PC）的体验设计中，用户的**视觉注意力分配**、**认知负荷**和**交互意图**是衡量产品可用性和体验质量的核心指标。传统的人因研究依赖人工标注和主观量表，效率低、信度有限。

本项目旨在构建一套**AI驱动的多模态人因分析平台**，以眼动数据为核心，融合生理信号（GSR/HR/EEG）、行为日志（触控/手势/语音）等多通道数据，实现：

- 自动化的用户体验评测
- 实时认知状态与交互意图预测
- 科学智能的体验设计工程方法闭环

### 1.2 核心目标

| 目标维度 | 具体目标 | 关键指标 |
|---------|---------|---------|
| **研究效率** | 眼动数据处理与特征提取全自动化 | 分析耗时降低 70%+ |
| **信度提升** | AI 辅助消除人工标注偏差 | 注视区域识别准确率 ≥ 95% |
| **效度增强** | 多模态融合交叉验证 | 认知负荷预测 R² ≥ 0.85 |
| **产品赋能** | 直接输出设计改进建议 | 每轮评测产出可执行洞察 ≥ 5 条 |

---

## 2. 人因研究框架

### 2.1 感觉通道-变量-指标映射

```
┌─────────────────────────────────────────────────────────────────┐
│                    人因变量拆解框架                               │
├──────────┬──────────────┬──────────────┬────────────────────────┤
│ 感觉通道  │   人因变量     │   采集指标    │     分析维度           │
├──────────┼──────────────┼──────────────┼────────────────────────┤
│          │ 视觉搜索效率   │ 注视时长      │ 信息层级可发现性        │
│  视觉     │ 认知负荷      │ 瞳孔直径      │ 界面复杂度评估          │
│ (Vision) │ 注意力分配    │ 扫视路径      │ 布局合理性              │
│          │ 信息加工深度   │ 回视频率      │ 内容可读性              │
│          │ 视觉疲劳      │ 眨眼频率/时长  │ 长时使用舒适度          │
├──────────┼──────────────┼──────────────┼────────────────────────┤
│          │ 听觉舒适度    │ 主观评分+GSR   │ 音频参数优化            │
│  听觉     │ 语音识别效率   │ 任务完成率    │ 语音交互设计            │
│ (Audio)  │ 声音空间感    │ 头动追踪      │ 空间音频调优            │
│          │ 听觉疲劳      │ 反应时变化    │ 耳机佩戴体验            │
├──────────┼──────────────┼──────────────┼────────────────────────┤
│          │ 触觉反馈感知   │ 触控压力/面积  │ 振动马达参数优化         │
│  触觉     │ 操作精准度    │ 触控偏移量    │ 交互热区设计            │
│ (Touch)  │ 手势自然度    │ 手势轨迹      │ 手势交互映射            │
│          │ 佩戴舒适度    │ 压力传感      │ 穿戴设备人体工学         │
└──────────┴──────────────┴──────────────┴────────────────────────┘
```

### 2.2 产品-通道-场景矩阵

| 产品 | 主通道 | 辅通道 | 典型评测场景 |
|------|-------|-------|------------|
| **手机** | 视觉 + 触觉 | 听觉 | UI 导航效率、单手操作体验、阅读舒适度 |
| **耳机** | 听觉 | 触觉 | 空间音频感知、降噪效果、触控手势操作 |
| **穿戴** | 触觉 + 视觉 | — | 表盘信息获取速度、运动场景交互、佩戴压力 |
| **平板PC** | 视觉 + 触觉 | — | 多任务分屏注意力、手写笔交互、长时阅读 |

---

## 3. 系统架构

### 3.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         多模态人因分析平台                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  │
│  │  Tobii Pro    │  │  Shimmer3    │  │  触控日志     │  │ 行为录屏    │  │
│  │  眼动仪       │  │  生理采集     │  │  埋点SDK      │  │ 采集SDK    │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └─────┬──────┘  │
│         │                  │                  │                │         │
│  ───────┴──────────────────┴──────────────────┴────────────────┴──────   │
│                        数据采集与同步层                                   │
│                   (LSL / 时间戳对齐 / 数据清洗)                           │
│  ────────────────────────────┬───────────────────────────────────────    │
│                              │                                          │
│  ┌───────────────────────────┴───────────────────────────────────────┐  │
│  │                      数据处理与存储层                               │  │
│  │  ┌─────────────┐  ┌──────────────┐  ┌──────────────────────────┐ │  │
│  │  │ Raw Data    │  │ Feature      │  │ Processed Data           │ │  │
│  │  │ Storage     │  │ Extraction   │  │ (PostgreSQL + TimescaleDB│ │  │
│  │  │ (MinIO/S3)  │  │ Pipeline     │  │  + Redis Cache)          │ │  │
│  │  └─────────────┘  └──────────────┘  └──────────────────────────┘ │  │
│  └───────────────────────────┬───────────────────────────────────────┘  │
│                              │                                          │
│  ┌───────────────────────────┴───────────────────────────────────────┐  │
│  │                       AI 分析引擎层                                │  │
│  │                                                                   │  │
│  │  ┌────────────────┐  ┌────────────────┐  ┌─────────────────────┐ │  │
│  │  │ 注视事件检测    │  │ AOI 自动分割    │  │ 认知状态预测模型    │ │  │
│  │  │ (I-VT/I-DT     │  │ (SAM/YOLO      │  │ (Transformer-based │ │  │
│  │  │  + CNN 优化)    │  │  语义分割)      │  │  多模态融合)       │ │  │
│  │  └────────────────┘  └────────────────┘  └─────────────────────┘ │  │
│  │                                                                   │  │
│  │  ┌────────────────┐  ┌────────────────┐  ┌─────────────────────┐ │  │
│  │  │ 扫视路径分析    │  │ 视觉疲劳评估    │  │ 交互意图预测        │ │  │
│  │  │ (HMM/序列模型)  │  │ (PERCLOS +     │  │ (序列决策模型       │ │  │
│  │  │               │  │  眨眼特征)      │  │  + LLM 推理)       │ │  │
│  │  └────────────────┘  └────────────────┘  └─────────────────────┘ │  │
│  └───────────────────────────┬───────────────────────────────────────┘  │
│                              │                                          │
│  ┌───────────────────────────┴───────────────────────────────────────┐  │
│  │                     可视化与洞察输出层                              │  │
│  │  ┌──────────┐  ┌───────────┐  ┌────────────┐  ┌──────────────┐  │  │
│  │  │ 热力图    │  │ 扫视路径图 │  │ 实时仪表盘  │  │ 自动报告生成  │  │  │
│  │  │ 渲染引擎  │  │ 可视化    │  │ (Grafana)  │  │ (LLM 驱动)  │  │  │
│  │  └──────────┘  └───────────┘  └────────────┘  └──────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 技术栈选型

| 层级 | 技术选型 | 选型理由 |
|-----|---------|---------|
| **数据采集** | Tobii Pro SDK (Python) + LSL | 工业级眼动采集标准，亚毫秒级同步 |
| **数据处理** | Python (pandas/numpy/scipy) | 人因研究主流工具链，生态成熟 |
| **特征工程** | scikit-learn + tsfresh | 自动化时序特征提取 |
| **深度学习** | PyTorch + HuggingFace | 多模态模型灵活性，预训练模型丰富 |
| **大模型推理** | Claude API / 本地部署 Qwen | 报告生成与洞察推理 |
| **数据库** | PostgreSQL + TimescaleDB | 时序数据高效存储与查询 |
| **可视化** | Plotly + Dash / Streamlit | 快速构建交互式分析面板 |
| **统计分析** | R (lme4/emmeans) + Python (statsmodels) | 混合效应模型，人因实验统计标准 |
| **版本管理** | Git + DVC | 代码 + 数据版本同步管理 |

---

## 4. 核心模块详细设计

### 4.1 模块一：眼动数据采集与预处理

#### 4.1.1 数据采集协议

```python
# 采集配置示例
COLLECTION_CONFIG = {
    "eye_tracker": {
        "device": "Tobii Pro Spark / Glasses 3",
        "sampling_rate": 120,  # Hz (移动端场景) / 300 Hz (实验室场景)
        "calibration": "5-point",
        "validation_threshold": 0.5,  # 度视角
    },
    "synchronization": {
        "protocol": "LSL (Lab Streaming Layer)",
        "clock_offset_correction": True,
        "marker_events": ["task_start", "task_end", "stimulus_onset", "response"],
    },
    "recording_metadata": {
        "participant_id": "auto_generated",
        "device_under_test": ["phone", "earphone", "wearable", "tablet"],
        "scenario": "...",
        "environment": "controlled_lab / field_study",
    }
}
```

#### 4.1.2 预处理流水线

```
原始数据 (gaze_x, gaze_y, pupil_diameter, timestamp)
    │
    ├─→ [1] 数据质量检查
    │       • 追踪率 (tracking ratio) ≥ 80%
    │       • 插值填补短缺失 (< 75ms, 线性插值)
    │       • 标记长缺失段 (> 200ms, 眨眼/离屏)
    │
    ├─→ [2] 噪声滤波
    │       • Savitzky-Golay 滤波 (窗口=5, 阶数=3)
    │       • 瞳孔直径: 带通滤波 + 基于MAD的离群值剔除
    │
    ├─→ [3] 事件检测
    │       • 注视 (Fixation): I-DT 算法 (阈值=1°, 最小时长=100ms)
    │       • 扫视 (Saccade): 速度阈值法 (>30°/s)
    │       • 眨眼 (Blink): 瞳孔信号缺失 + 时长判断
    │       • 微扫视 (Microsaccade): Engbert & Kliegl 算法
    │
    └─→ [4] AOI 映射
            • 基于 UI 元素树自动生成 AOI
            • SAM (Segment Anything Model) 辅助动态场景分割
            • 注视点 → AOI 归属判定
```

#### 4.1.3 核心预处理代码结构

```python
# eye_tracking/preprocessing/pipeline.py

class EyeTrackingPipeline:
    """眼动数据预处理全流程管线"""

    def __init__(self, config: CollectionConfig):
        self.config = config
        self.quality_checker = DataQualityChecker(min_tracking_ratio=0.8)
        self.filter = GazeFilter(method="savgol", window=5, order=3)
        self.event_detector = EventDetector(
            fixation_algo="idt",
            dispersion_threshold=1.0,  # degrees
            min_duration=100,           # ms
            saccade_velocity_threshold=30,  # deg/s
        )
        self.aoi_mapper = AOIMapper()

    def process(self, raw_data: pd.DataFrame) -> ProcessedEyeData:
        # Step 1: 质量校验
        quality_report = self.quality_checker.validate(raw_data)
        if not quality_report.is_acceptable:
            raise DataQualityError(quality_report)

        # Step 2: 插值 + 滤波
        interpolated = self._interpolate_gaps(raw_data, max_gap_ms=75)
        filtered = self.filter.apply(interpolated)

        # Step 3: 事件检测
        fixations = self.event_detector.detect_fixations(filtered)
        saccades = self.event_detector.detect_saccades(filtered)
        blinks = self.event_detector.detect_blinks(filtered)

        # Step 4: AOI 映射
        aoi_assignments = self.aoi_mapper.assign(fixations, self.config.aoi_definition)

        return ProcessedEyeData(
            raw=raw_data,
            filtered=filtered,
            fixations=fixations,
            saccades=saccades,
            blinks=blinks,
            aoi_assignments=aoi_assignments,
            quality_report=quality_report,
        )
```

### 4.2 模块二：AI 驱动的特征提取与分析

#### 4.2.1 眼动特征体系

```
眼动特征 (3 层级, 40+ 指标)
│
├── 基础指标层 (Basic Metrics)
│   ├── 注视: 总注视次数, 平均注视时长, 首次注视时长, 注视频率
│   ├── 扫视: 扫视幅度, 扫视速度, 扫视方向分布
│   ├── 瞳孔: 平均瞳孔直径, 瞳孔直径变化率 (TEPR)
│   └── 眨眼: 眨眼频率, 眨眼时长, PERCLOS
│
├── AOI 指标层 (AOI-based Metrics)
│   ├── 时间维度: 首次注视到达时间, AOI 总驻留时间, AOI 注视占比
│   ├── 频次维度: AOI 访问次数, AOI 回视次数, AOI 转移概率矩阵
│   └── 序列维度: AOI 访问序列, 转移熵, 序列相似度 (Levenshtein)
│
└── 高阶认知指标层 (Cognitive Metrics)
    ├── 视觉搜索效率: 搜索时间, 搜索路径最优比
    ├── 认知负荷: ICA (Index of Cognitive Activity), 瞳孔散度
    ├── 注意力分散度: 凝视熵 (Gaze Entropy), 注意力热区集中度
    └── 信息加工: 首次通过注视时长, 二次通过注视时长, 回视概率
```

#### 4.2.2 自动特征提取引擎

```python
# eye_tracking/features/extractor.py

class AIFeatureExtractor:
    """基于机器学习的自动特征提取引擎"""

    def __init__(self):
        self.basic_extractor = BasicMetricsExtractor()
        self.aoi_extractor = AOIMetricsExtractor()
        self.cognitive_extractor = CognitiveMetricsExtractor()
        self.ts_extractor = TimeSeriesFeatureExtractor()  # tsfresh wrapper

    def extract_all(self, processed_data: ProcessedEyeData) -> FeatureMatrix:
        """全量特征提取"""
        features = {}

        # 基础指标
        features.update(self.basic_extractor.extract(processed_data))

        # AOI 指标
        features.update(self.aoi_extractor.extract(processed_data))

        # 认知指标
        features.update(self.cognitive_extractor.extract(processed_data))

        # 时序自动特征 (tsfresh)
        ts_features = self.ts_extractor.extract(
            processed_data.pupil_timeseries,
            column_id="participant",
            column_sort="timestamp",
        )
        features.update(ts_features)

        return FeatureMatrix(features)

    def extract_cognitive_load(self, processed_data: ProcessedEyeData) -> float:
        """认知负荷指数计算 (ICA 改进版)"""
        pupil_signal = processed_data.filtered.pupil_diameter
        # 小波分解提取高频成分 (认知相关)
        coeffs = pywt.wavedec(pupil_signal, 'db4', level=5)
        ica_index = np.sum(np.abs(coeffs[1])) / len(coeffs[1])
        return ica_index
```

#### 4.2.3 多模态融合模型

```python
# eye_tracking/models/multimodal_fusion.py

class MultimodalCognitiveModel(nn.Module):
    """
    多模态认知状态预测模型
    输入: 眼动特征 + 生理信号 + 行为日志
    输出: 认知负荷等级 / 用户状态 / 交互意图
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        # 眼动特征编码器 (时序Transformer)
        self.gaze_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.gaze_dim,
                nhead=8,
                dim_feedforward=256,
                dropout=0.1,
            ),
            num_layers=4,
        )

        # 生理信号编码器 (1D-CNN + LSTM)
        self.physio_encoder = PhysioEncoder(
            input_channels=config.physio_channels,  # GSR, HR, EDA
            hidden_dim=128,
        )

        # 行为序列编码器
        self.behavior_encoder = BehaviorEncoder(
            vocab_size=config.action_vocab_size,
            embed_dim=64,
        )

        # 跨模态注意力融合
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.fusion_dim,
            num_heads=8,
        )

        # 预测头
        self.cognitive_load_head = nn.Linear(config.fusion_dim, 4)   # 4级认知负荷
        self.user_state_head = nn.Linear(config.fusion_dim, 6)       # 6种状态
        self.intent_head = nn.Linear(config.fusion_dim, config.num_intents)

    def forward(self, gaze_seq, physio_seq, behavior_seq):
        gaze_feat = self.gaze_encoder(gaze_seq)
        physio_feat = self.physio_encoder(physio_seq)
        behavior_feat = self.behavior_encoder(behavior_seq)

        # 拼接 → 跨模态注意力
        multimodal = torch.cat([gaze_feat, physio_feat, behavior_feat], dim=1)
        fused, attn_weights = self.cross_attention(multimodal, multimodal, multimodal)
        pooled = fused.mean(dim=1)

        return {
            "cognitive_load": self.cognitive_load_head(pooled),
            "user_state": self.user_state_head(pooled),
            "intent": self.intent_head(pooled),
            "attention_weights": attn_weights,  # 可解释性
        }
```

### 4.3 模块三：LLM 驱动的智能洞察生成

```python
# eye_tracking/insights/llm_analyzer.py

class LLMInsightGenerator:
    """大模型驱动的人因洞察自动生成"""

    SYSTEM_PROMPT = """你是一位资深人因工程研究员，擅长从眼动和多模态数据中
    提取设计洞察。请基于以下数据分析结果，生成结构化的体验设计建议。
    要求：1) 指出具体问题 2) 引用数据证据 3) 给出可执行的设计改进方案"""

    def generate_report(
        self,
        feature_matrix: FeatureMatrix,
        model_predictions: dict,
        experiment_context: ExperimentContext,
    ) -> InsightReport:
        """自动生成人因分析报告"""

        # 构建数据摘要
        data_summary = self._build_data_summary(feature_matrix, model_predictions)

        # 调用 LLM 生成洞察
        response = self.llm_client.generate(
            system=self.SYSTEM_PROMPT,
            user=f"""
            ## 实验信息
            产品: {experiment_context.product}
            场景: {experiment_context.scenario}
            被试数: {experiment_context.n_participants}

            ## 关键数据发现
            {data_summary}

            ## 模型预测结果
            认知负荷分布: {model_predictions['cognitive_load_distribution']}
            高负荷时段: {model_predictions['high_load_segments']}
            注意力热区异常: {model_predictions['attention_anomalies']}

            请生成包含以下部分的分析报告:
            1. 核心发现 (Top 5)
            2. 各 AOI 分析详情
            3. 认知负荷时序分析
            4. 设计改进建议 (按优先级排序)
            5. 后续研究建议
            """,
        )

        return InsightReport.parse(response)
```

### 4.4 模块四：可视化与交互面板

```python
# eye_tracking/visualization/dashboard.py

class AnalysisDashboard:
    """基于 Streamlit 的交互式分析面板"""

    def __init__(self):
        self.heatmap_renderer = HeatmapRenderer()
        self.scanpath_renderer = ScanpathRenderer()
        self.stats_panel = StatisticsPanel()

    def render(self):
        st.title("多模态人因分析平台")

        # 侧边栏: 实验选择与过滤
        with st.sidebar:
            experiment = st.selectbox("选择实验", self.list_experiments())
            participant_filter = st.multiselect("被试筛选", ...)
            metric_group = st.radio("指标组", ["基础", "AOI", "认知", "全部"])

        # 主面板布局
        col1, col2 = st.columns(2)

        with col1:
            # 注意力热力图 (支持时间滑块)
            st.subheader("注意力热力图")
            time_range = st.slider("时间窗口 (s)", 0.0, max_time, (0.0, max_time))
            heatmap = self.heatmap_renderer.generate(data, time_range)
            st.plotly_chart(heatmap)

        with col2:
            # 扫视路径图
            st.subheader("扫视路径")
            scanpath = self.scanpath_renderer.generate(data, top_n=10)
            st.plotly_chart(scanpath)

        # 认知负荷时序图
        st.subheader("认知负荷时序变化")
        cognitive_timeline = self.stats_panel.cognitive_load_timeline(data)
        st.plotly_chart(cognitive_timeline)

        # AOI 转移矩阵
        st.subheader("AOI 注意力转移矩阵")
        transition_matrix = self.stats_panel.aoi_transition_heatmap(data)
        st.plotly_chart(transition_matrix)

        # AI 洞察报告
        st.subheader("AI 生成洞察")
        if st.button("生成分析报告"):
            report = self.insight_generator.generate_report(...)
            st.markdown(report.to_markdown())
```

---

## 5. 人因研究全流程工作规范

### 5.1 研究流程

```
Phase 1: 研究设计          Phase 2: 数据采集         Phase 3: 分析建模
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ ① 文献调研       │    │ ⑤ 设备校准       │    │ ⑧ 数据预处理     │
│ ② 变量操作化定义  │ →  │ ⑥ 被试招募与筛选  │ →  │ ⑨ 特征提取       │
│ ③ 实验设计       │    │ ⑦ 实验实施       │    │ ⑩ 统计/ML 分析   │
│ ④ 伦理审批       │    │   + 多模态同步采集 │    │ ⑪ 多模态融合建模  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
Phase 5: 迭代优化          Phase 4: 洞察输出            │
┌─────────────────┐    ┌─────────────────┐             │
│ ⑮ A/B 验证       │    │ ⑫ 结果可视化     │  ←─────────┘
│ ⑯ 评测模型迭代   │ ←  │ ⑬ LLM 洞察生成   │
│ ⑰ 研究方法论沉淀  │    │ ⑭ 设计建议交付   │
└─────────────────┘    └─────────────────┘
```

### 5.2 实验设计模板

| 维度 | 规范要求 |
|-----|---------|
| **设计类型** | 被试内 / 被试间 / 混合设计，视研究问题选择 |
| **自变量** | 明确操作化定义，UI 方案 / 交互模式 / 设备参数等 |
| **因变量** | 眼动指标 + 主观评分 (SUS/NASA-TLX) + 任务绩效 |
| **样本量** | G*Power 功效分析，α=0.05, power=0.8, 中等效应量 |
| **对抗序效应** | 拉丁方 / 完全随机化 |
| **统计方法** | 重复测量 ANOVA / 线性混合效应模型 (LMM) / Bayesian |

---

## 6. 项目目录结构

```
multimodal-hf-platform/
├── README.md
├── pyproject.toml                  # 项目配置 (Poetry)
├── configs/
│   ├── collection_config.yaml      # 采集参数配置
│   ├── model_config.yaml           # 模型超参数
│   └── experiment_templates/       # 实验设计模板
│       ├── phone_ui_evaluation.yaml
│       ├── earphone_spatial_audio.yaml
│       ├── wearable_glanceability.yaml
│       └── tablet_multitask.yaml
│
├── src/
│   ├── eye_tracking/
│   │   ├── __init__.py
│   │   ├── collection/             # 数据采集模块
│   │   │   ├── tobii_connector.py
│   │   │   ├── lsl_synchronizer.py
│   │   │   └── recorder.py
│   │   ├── preprocessing/          # 预处理模块
│   │   │   ├── pipeline.py
│   │   │   ├── filters.py
│   │   │   ├── event_detection.py
│   │   │   ├── aoi_mapper.py
│   │   │   └── quality_check.py
│   │   ├── features/               # 特征工程模块
│   │   │   ├── extractor.py
│   │   │   ├── basic_metrics.py
│   │   │   ├── aoi_metrics.py
│   │   │   ├── cognitive_metrics.py
│   │   │   └── timeseries_features.py
│   │   ├── models/                 # AI 模型模块
│   │   │   ├── multimodal_fusion.py
│   │   │   ├── cognitive_load_predictor.py
│   │   │   ├── intent_predictor.py
│   │   │   └── fatigue_detector.py
│   │   ├── insights/               # 洞察生成模块
│   │   │   ├── llm_analyzer.py
│   │   │   └── report_generator.py
│   │   └── visualization/          # 可视化模块
│   │       ├── dashboard.py
│   │       ├── heatmap.py
│   │       ├── scanpath.py
│   │       └── statistics_panel.py
│   │
│   ├── physio/                     # 生理信号处理
│   │   ├── gsr_processor.py
│   │   ├── hr_processor.py
│   │   └── eda_analyzer.py
│   │
│   ├── behavior/                   # 行为数据处理
│   │   ├── touch_analyzer.py
│   │   ├── gesture_recognizer.py
│   │   └── task_performance.py
│   │
│   └── common/                     # 公共组件
│       ├── data_models.py
│       ├── config.py
│       ├── database.py
│       └── utils.py
│
├── experiments/                    # 实验脚本与数据
│   ├── exp001_phone_navigation/
│   ├── exp002_earphone_anc/
│   └── exp003_watch_glance/
│
├── notebooks/                      # 分析 Notebooks
│   ├── exploratory_analysis.ipynb
│   ├── model_training.ipynb
│   └── statistical_tests.ipynb
│
├── tests/                          # 测试
│   ├── test_preprocessing.py
│   ├── test_features.py
│   ├── test_models.py
│   └── fixtures/
│
└── docs/                           # 文档
    ├── research_protocol.md
    ├── api_reference.md
    └── experiment_guide.md
```

---

## 7. 人因能力建设：AI 赋能路线图

### 7.1 三阶段建设计划

```
阶段一 (Month 1)                阶段二 (Month 2)               阶段三 (Month 3)
━━━━━━━━━━━━━━━━━━━━           ━━━━━━━━━━━━━━━━━━━━          ━━━━━━━━━━━━━━━━━━━━
┌──────────────────┐           ┌──────────────────┐          ┌──────────────────┐
│ 基础设施搭建      │           │ AI 模型开发       │          │ 智能闭环构建      │
│                  │           │                  │          │                  │
│ • 数据采集流水线   │           │ • 多模态融合模型   │          │ • 实时分析系统     │
│ • 预处理自动化    │     →     │ • 认知负荷预测    │    →     │ • 自动报告生成     │
│ • 特征提取引擎    │           │ • 意图识别模型    │          │ • 设计建议闭环     │
│ • 基础可视化      │           │ • 评测模型构建    │          │ • 方法论文档沉淀   │
└──────────────────┘           └──────────────────┘          └──────────────────┘
```

### 7.2 AI 技术赋能人因研究的关键创新点

| 创新方向 | 传统方式 | AI 赋能方式 | 效率提升 |
|---------|---------|------------|---------|
| **AOI 定义** | 手工在界面上画矩形 | SAM 自动语义分割 + UI 元素树映射 | 10x |
| **事件检测** | 固定阈值 I-VT/I-DT | CNN 自适应事件分类 | 准确率 +8% |
| **特征提取** | 手动计算 10-15 个指标 | tsfresh 自动提取 400+ 特征 | 30x |
| **认知评估** | NASA-TLX 主观量表 | 实时瞳孔 + 生理多模态预测 | 实时化 |
| **报告生成** | 手写 PPT 2-3 天 | LLM 自动生成结构化报告 | 5x |
| **模式发现** | 研究员经验判断 | 聚类 + 异常检测自动发现 | 发现隐藏模式 |

---

## 8. 典型应用场景

### 8.1 场景一：手机 UI 导航效率评测

```
研究问题: 新版设置页信息架构是否降低用户查找目标设置项的认知负荷？

实验设计:
  ├── 被试内设计: 旧版 vs 新版 (拉丁方平衡)
  ├── 任务: 6 个典型设置查找任务
  ├── 因变量:
  │   ├── 眼动: 首次注视到达时间, 搜索路径最优比, 注视熵
  │   ├── 瞳孔: 任务期间 TEPR (认知负荷指标)
  │   ├── 行为: 任务完成时间, 错误率
  │   └── 主观: SUS, NASA-TLX
  ├── 被试: N=30 (G*Power: α=0.05, d=0.5, power=0.8)
  └── 分析:
      ├── 统计: 配对 t-test / Wilcoxon (非正态), LMM 控制个体差异
      └── AI: 多模态融合模型预测认知负荷, LLM 生成设计改进建议
```

### 8.2 场景二：穿戴设备表盘信息获取速度

```
研究问题: 不同表盘布局下，用户手腕抬起后多快能获取关键信息？

关键指标:
  ├── Glance Duration: 抬腕到获取信息的总时长 (目标 < 3s)
  ├── 首次注视落点: 是否落在关键信息区 (时间/通知)
  ├── 注视次数: 获取目标信息所需注视次数 (越少越好)
  └── 认知负荷: 信息密度对瞳孔散度的影响

AI 加持:
  └── 训练 Glanceability 预测模型: 输入表盘设计参数 → 输出预测获取时长
```

### 8.3 场景三：耳机空间音频感知评测

```
研究问题: 空间音频渲染算法 A vs B，哪个带来更自然的声源定位感知？

多通道协同:
  ├── 听觉: 声源方位判断准确率, 响应时间
  ├── 视觉(眼动): 视觉搜索与听觉线索的一致性 (视听整合)
  ├── 行为: 头动追踪数据 (头部转向与声源方位匹配度)
  └── 生理: GSR 响应 (沉浸感指标)
```

---

## 9. 关键依赖与环境要求

### 9.1 Python 环境

```toml
# pyproject.toml (核心依赖)
[tool.poetry.dependencies]
python = "^3.10"

# 数据处理
pandas = "^2.2"
numpy = "^1.26"
scipy = "^1.12"

# 眼动专用
tobii-research = "^1.11"      # Tobii Pro SDK
pylsl = "^1.16"               # Lab Streaming Layer

# 机器学习
scikit-learn = "^1.4"
tsfresh = "^0.20"
pytorch = "^2.2"
transformers = "^4.38"

# 统计
statsmodels = "^0.14"
pingouin = "^0.5"             # 人因常用统计包

# 可视化
plotly = "^5.18"
streamlit = "^1.31"
matplotlib = "^3.8"
seaborn = "^0.13"

# LLM
anthropic = "^0.45"           # Claude API

# 计算机视觉 (AOI 自动分割)
segment-anything = "^1.0"
ultralytics = "^8.1"          # YOLO

# 信号处理
pywavelets = "^1.5"
neurokit2 = "^0.2"            # 生理信号处理
```

### 9.2 硬件要求

| 设备 | 推荐型号 | 用途 |
|-----|---------|-----|
| 屏幕式眼动仪 | Tobii Pro Spark (120Hz) | 实验室 UI 评测 |
| 穿戴式眼动仪 | Tobii Pro Glasses 3 (100Hz) | 移动场景 + 穿戴设备评测 |
| 生理采集设备 | Shimmer3 GSR+ / Empatica E4 | GSR, HR, EDA |
| GPU 工作站 | NVIDIA RTX 4090 / A100 | 模型训练与推理 |

---

## 10. 风险与对策

| 风险 | 等级 | 对策 |
|-----|------|------|
| 眼动数据质量不稳定 | 高 | 自动质量检查 + 严格校准协议 + 采集后即时验证 |
| 多模态同步精度不足 | 中 | LSL 统一时钟 + 事件标记交叉校验 |
| 模型泛化性差 | 中 | 跨产品/跨用户迁移学习 + 在线微调 |
| 伦理合规风险 | 高 | 知情同意 + 数据脱敏 + IRB 审批 |
| LLM 洞察幻觉 | 中 | 数据锚定提示 + 人工审核闭环 |

---

## 11. 交付物清单

| 阶段 | 交付物 | 形式 |
|------|-------|------|
| Month 1 | 数据采集与预处理 Pipeline | 可运行代码 + 单元测试 |
| Month 1 | 眼动特征提取引擎 | Python 模块 + API 文档 |
| Month 2 | 多模态认知负荷预测模型 | 训练脚本 + 模型权重 + 评估报告 |
| Month 2 | 交互意图预测模型 | 同上 |
| Month 3 | 可视化分析面板 | Streamlit 应用 |
| Month 3 | LLM 洞察生成模块 | 集成至面板 |
| Month 3 | 人因研究方法论文档 | Markdown + 实验模板 |
| 全程 | 至少 1 个完整产品评测案例 | 数据 + 分析 + 报告 |

---

## 附录 A: 岗位能力匹配说明

| 岗位要求 | 本项目体现的能力 |
|---------|----------------|
| 制定人因研究计划和方案 | 第5节完整研究流程 + 第8节三个典型场景设计 |
| 视/听/触多通道研究 | 第2节感觉通道-变量-指标映射框架 |
| 手机/耳机/穿戴/平板PC产品 | 第2.2节产品-通道-场景矩阵 |
| 运用AI技术提升研究效率/信度/效度 | 第4.2-4.3节 AI 特征提取 + 多模态融合 + LLM 洞察 |
| 人因工程方法论理解 | 第5节研究全流程 + 第5.2节实验设计规范 |
| 从生理/心理/行为角度拆解人因变量 | 第2.1节三层变量拆解框架 |
| 独立完成人因研究全流程 | 第5.1节 17 步全流程覆盖 + 第11节交付物清单 |
| Python/数据分析能力 | 全部代码示例均为 Python，涵盖统计 + ML |
| AI驱动人因应用 | 多模态融合模型 + LLM报告生成 + 自动特征提取 |
| 眼动/生理/行为多模态数据处理 | 第4节四大核心模块完整覆盖 |

---

*文档结束 — 本文档作为多模态人因分析平台的开发蓝图和能力展示材料。*

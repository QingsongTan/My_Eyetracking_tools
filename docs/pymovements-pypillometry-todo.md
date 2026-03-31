# `pymovements` + `pypillometry` 开发任务清单

这份 TODO 只压缩当前已经确认的接入方案，按可逐步实现的顺序拆开，避免一次性改太大。

## Phase 1A：`pypillometry` 最小接入

- [x] 新增 `src/gaze_toolkit/pupil_preprocess.py`
- [x] 定义 `PupilProcessingResult`
- [x] 实现 `preprocess_pupil_signal()`
- [x] 实现 `extract_pupil_load_features()`
- [x] 保持 `pypillometry` 为可选依赖，不阻塞现有项目运行

## Phase 1B：特征链路接入

- [x] 在 `src/gaze_toolkit/features.py` 增加 `include_pupil_load_features`
- [x] 把 pupil load 特征 merge 到 `extract_features()` 输出
- [x] 保持默认行为不变，只有显式开启时才添加新特征

## Phase 1C：测试与验收

- [x] 新增 pupil 预处理测试
- [x] 覆盖缺失值、无效值、blink 片段插值
- [x] 覆盖 `extract_features(..., include_pupil_load_features=True)`
- [x] 确认没有 pupil 数据时不会崩溃

## Phase 2：分析与流水线联动

- [x] `analysis.py` 增加 pupil summary / trace 输出
- [x] `pipeline.py` 增加 `include_pupil_load_features` 参数透传
- [x] 在配置文件里开放 pupil 特征开关

## Phase 3：Dashboard 展示

- [x] 增加 `Pupil / Cognitive Load` 区块
- [x] 展示清洗前后 pupil 曲线
- [x] 展示 baseline-corrected pupil 曲线
- [x] 展示 pupil 负荷特征摘要

## Phase 4：`pymovements` adapter

- [x] 新增 `pymovements` adapter 模块
- [x] 先做数据读取与标准化，不替换现有 `GazeRecording`
- [x] 增加公共数据集加载与事件检测对照实验主模块
- [ ] 再评估是否接更多公共数据集下载和事件检测对照

# 真实 workload 标签数据示例

这份示例用来说明：如果你要把眼动数据接成真正的工作负荷分类 / 回归实验，数据应该怎么整理。

## 1. 先明确粒度

真实 workload 标签通常不是“逐采样点”标注，而是按下面粒度整理：

- `session` 粒度：一次完整任务会话
- `trial` 粒度：一次试次、一次题目、一次页面交互

当前这个项目的建模入口，最适合的是 **一行 = 一个 trial/session 的特征表**。

## 2. 推荐字段

### 2.1 标签列

最少建议保留这两个目标列：

- `workload_level`：分类标签，示例值 `low / medium / high`
- `workload_score`：回归标签，示例值 `0-100` 的连续分数

如果你手里有更原始的量表，也可以保留：

- `nasa_tlx_total`
- `nasa_tlx_mental`
- `nasa_tlx_temporal`
- `nasa_tlx_physical`
- `nasa_tlx_performance`
- `nasa_tlx_effort`
- `nasa_tlx_frustration`

### 2.2 特征列

特征列就是你从眼动 / 瞳孔里提取出来的数值特征，例如：

- `fixation_count`
- `fixation_duration_mean`
- `saccade_count`
- `blink_rate_hz`
- `pupil_bc_mean`
- `pupil_bc_peak`
- `pupil_phasic_peak`

### 2.3 元数据列

建议保留这些列帮助你追溯实验语境：

- `participant_id`
- `session_id`
- `trial_id`
- `task_name`
- `condition`
- `task_difficulty`
- `stimulus_id`

## 3. 适合当前项目的 CSV 形状

下面是一种可以直接喂给当前“工作负荷实验台”的结构：

```csv
participant_id,session_id,trial_id,task_name,condition,task_difficulty,workload_level,workload_score,nasa_tlx_total,fixation_count,fixation_duration_mean,saccade_count,blink_rate_hz,pupil_bc_mean,pupil_bc_peak,pupil_phasic_peak
P01,S01,T01,search,low,1,low,21,24,41,168.2,19,0.42,0.11,0.18,0.20
P01,S01,T02,search,medium,2,medium,46,53,56,142.7,26,0.58,0.22,0.35,0.38
P01,S01,T03,search,high,3,high,73,76,68,118.4,33,0.79,0.36,0.55,0.61
P02,S02,T01,reading,low,1,low,18,20,39,174.5,17,0.36,0.10,0.15,0.17
P02,S02,T02,reading,high,3,high,69,71,65,121.3,31,0.74,0.33,0.52,0.58
```

## 4. 这份表怎么用

- 如果你要做 **分类**，选 `workload_level`
- 如果你要做 **回归**，选 `workload_score`
- 其余数值列都可以作为特征
- `participant_id`、`session_id`、`trial_id` 这类 ID 列不要作为模型特征

## 5. 实操建议

- 如果你是用 NASA-TLX，建议先保留原始分量，再派生一个总分或标准化分数
- 如果你是用任务难度做标签，建议把难度等级和真实量表分开保存，不要混成一个列
- 如果一条记录对应多个时间窗，可以在 `trial_id` 下再加 `window_id`

## 6. 与当前项目的对应关系

当前这个项目的认知负荷实验台，读取的是“特征表”，不是原始采样流。

也就是说，真实流程通常是：

1. 原始眼动 / 瞳孔数据
2. 预处理与特征提取
3. 合并 workload 标签
4. 形成一行一个 trial/session 的建模表
5. 进入分类 / 回归实验台

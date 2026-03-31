# 事件检测方法验证报告

- generated_at: 2026-03-28 11:57:41
- dataset_name: ToyDatasetEyeLink
- recording_label: ToyDatasetEyeLink_0

## 研究结论摘要

**pymovements I-VT 与 EyeLink Ground Truth 最接近**

- 最佳样本重叠率为 0.847，F1 为 0.917。
- 项目原生阈值法与 pymovements I-VT 的结果高度一致，二者样本重叠率为 1.000。
- 当前参数下，pymovements I-DT 与 Ground Truth 偏差较大，适合作为方法敏感性讨论案例。

## 方法摘要对照

| method | fixation_count | fixation_duration_mean_ms | fixation_duration_total_ms | fixation_amplitude_mean |
| --- | --- | --- | --- | --- |
| EyeLink Ground Truth | 409 | 283.7139 | 116039 | 0 |
| Native Threshold | 553 | 272.5226 | 150705 | 6.0562 |
| pymovements I-VT | 553 | 272.5208 | 150704 | 6.0561 |
| pymovements I-DT | 0 | 0 | 0 | 0 |

## 样本级一致性对照

| comparison | sample_overlap_ratio | precision | recall | f1 |
| --- | --- | --- | --- | --- |
| ground_truth_vs_native | 0.8475 | 0.9913 | 0.8538 | 0.9174 |
| ground_truth_vs_pymovements_ivt | 0.8475 | 0.9913 | 0.8538 | 0.9175 |
| ground_truth_vs_pymovements_idt | 0 | 0 | 0 | 0 |
| native_vs_pymovements_ivt | 1 | 1 | 1 | 1 |
| native_vs_pymovements_idt | 0 | 0 | 0 | 0 |
| pymovements_ivt_vs_idt | 0 | 0 | 0 | 0 |

## 解释建议

- Ground Truth 来自 EyeLink ASC 中的 EFIX 事件时，可用于近似方法学验证，不应等同于人工二次标注金标准。
- 原生阈值法与 pymovements I-VT 若高度一致，说明当前阈值设置具有较好的跨工具可复现性。
- 若 I-DT 与 Ground Truth 偏差明显，可在面试中作为“参数敏感性”和“方法选择”讨论点。

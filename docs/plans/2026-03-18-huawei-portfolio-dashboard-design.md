# Huawei Portfolio Dashboard Design

## Objective

把现有眼动工具包收敛成一个更适合向华为人因研究专家和面试官展示的作品集原型。核心不是追求功能堆叠，而是证明一条完整能力链：多模态用户数据处理、自动特征提取、状态/意图建模、研究型可视化操作界面，以及面向后续扩展的工程结构。

## Input

- 原始眼动时序数据，默认来自 CSV 或模拟数据
- 可选刺激图
- 默认第二模态为模拟心率信号

## Processing Flow

1. 标准化眼动数据输入
2. 插值、平滑、坐标归一化
3. I-VT 事件检测
4. 单条记录特征提取
5. 合成 cohort 基线实验
6. 眼动单模态与眼动+心率多模态比较
7. 结果通过 Streamlit 研究控制台展示

## State Changes

系统显式保留以下状态：

- `raw_recording`
- `processed_recording`
- `enriched_recording`
- `feature_map`
- `event_table`
- `model_result`
- `holdout_predictions`
- `modality_comparison`

这样做是为了让研究过程可解释，而不是只返回最终指标。

## Output

- 单条记录的研究摘要
- 可直接展示的扫描路径、热图、信号概览
- 可复核的事件表与特征表
- 基线意图分类结果
- 多模态对比表
- 面向作品集陈述的话术结构

## Upstream / Downstream Impact

上游只要求时间戳和基本坐标即可接入工具链。下游可以继续接 notebook、批量实验、在线预测、深度模型和更多传感器模态。当前没有默认引入 EDF 原生解析或重型深度模型，是为了保证作品集版本保持可运行、可解释、可安装，而不是把复杂度前置到面试演示阶段。

## Assumptions And Unverified Preconditions

- 假设面试展示场景允许本地运行 Streamlit。
- 假设当前最重要的是证明“AI 驱动人因应用经验”，而不是做完整商用产品。
- 未验证真实华为内部数据格式，因此真实接入前仍需做字段映射或 loader 适配。

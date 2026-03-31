# 方法验证展示说明

## 目标

这部分能力不是为了证明“又接了一个外部包”，而是为了证明项目具备更接近正式研究工作的三件事：

1. 能接公共数据集，而不是只分析自造数据
2. 能利用原始 EyeLink 事件做近似 ground truth 对照
3. 能把方法差异转成可解释的研究结论和报告

这正好对应人因研究岗位里很关键的一类能力：

- 数据处理
- 方法验证
- 结果解释
- 研究信度意识

---

## 当前已实现的展示链路

### 1. 数据来源

- `pymovements ToyDataset`
- `pymovements ToyDatasetEyeLink`

其中 `ToyDatasetEyeLink` 会保留原始 `.asc` 文件路径，项目会从 `.asc` 中直接解析 `EFIX` 事件，作为近似 ground truth。

### 2. 对照方法

- 项目原生阈值法
- `pymovements I-VT`
- `pymovements I-DT`

### 3. 输出结果

- 方法摘要对照表
- 样本级一致性对照表
- 自动研究结论摘要
- Markdown 报告导出

---

## 在 Dashboard 里怎么演示

路径：

`意图建模实验台 -> pymovements 公共数据集事件检测对照`

推荐演示方式：

1. 选择 `ToyDatasetEyeLink`
2. `recording 索引` 保持 `0`
3. 点击“运行事件检测对照实验”
4. 讲解：
   - 哪种方法最接近 EyeLink Ground Truth
   - 原生方法与 `pymovements I-VT` 是否一致
   - `I-DT` 为什么偏差更大
5. 点击“下载方法验证 Markdown 报告”

---

## 面试时怎么讲

推荐讲法：

“这个项目不只是做眼动可视化。我专门加了一块方法验证模块，拿 `pymovements` 的公共 EyeLink toy dataset 做事件检测对照。因为 `ToyDatasetEyeLink` 保留了原始 `.asc`，所以我进一步解析了其中的 `EFIX` 事件，把它作为近似 ground truth。然后我把项目里的原生阈值法、`pymovements I-VT` 和 `I-DT` 放到同一条 recording 上做样本级 overlap、precision、recall、F1 对比。这样我在作品集中展示的不只是‘能算特征、能画图’，而是‘我知道怎么验证一种方法是否可信，以及怎么解释不同事件检测算法在真实数据上的差异’。”

---

## 当前最适合强调的研究结论

在当前默认验证结果里：

- `pymovements I-VT` 与 EyeLink Ground Truth 最接近
- 项目原生阈值法与 `pymovements I-VT` 高度一致
- `pymovements I-DT` 在当前参数下偏差明显更大

这很适合作为“方法敏感性”讨论点，说明你不是把算法输出当作黑箱结论，而是在做研究级判断。

---

## 当前边界

- 这里的 ground truth 来自 EyeLink 原始事件行，不等于人工二次标注金标准
- `I-DT` 对时间步长和缺失值更敏感，当前桥接层已经做了稳定化处理
- 这块最适合作为作品集中的“方法验证能力展示”，不应直接表述为正式算法 benchmark 论文结果

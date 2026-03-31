# Auto-extracted from 02-特征提取与探索.ipynb — headless verification
# plt.show() replaced with plt.close('all')

# ── Cell 1 ───────────────────────────────────────────────────────────────
import warnings, sys, os, io
warnings.filterwarnings('ignore')

# Force UTF-8 output on Windows to avoid GBK encoding errors
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Use Agg backend for headless matplotlib
import matplotlib
matplotlib.use('Agg')

# Resolve paths relative to this script's directory (notebooks/)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_SCRIPT_DIR)
sys.path.insert(0, '../src')

from pathlib import Path
import pymovements as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from gaze_toolkit.pymovements_adapter import from_pymovements
from gaze_toolkit.preprocess import preprocess
from gaze_toolkit.events import attach_events
from gaze_toolkit.features import extract_features
from gaze_toolkit.quality import assess_quality

# ── 加载数据集并批量提取特征 ─────────────────────────────────────────────
DATA_PATH = Path('..').resolve() / '.cache' / 'pm_gt_probe' / 'ToyDataset'
ds = pm.Dataset('ToyDataset', path=DATA_PATH)
ds.load()

fileinfo = ds.fileinfo['gaze'].to_pandas()
fileinfo = fileinfo.drop_duplicates(subset=['text_id', 'page_id']).reset_index(drop=True)

rows = []
for idx in fileinfo.index:
    text_id = int(fileinfo.loc[idx, 'text_id'])
    page_id = int(fileinfo.loc[idx, 'page_id'])
    rec = from_pymovements(ds.gaze[idx], sampling_rate_hz=1000.0)
    rec_clean = preprocess(rec)
    rec_ev = attach_events(rec_clean)
    feats = extract_features(rec_ev)
    feats['text_id'] = text_id
    feats['page_id'] = page_id
    rows.append(feats)

df = pd.DataFrame(rows)
feature_cols = [c for c in df.columns if c not in ('text_id', 'page_id')]
print(f'提取完成：{len(df)} 个 trial × {len(feature_cols)} 维特征')
print(f'\n前 10 个特征名：')
for i, c in enumerate(feature_cols[:10]):
    print(f'  {i+1:2d}. {c}')
print(f'  ...')
print(f'  {len(feature_cols)}. {feature_cols[-1]}')

# ── Cell 2 ───────────────────────────────────────────────────────────────
# ── 按人因研究语义分组展示 ────────────────────────────────────────────────
FEATURE_GROUPS = {
    '时序基础': ['duration_ms', 'sample_count', 'valid_ratio', 'path_length'],
    '注视指标': ['fixation_count', 'fixation_duration_mean', 'fixation_duration_total', 'fixation_density'],
    '扫视指标': ['saccade_count', 'saccade_amplitude_mean', 'saccade_peak_velocity_mean', 'saccade_latency_mean'],
    '眨眼指标': ['blink_count', 'blink_rate_hz', 'blink_duration_mean'],
    '速度统计': ['velocity_mean', 'velocity_peak'],
    '瞳孔指标': ['pupil_baseline', 'pupil_change_rate'],
}

for group_name, cols in FEATURE_GROUPS.items():
    available = [c for c in cols if c in df.columns]
    if available:
        print(f'\n{"="*60}')
        print(f'  {group_name}')
        print(f'{"="*60}')
        desc = df[available].describe().T[['mean', 'std', 'min', 'max']]
        desc.columns = ['均值', '标准差', '最小值', '最大值']
        print(desc.round(2).to_string())

# ── Cell 3 ───────────────────────────────────────────────────────────────
# ── 选择方差非零的数值特征 ────��───────────────────────────────────────────
numeric_cols = [c for c in feature_cols if df[c].std() > 1e-8]

# 按语义分组排列，让热力图更易读
ORDERED_GROUPS = [
    # 注视相关
    'fixation_count', 'fixation_duration_mean', 'fixation_duration_total', 'fixation_density',
    # 扫视相关
    'saccade_count', 'saccade_amplitude_mean', 'saccade_peak_velocity_mean',
    # 基础时序
    'duration_ms', 'sample_count', 'path_length', 'velocity_mean', 'velocity_peak',
    # 位置统计
    'x_mean', 'x_std', 'y_mean', 'y_std',
    # 复杂度
    'x_approx_entropy', 'y_approx_entropy',
]
ordered_cols = [c for c in ORDERED_GROUPS if c in numeric_cols]
# 加上遗漏的列
ordered_cols += [c for c in numeric_cols if c not in ordered_cols]

corr = df[ordered_cols].corr()

fig, ax = plt.subplots(figsize=(14, 11))
im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

ax.set_xticks(range(len(ordered_cols)))
ax.set_yticks(range(len(ordered_cols)))
ax.set_xticklabels(ordered_cols, rotation=90, fontsize=7)
ax.set_yticklabels(ordered_cols, fontsize=7)
plt.colorbar(im, ax=ax, label='Pearson r', shrink=0.8)
ax.set_title(f'特征相关性矩阵（{len(ordered_cols)} 维，按语义分组排列）', fontsize=13, pad=12)
plt.tight_layout()
plt.savefig('../examples/nb02_correlation_heatmap.png', dpi=120, bbox_inches='tight')
plt.close('all')

# 输出高相关对
high_corr_pairs = []
for i in range(len(ordered_cols)):
    for j in range(i+1, len(ordered_cols)):
        r = abs(corr.iloc[i, j])
        if r > 0.85:
            high_corr_pairs.append((ordered_cols[i], ordered_cols[j], corr.iloc[i, j]))

print(f'\n|r| > 0.85 的高相关特征对（共 {len(high_corr_pairs)} 对）：')
for a, b, r in sorted(high_corr_pairs, key=lambda x: -abs(x[2]))[:8]:
    print(f'  {a:30s} ↔ {b:30s}  r={r:+.3f}')

# ── Cell 4 ───────────────────────────────────────────────────────────────
# ── PCA ──────────────────────────────────────────────────────────────────
X = df[numeric_cols].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=min(10, len(numeric_cols)))
X_pca = pca.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']
text_labels = df['text_id'].values

# ── 子图1：前两个主成分散点 ──────────────────────────────────────────────
ax = axes[0]
for tid in sorted(df['text_id'].unique()):
    mask = text_labels == tid
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=colors[tid], s=80, alpha=0.8,
               label=f'Text {tid}', edgecolors='white', linewidth=0.5)
    # 标注页码
    for j, (px, py) in enumerate(zip(X_pca[mask, 0], X_pca[mask, 1])):
        page = df.loc[mask, 'page_id'].iloc[j]
        ax.annotate(f'p{page}', (px, py), fontsize=7, ha='center', va='bottom')

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} 方差)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} 方差)')
ax.set_title('PCA 降维：不同文本在特征空间中的位置', fontsize=11)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# ── 子图2：前 10 个主成分的累计方差解释率 ────────────────────────────────
ax2 = axes[1]
cumvar = np.cumsum(pca.explained_variance_ratio_)
n_components = len(cumvar)
ax2.bar(range(1, n_components+1), pca.explained_variance_ratio_,
        color='steelblue', alpha=0.7, label='单个成分')
ax2.plot(range(1, n_components+1), cumvar, 'ro-', lw=1.5, label='累计方差')
ax2.axhline(0.9, color='gray', ls='--', lw=1, alpha=0.6, label='90% 阈值')
n90 = int(np.searchsorted(cumvar, 0.9)) + 1
ax2.axvline(n90, color='red', ls=':', lw=1, alpha=0.6)
ax2.set_xlabel('主成分序号')
ax2.set_ylabel('方差解释率')
ax2.set_title(f'方差解释率（{n90} 个成分达到 90%）', fontsize=11)
ax2.legend(fontsize=9)
ax2.set_xticks(range(1, n_components+1))

plt.tight_layout()
plt.savefig('../examples/nb02_pca.png', dpi=120, bbox_inches='tight')
plt.close('all')

print(f'PC1+PC2 解释方差: {cumvar[1]:.1%}')
print(f'达到 90% 方差需要: {n90} 个主成分（共 {len(numeric_cols)} 维原始特征）')

# ── Cell 5 ───────────────────────────────────────────────────────────────
KEY_METRICS = {
    'fixation_duration_mean': '平均注视时长 (ms)\n↑ = 加工更深',
    'fixation_count': '注视次数\n↑ = 内容量大 / 搜索多',
    'saccade_amplitude_mean': '平均扫视幅度 (px)\n↑ = 跨度更广',
    'velocity_mean': '平均眼速 (px/s)\n↑ = 阅读速度快',
    'x_approx_entropy': 'X 近似熵\n↑ = 注视序列更不规则',
    'y_approx_entropy': 'Y 近似熵\n↑ = 垂直扫描更复杂',
}

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes_flat = axes.flatten()
colors_box = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']

for i, (col, ylabel) in enumerate(KEY_METRICS.items()):
    ax = axes_flat[i]
    data_by_text = [df.loc[df['text_id'] == tid, col].values for tid in range(4)]

    bp = ax.boxplot(data_by_text, patch_artist=True, widths=0.6,
                    medianprops={'color': 'black', 'linewidth': 1.5})
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # 叠加散点
    for tid in range(4):
        jitter = np.random.default_rng(tid).uniform(-0.12, 0.12, len(data_by_text[tid]))
        ax.scatter(np.full_like(data_by_text[tid], tid+1) + jitter,
                  data_by_text[tid], s=30, alpha=0.7, color=colors_box[tid],
                  edgecolors='white', linewidth=0.3, zorder=3)

    ax.set_xticklabels([f'Text {t}' for t in range(4)], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis='y', alpha=0.3)

fig.suptitle('关键人因指标的跨文本对比（20 trials, EyeLink 1000 Hz）', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('../examples/nb02_boxplots.png', dpi=120, bbox_inches='tight')
plt.close('all')

# ── Cell 6 ───────────────────────────────────────────────────────────────
# ── PC1 载荷条形图 ───────────────────────────────────────────────────────
loadings = pd.Series(pca.components_[0], index=numeric_cols)
top_loadings = loadings.abs().sort_values(ascending=False).head(15)
top_features = top_loadings.index.tolist()

fig, ax = plt.subplots(figsize=(10, 5))
values = loadings[top_features]
bar_colors = ['#FF5722' if v > 0 else '#2196F3' for v in values]
ax.barh(range(len(top_features)), values, color=bar_colors, alpha=0.8)
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features, fontsize=9)
ax.set_xlabel(f'PC1 载荷 ({pca.explained_variance_ratio_[0]:.1%} 方差)')
ax.set_title('PC1 特征载荷 Top-15（红色 = 正载荷，蓝色 = 负载荷）', fontsize=11)
ax.axvline(0, color='gray', lw=0.8)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('../examples/nb02_pc1_loadings.png', dpi=120, bbox_inches='tight')
plt.close('all')

print('PC1 解读：')
pos_feats = [f for f in top_features[:8] if loadings[f] > 0]
neg_feats = [f for f in top_features[:8] if loadings[f] < 0]
if pos_feats:
    print(f'  正载荷主要来自: {", ".join(pos_feats[:4])}')
    print(f'    → PC1 高分方向代表更长时间、更多样本、更多事件的 trial')
if neg_feats:
    print(f'  负载荷主要来自: {", ".join(neg_feats[:4])}')
    print(f'    → PC1 低分方向代表更短、更集中的阅读行为')

print('\n✅ _verify_nb02.py 全部执行完毕，无错误。')

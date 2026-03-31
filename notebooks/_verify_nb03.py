# Auto-extracted from 03-构建意图分类器（传统方法）.ipynb — headless verification script
import sys, io, os

# Fix encoding for Windows GBK terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Make relative paths work from the script's own location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Force non-interactive matplotlib backend BEFORE any other matplotlib import
import matplotlib
matplotlib.use('Agg')

# ============================================================================
# Cell 1 — imports + data loading + feature extraction
# ============================================================================
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '../src')

from pathlib import Path
import pymovements as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.inspection import permutation_importance

from gaze_toolkit.pymovements_adapter import from_pymovements
from gaze_toolkit.preprocess import preprocess
from gaze_toolkit.events import attach_events
from gaze_toolkit.features import extract_features

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

# 去除零方差特征
feature_cols = [c for c in feature_cols if df[c].std() > 1e-8]

X = df[feature_cols].fillna(0).values
y = df['text_id'].values

print(f'分类数据集：{X.shape[0]} 样本 × {X.shape[1]} 特征，{len(np.unique(y))} 类')
print(f'类别分布：{dict(zip(*np.unique(y, return_counts=True)))}')
print(f'标签含义：text_id ∈ {{0,1,2,3}}，每个 text 有 5 页（5 trials）')

# ============================================================================
# Cell 2 — LOO cross-validation, 4 classifiers
# ============================================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

MODELS = {
    'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=2000, random_state=42),
}

loo = LeaveOneOut()
results = {}

for name, model in MODELS.items():
    y_pred = cross_val_predict(model, X_scaled, y, cv=loo)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='macro')
    results[name] = {
        'accuracy': acc,
        'f1_macro': f1,
        'y_pred': y_pred,
    }
    print(f'{name:25s}  Accuracy={acc:.1%}  F1(macro)={f1:.3f}')

# 选择最佳模型
best_name = max(results, key=lambda k: results[k]['f1_macro'])
print(f'\n最佳模型: {best_name} (F1={results[best_name]["f1_macro"]:.3f})')

# ── 柱状图对比 ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
model_names = list(results.keys())
accs = [results[n]['accuracy'] for n in model_names]
f1s = [results[n]['f1_macro'] for n in model_names]
x_pos = np.arange(len(model_names))

bars1 = ax.bar(x_pos - 0.18, accs, 0.35, label='Accuracy', color='#2196F3', alpha=0.8)
bars2 = ax.bar(x_pos + 0.18, f1s, 0.35, label='F1 (macro)', color='#FF5722', alpha=0.8)

# 标注数值
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.0%}', ha='center', fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', fontsize=9)

ax.set_xticks(x_pos)
ax.set_xticklabels(model_names, fontsize=10)
ax.set_ylabel('Score')
ax.set_title('LOO 交叉验证：4 模型分类性能对比（4 类文本，20 trials）', fontsize=12)
ax.legend(fontsize=10)
ax.set_ylim(0, 1.15)
ax.axhline(0.25, color='gray', ls='--', lw=1, alpha=0.5, label='随机基线 (25%)')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('../examples/nb03_model_comparison.png', dpi=120, bbox_inches='tight')
plt.close('all')

# ============================================================================
# Cell 3 — Confusion matrix & classification report
# ============================================================================
y_pred_best = results[best_name]['y_pred']
cm = confusion_matrix(y, y_pred_best)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 子图 1：混淆矩阵热力图
ax = axes[0]
disp = ConfusionMatrixDisplay(cm, display_labels=[f'Text {i}' for i in range(4)])
disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
ax.set_title(f'{best_name} 混淆矩阵（LOO, n=20）', fontsize=11)

# 子图 2：各类别的 precision / recall / f1
ax2 = axes[1]
report = classification_report(y, y_pred_best, target_names=[f'Text {i}' for i in range(4)],
                                output_dict=True)
class_metrics = pd.DataFrame({
    'Precision': [report[f'Text {i}']['precision'] for i in range(4)],
    'Recall': [report[f'Text {i}']['recall'] for i in range(4)],
    'F1': [report[f'Text {i}']['f1-score'] for i in range(4)],
}, index=[f'Text {i}' for i in range(4)])

class_metrics.plot(kind='bar', ax=ax2, color=['#2196F3', '#FF5722', '#4CAF50'], alpha=0.8)
ax2.set_ylabel('Score')
ax2.set_title('各文本分类指标', fontsize=11)
ax2.set_ylim(0, 1.15)
ax2.legend(fontsize=9)
ax2.set_xticklabels(class_metrics.index, rotation=0)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('../examples/nb03_confusion_matrix.png', dpi=120, bbox_inches='tight')
plt.close('all')

# 文本报告
print(classification_report(y, y_pred_best,
                             target_names=[f'Text {i}' for i in range(4)]))

# ============================================================================
# Cell 4 — Permutation importance
# ============================================================================
best_model = MODELS[best_name].__class__(**MODELS[best_name].get_params())
best_model.fit(X_scaled, y)

perm_imp = permutation_importance(best_model, X_scaled, y,
                                   n_repeats=20, random_state=42,
                                   scoring='accuracy')

imp_df = pd.DataFrame({
    'feature': feature_cols,
    'importance_mean': perm_imp.importances_mean,
    'importance_std': perm_imp.importances_std,
}).sort_values('importance_mean', ascending=False)

# ── Top-15 排列重要性条形图 ───────────────────────────────────────────────
top_n = 15
top = imp_df.head(top_n)

fig, ax = plt.subplots(figsize=(10, 6))
colors_imp = ['#FF5722' if v > 0.01 else '#9E9E9E' for v in top['importance_mean']]
ax.barh(range(top_n), top['importance_mean'], xerr=top['importance_std'],
        color=colors_imp, alpha=0.8, capsize=3)
ax.set_yticks(range(top_n))
ax.set_yticklabels(top['feature'], fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('排列重要性（Accuracy 下降量）')
ax.set_title(f'{best_name} 排列重要性 Top-{top_n}（红色 = 高贡献）', fontsize=11)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('../examples/nb03_feature_importance.png', dpi=120, bbox_inches='tight')
plt.close('all')

# 人因解读
print('排列重要性 Top-5：')
for i, row in top.head(5).iterrows():
    print(f'  {row["feature"]:30s}  {row["importance_mean"]:.4f} +/- {row["importance_std"]:.4f}')

# ============================================================================
# Cell 5 — Misclassification analysis
# ============================================================================
misclassified = df.loc[y != y_pred_best, ['text_id', 'page_id',
                                            'duration_ms', 'fixation_count',
                                            'fixation_duration_mean',
                                            'saccade_amplitude_mean']].copy()
misclassified['predicted'] = y_pred_best[y != y_pred_best]
misclassified.columns = ['真实text', '页码', '时长(ms)', '注视数', '注视均时(ms)', '扫视幅度(px)', '预测text']

if len(misclassified) > 0:
    print(f'误分类 trial 数: {len(misclassified)} / {len(df)} ({len(misclassified)/len(df):.0%})\n')
    print(misclassified.to_string(index=False))

    # ── 可视化：误分类 trial 在 PCA 空间中的位置 ──────────────────────────
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']

    for tid in range(4):
        mask = y == tid
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=colors[tid], s=60, alpha=0.6,
                   label=f'Text {tid}', edgecolors='white', linewidth=0.5)

    # 标记误分类点
    mis_mask = y != y_pred_best
    ax.scatter(X_pca[mis_mask, 0], X_pca[mis_mask, 1],
               facecolors='none', edgecolors='red', s=200, linewidths=2,
               label='误分类', zorder=5)
    for j in np.where(mis_mask)[0]:
        ax.annotate(f'T{y[j]}p{df.iloc[j]["page_id"]}',
                   (X_pca[j, 0], X_pca[j, 1]),
                   fontsize=8, color='red', fontweight='bold',
                   xytext=(5, 5), textcoords='offset points')

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_title('误分类 trial 在 PCA 特征空间中的位置（红圈）', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('../examples/nb03_misclassified_pca.png', dpi=120, bbox_inches='tight')
    plt.close('all')
else:
    print('所有 trial 均正确分类！LOO Accuracy = 100%')
    print('注意：这可能表明模型过拟合，或 4 段文本的眼动模式确实差异显著。')

# ============================================================================
# Cell 6 — Feature ablation experiment
# ============================================================================
sorted_features = imp_df['feature'].tolist()
n_tests = [1, 2, 3, 5, 8, 10, 15, 20, len(feature_cols)]
n_tests = sorted(set([n for n in n_tests if n <= len(feature_cols)]))

ablation_results = []
for n_feat in n_tests:
    selected = sorted_features[:n_feat]
    col_idx = [feature_cols.index(f) for f in selected]
    X_sub = X_scaled[:, col_idx]

    model = MODELS[best_name].__class__(**MODELS[best_name].get_params())
    y_pred_sub = cross_val_predict(model, X_sub, y, cv=loo)
    acc = accuracy_score(y, y_pred_sub)
    ablation_results.append({'n_features': n_feat, 'accuracy': acc})
    print(f'  Top-{n_feat:2d} 特征 → LOO Accuracy = {acc:.1%}')

abl_df = pd.DataFrame(ablation_results)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(abl_df['n_features'], abl_df['accuracy'], 'o-', lw=2,
        color='#2196F3', markersize=8, markerfacecolor='white', markeredgewidth=2)
ax.axhline(0.25, color='gray', ls='--', lw=1, alpha=0.5, label='随机基线')
ax.fill_between(abl_df['n_features'], 0.25, abl_df['accuracy'],
                alpha=0.1, color='#2196F3')
ax.set_xlabel('使用的特征数量（按排列重要性排序）')
ax.set_ylabel('LOO Accuracy')
ax.set_title(f'特征消融实验：{best_name} 分类准确率 vs 特征数量', fontsize=11)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig('../examples/nb03_feature_ablation.png', dpi=120, bbox_inches='tight')
plt.close('all')

# 找到达到最高准确率 95% 的最少特征数
max_acc = abl_df['accuracy'].max()
threshold = max_acc * 0.95
sufficient = abl_df[abl_df['accuracy'] >= threshold].iloc[0]
print(f'\n达到最高准确率 95%（{threshold:.1%}）只需 Top-{int(sufficient["n_features"])} 个特征')

print('\n✅ _verify_nb03.py 全部执行完毕，无错误。')

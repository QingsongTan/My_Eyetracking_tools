from __future__ import annotations

from dataclasses import dataclass
import math
import warnings
from typing import Optional

import numpy as np
import pandas as pd


def _require_scipy():
    try:
        from scipy import stats

        return stats
    except ImportError as exc:
        raise ImportError(
            "统计分析模块需要 scipy。请运行: pip install gaze-toolkit[stats]"
        ) from exc


@dataclass
class StatTestResult:
    """统计检验结果。"""

    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    effect_size_name: str
    ci_lower: float
    ci_upper: float
    n: int
    conclusion: str

    def to_apa_string(self) -> str:
        """输出 APA 风格的统计报告字符串。"""
        statistic_label = _statistic_label(self.test_name)
        effect_label = "η²" if self.effect_size_name == "eta_sq" else self.effect_size_name
        p_text = _format_p_value(self.p_value, include_operator=True)

        if statistic_label == "t":
            df = max(self.n - (1 if "配对" in self.test_name else 2), 1)
            return f"t({df}) = {self.statistic:.2f}, p {p_text}, {effect_label} = {self.effect_size:.2f}"
        if statistic_label == "F":
            return f"F = {self.statistic:.2f}, p {p_text}, {effect_label} = {self.effect_size:.2f}"
        if statistic_label == "U":
            return f"U = {self.statistic:.2f}, p {p_text}, {effect_label} = {self.effect_size:.2f}"
        return f"W = {self.statistic:.2f}, p {p_text}, {effect_label} = {self.effect_size:.2f}"


def independent_t_test(group1: np.ndarray, group2: np.ndarray, var_name: str = "指标") -> StatTestResult:
    """执行独立样本 Welch t 检验，并计算 Cohen's d 与均值差 CI。"""
    stats = _require_scipy()
    left = _clean_numeric_array(group1)
    right = _clean_numeric_array(group2)
    _validate_group_lengths(left, right)

    test_result = stats.ttest_ind(left, right, equal_var=False)
    mean_diff = float(np.mean(left) - np.mean(right))
    pooled_std = _pooled_std(left, right)
    effect_size = mean_diff / pooled_std if pooled_std > 0 else 0.0

    se = math.sqrt(np.var(left, ddof=1) / len(left) + np.var(right, ddof=1) / len(right))
    df = _welch_df(left, right)
    t_crit = float(stats.t.ppf(0.975, df)) if df > 0 else np.nan
    ci_lower = mean_diff - t_crit * se if np.isfinite(t_crit) else np.nan
    ci_upper = mean_diff + t_crit * se if np.isfinite(t_crit) else np.nan

    return StatTestResult(
        test_name="独立样本 t 检验",
        statistic=float(test_result.statistic),
        p_value=float(test_result.pvalue),
        effect_size=float(effect_size),
        effect_size_name="Cohen's d",
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        n=int(len(left) + len(right)),
        conclusion=_build_conclusion(var_name, float(test_result.pvalue), effect_size, "Cohen's d"),
    )


def paired_t_test(before: np.ndarray, after: np.ndarray, var_name: str = "指标") -> StatTestResult:
    """执行配对样本 t 检验，并计算 Cohen's d_z 与差值 CI。"""
    stats = _require_scipy()
    left, right = _aligned_numeric_pairs(before, after)
    _validate_paired_length(left)

    differences = right - left
    test_result = stats.ttest_rel(left, right)
    std_diff = float(np.std(differences, ddof=1))
    effect_size = float(np.mean(differences) / std_diff) if std_diff > 0 else 0.0

    se = std_diff / math.sqrt(len(differences))
    t_crit = float(stats.t.ppf(0.975, len(differences) - 1))
    mean_diff = float(np.mean(differences))
    ci_lower = mean_diff - t_crit * se
    ci_upper = mean_diff + t_crit * se

    return StatTestResult(
        test_name="配对样本 t 检验",
        statistic=float(test_result.statistic),
        p_value=float(test_result.pvalue),
        effect_size=effect_size,
        effect_size_name="Cohen's d",
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        n=int(len(differences)),
        conclusion=_build_conclusion(var_name, float(test_result.pvalue), effect_size, "Cohen's d"),
    )


def wilcoxon_test(before: np.ndarray, after: np.ndarray, var_name: str = "指标") -> StatTestResult:
    """执行 Wilcoxon 符号秩检验，并计算 r 效应量。"""
    stats = _require_scipy()
    left, right = _aligned_numeric_pairs(before, after)
    _validate_paired_length(left)

    differences = right - left
    test_result = stats.wilcoxon(left, right, alternative="two-sided")
    n_pairs = len(differences[differences != 0])
    effect_size = _wilcoxon_effect_size(float(test_result.statistic), n_pairs, differences)

    return StatTestResult(
        test_name="Wilcoxon 符号秩检验",
        statistic=float(test_result.statistic),
        p_value=float(test_result.pvalue),
        effect_size=effect_size,
        effect_size_name="r",
        ci_lower=float("nan"),
        ci_upper=float("nan"),
        n=int(len(left)),
        conclusion=_build_conclusion(var_name, float(test_result.pvalue), effect_size, "r"),
    )


def mann_whitney_test(group1: np.ndarray, group2: np.ndarray, var_name: str = "指标") -> StatTestResult:
    """执行 Mann-Whitney U 检验，并计算 r 效应量。"""
    stats = _require_scipy()
    left = _clean_numeric_array(group1)
    right = _clean_numeric_array(group2)
    _validate_group_lengths(left, right)

    test_result = stats.mannwhitneyu(left, right, alternative="two-sided")
    effect_size = _mann_whitney_effect_size(float(test_result.statistic), len(left), len(right), left, right)

    return StatTestResult(
        test_name="Mann-Whitney U 检验",
        statistic=float(test_result.statistic),
        p_value=float(test_result.pvalue),
        effect_size=effect_size,
        effect_size_name="r",
        ci_lower=float("nan"),
        ci_upper=float("nan"),
        n=int(len(left) + len(right)),
        conclusion=_build_conclusion(var_name, float(test_result.pvalue), effect_size, "r"),
    )


def repeated_measures_anova(data: pd.DataFrame, dv: str, within: str, subject: str) -> StatTestResult:
    """手动实现单因素重复测量 ANOVA。"""
    stats = _require_scipy()
    required_columns = {dv, within, subject}
    missing = sorted(required_columns - set(data.columns))
    if missing:
        raise ValueError(f"Data is missing required columns: {missing}")

    frame = data[[dv, within, subject]].copy()
    frame[dv] = pd.to_numeric(frame[dv], errors="coerce")
    frame = frame.dropna(subset=[dv, within, subject])
    if frame.empty:
        raise ValueError("No valid rows available for repeated-measures ANOVA.")

    duplicate_mask = frame.duplicated([subject, within], keep=False)
    if duplicate_mask.any():
        raise ValueError("输入必须是 subject × condition 粒度的汇总表")

    pivot = frame.pivot(index=subject, columns=within, values=dv).dropna()
    if pivot.shape[0] < 2 or pivot.shape[1] < 3:
        raise ValueError("Repeated-measures ANOVA requires >=2 subjects and >=3 conditions.")

    matrix = pivot.to_numpy(dtype=float)
    n_subjects, n_conditions = matrix.shape
    grand_mean = float(matrix.mean())
    condition_means = matrix.mean(axis=0)
    subject_means = matrix.mean(axis=1)

    ss_total = float(np.sum((matrix - grand_mean) ** 2))
    ss_between = float(n_subjects * np.sum((condition_means - grand_mean) ** 2))
    ss_subjects = float(n_conditions * np.sum((subject_means - grand_mean) ** 2))
    ss_error = float(max(ss_total - ss_between - ss_subjects, 0.0))

    df_between = n_conditions - 1
    df_error = (n_conditions - 1) * (n_subjects - 1)
    ms_between = ss_between / df_between
    ms_error = ss_error / df_error if df_error > 0 else np.nan
    statistic = float(ms_between / ms_error) if ms_error and ms_error > 0 else float("inf")
    p_value = float(stats.f.sf(statistic, df_between, df_error)) if np.isfinite(statistic) else 0.0
    eta_sq = float(ss_between / ss_total) if ss_total > 0 else 0.0

    return StatTestResult(
        test_name="重复测量方差分析",
        statistic=statistic,
        p_value=p_value,
        effect_size=eta_sq,
        effect_size_name="eta_sq",
        ci_lower=float("nan"),
        ci_upper=float("nan"),
        n=int(n_subjects * n_conditions),
        conclusion=_build_conclusion(dv, p_value, eta_sq, "eta_sq"),
    )


def descriptive_table(data: pd.DataFrame, group_col: str, value_cols: list[str]) -> pd.DataFrame:
    """按组输出均值、标准差、中位数和近似 95% CI。"""
    if group_col not in data.columns:
        raise ValueError(f"Unknown group column: {group_col}")

    rows: list[dict[str, object]] = []
    for group_value, group_frame in data.groupby(group_col):
        for metric in value_cols:
            if metric not in group_frame.columns:
                continue
            values = pd.to_numeric(group_frame[metric], errors="coerce").dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue
            mean_value = float(np.mean(values))
            sd_value = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            margin = 1.96 * sd_value / math.sqrt(len(values)) if len(values) > 0 else 0.0
            rows.append(
                {
                    group_col: group_value,
                    "metric": metric,
                    "mean": mean_value,
                    "sd": sd_value,
                    "median": float(np.median(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "n": int(len(values)),
                    "ci_lower": float(mean_value - margin),
                    "ci_upper": float(mean_value + margin),
                }
            )

    return pd.DataFrame(rows)


def compare_conditions(
    summary_df: pd.DataFrame,
    condition_col: str,
    metric_cols: list[str],
    paired: bool = False,
    subject_col: Optional[str] = None,
) -> pd.DataFrame:
    """对 subject × condition 粒度的汇总表执行批量条件比较。"""
    stats = _require_scipy()
    if condition_col not in summary_df.columns:
        raise ValueError(f"Unknown condition column: {condition_col}")

    resolved_subject_col = subject_col or ("subject_id" if "subject_id" in summary_df.columns else None)
    _validate_summary_granularity(summary_df, condition_col, resolved_subject_col)

    conditions = [value for value in summary_df[condition_col].dropna().unique().tolist()]
    if len(conditions) != 2:
        raise ValueError("compare_conditions 仅支持恰好 2 个条件；>2 个条件请使用 repeated_measures_anova")

    rows: list[dict[str, object]] = []
    for metric in metric_cols:
        if metric not in summary_df.columns:
            continue

        subset_columns = [condition_col, metric]
        if resolved_subject_col is not None:
            subset_columns.append(resolved_subject_col)
        subset = summary_df[subset_columns].copy()
        subset[metric] = pd.to_numeric(subset[metric], errors="coerce")
        subset = subset.dropna(subset=[metric, condition_col])

        if paired:
            if resolved_subject_col is None:
                raise ValueError("配对检验需要 subject_col 或 subject_id 列。")
            pivot = subset.pivot(index=resolved_subject_col, columns=condition_col, values=metric)
            pivot = pivot.reindex(columns=conditions).dropna()
            if len(pivot) < 2:
                raise ValueError("配对检验需要至少 2 组有效配对样本。")
            group1 = pivot[conditions[0]].to_numpy(dtype=float)
            group2 = pivot[conditions[1]].to_numpy(dtype=float)
            normal = _is_normal(group1, stats=stats) and _is_normal(group2, stats=stats)
            result = paired_t_test(group1, group2, var_name=metric) if normal else wilcoxon_test(group1, group2, var_name=metric)
        else:
            group1 = subset.loc[subset[condition_col] == conditions[0], metric].dropna().to_numpy(dtype=float)
            group2 = subset.loc[subset[condition_col] == conditions[1], metric].dropna().to_numpy(dtype=float)
            _validate_group_lengths(group1, group2)
            normal = _is_normal(group1, stats=stats) and _is_normal(group2, stats=stats)
            result = (
                independent_t_test(group1, group2, var_name=metric)
                if normal
                else mann_whitney_test(group1, group2, var_name=metric)
            )

        rows.append(
            {
                "metric": metric,
                "test_name": result.test_name,
                "statistic": result.statistic,
                "p_value": result.p_value,
                "effect_size": result.effect_size,
                "effect_size_name": result.effect_size_name,
                "ci_lower": result.ci_lower,
                "ci_upper": result.ci_upper,
                "conclusion": result.conclusion,
            }
        )

    return pd.DataFrame(rows)


def _clean_numeric_array(values: np.ndarray | pd.Series | list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def _aligned_numeric_pairs(before: np.ndarray, after: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(before, dtype=float)
    right = np.asarray(after, dtype=float)
    if left.shape != right.shape:
        raise ValueError("Paired comparisons require arrays with the same shape.")
    mask = np.isfinite(left) & np.isfinite(right)
    return left[mask], right[mask]


def _validate_group_lengths(group1: np.ndarray, group2: np.ndarray) -> None:
    if len(group1) < 2 or len(group2) < 2:
        raise ValueError("Each group must contain at least 2 valid observations.")


def _validate_paired_length(values: np.ndarray) -> None:
    if len(values) < 2:
        raise ValueError("Paired comparison requires at least 2 valid pairs.")


def _pooled_std(group1: np.ndarray, group2: np.ndarray) -> float:
    n1 = len(group1)
    n2 = len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1 = float(np.var(group1, ddof=1))
    var2 = float(np.var(group2, ddof=1))
    pooled_var = (((n1 - 1) * var1) + ((n2 - 1) * var2)) / max(n1 + n2 - 2, 1)
    return float(math.sqrt(max(pooled_var, 0.0)))


def _welch_df(group1: np.ndarray, group2: np.ndarray) -> float:
    var1 = float(np.var(group1, ddof=1))
    var2 = float(np.var(group2, ddof=1))
    term1 = var1 / len(group1)
    term2 = var2 / len(group2)
    denominator = 0.0
    if len(group1) > 1:
        denominator += (term1**2) / (len(group1) - 1)
    if len(group2) > 1:
        denominator += (term2**2) / (len(group2) - 1)
    numerator = (term1 + term2) ** 2
    return numerator / denominator if denominator > 0 else float("nan")


def _wilcoxon_effect_size(statistic: float, non_zero_pairs: int, differences: np.ndarray) -> float:
    if non_zero_pairs <= 0:
        return 0.0
    expected = non_zero_pairs * (non_zero_pairs + 1) / 4.0
    sd = math.sqrt(non_zero_pairs * (non_zero_pairs + 1) * (2 * non_zero_pairs + 1) / 24.0)
    if sd == 0.0:
        return 0.0
    sign = 1.0 if float(np.mean(differences)) >= 0 else -1.0
    z_value = ((statistic - expected) / sd) * sign
    return float(z_value / math.sqrt(non_zero_pairs))


def _mann_whitney_effect_size(
    statistic: float,
    n1: int,
    n2: int,
    group1: np.ndarray,
    group2: np.ndarray,
) -> float:
    mean_u = n1 * n2 / 2.0
    sd_u = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    if sd_u == 0.0:
        return 0.0
    sign = 1.0 if float(np.mean(group1) - np.mean(group2)) >= 0 else -1.0
    z_value = ((statistic - mean_u) / sd_u) * sign
    return float(z_value / math.sqrt(n1 + n2))


def _build_conclusion(var_name: str, p_value: float, effect_size: float, effect_size_name: str) -> str:
    significance = "差异显著" if p_value < 0.05 else "差异不显著"
    effect_descriptor = _effect_size_descriptor(effect_size, effect_size_name)
    effect_label = "η²" if effect_size_name == "eta_sq" else effect_size_name
    return f"{var_name}{significance}, p {_format_p_value(p_value, include_operator=True)}, {effect_label} = {effect_size:.2f}（{effect_descriptor}）"


def _effect_size_descriptor(effect_size: float, effect_size_name: str) -> str:
    magnitude = abs(float(effect_size))
    if effect_size_name == "Cohen's d":
        if magnitude < 0.2:
            return "微小"
        if magnitude < 0.5:
            return "小"
        if magnitude < 0.8:
            return "中"
        return "大"
    if effect_size_name == "r":
        if magnitude < 0.1:
            return "微小"
        if magnitude < 0.3:
            return "小"
        if magnitude < 0.5:
            return "中"
        return "大"
    if magnitude < 0.01:
        return "微小"
    if magnitude < 0.06:
        return "小"
    if magnitude < 0.14:
        return "中"
    return "大"


def _format_p_value(p_value: float, *, include_operator: bool) -> str:
    if p_value < 0.001:
        return "< .001" if include_operator else ".001"
    text = f"{p_value:.3f}"
    if text.startswith("0"):
        text = text[1:]
    return f"= {text}" if include_operator else text


def _statistic_label(test_name: str) -> str:
    if "t 检验" in test_name:
        return "t"
    if "方差分析" in test_name:
        return "F"
    if "Mann-Whitney" in test_name:
        return "U"
    return "W"


def _is_normal(values: np.ndarray, *, stats) -> bool:
    cleaned = _clean_numeric_array(values)
    if len(cleaned) < 3:
        return False
    if np.allclose(cleaned, cleaned[0]):
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shapiro = stats.shapiro(cleaned)
    return bool(np.isfinite(shapiro.pvalue) and shapiro.pvalue > 0.05)


def _validate_summary_granularity(summary_df: pd.DataFrame, condition_col: str, subject_col: str | None) -> None:
    if subject_col is not None and subject_col in summary_df.columns:
        duplicate_mask = summary_df.duplicated([subject_col, condition_col], keep=False)
        if duplicate_mask.any():
            raise ValueError("输入必须是 subject × condition 粒度的汇总表")

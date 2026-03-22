from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gaze_toolkit.statistics import (
    StatTestResult,
    compare_conditions,
    descriptive_table,
    independent_t_test,
    mann_whitney_test,
    paired_t_test,
    repeated_measures_anova,
    wilcoxon_test,
)


def test_independent_t_test_significant() -> None:
    group1 = np.array([10.0, 11.0, 10.5, 12.0, 11.5, 10.8], dtype=float)
    group2 = np.array([1.0, 2.0, 1.5, 2.2, 1.2, 1.8], dtype=float)

    result = independent_t_test(group1, group2, var_name="fixation_count")

    assert result.test_name == "独立样本 t 检验"
    assert result.p_value < 0.05
    assert result.effect_size > 0.5
    assert result.ci_lower < result.ci_upper


def test_independent_t_test_not_significant() -> None:
    group1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    group2 = np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=float)

    result = independent_t_test(group1, group2, var_name="path_length")

    assert result.p_value > 0.05
    assert result.effect_size == pytest.approx(0.0)


def test_paired_t_test() -> None:
    before = np.array([10.0, 12.0, 11.0, 14.0, 13.0, 15.0, 16.0, 15.0], dtype=float)
    after = np.array([11.0, 14.0, 12.0, 15.0, 15.0, 16.0, 17.0, 17.0], dtype=float)

    result = paired_t_test(before, after, var_name="fixation_duration_mean")

    assert result.test_name == "配对样本 t 检验"
    assert result.n == len(before)
    assert result.p_value < 0.05
    assert result.ci_lower < result.ci_upper
    assert isinstance(result.conclusion, str)


def test_wilcoxon_test() -> None:
    before = np.array([2.0, 3.0, 5.0, 6.0, 8.0, 9.0, 11.0, 12.0], dtype=float)
    after = np.array([6.0, 8.0, 9.0, 11.0, 13.0, 13.0, 16.0, 17.0], dtype=float)

    result = wilcoxon_test(before, after, var_name="blink_rate_hz")

    assert result.test_name == "Wilcoxon 符号秩检验"
    assert result.p_value < 0.05
    assert np.isfinite(result.effect_size)
    assert result.effect_size_name == "r"


def test_mann_whitney_test() -> None:
    group1 = np.array([12.0, 14.0, 13.0, 15.0, 16.0, 18.0], dtype=float)
    group2 = np.array([2.0, 3.0, 4.0, 5.0, 4.5, 3.5], dtype=float)

    result = mann_whitney_test(group1, group2, var_name="saccade_count")

    assert result.test_name == "Mann-Whitney U 检验"
    assert result.p_value < 0.05
    assert np.isfinite(result.effect_size)
    assert result.effect_size_name == "r"


def test_repeated_measures_anova() -> None:
    np.random.seed(7)
    rows: list[dict[str, object]] = []
    for subject_index in range(8):
        baseline = 10.0 + subject_index * 0.25
        for condition, offset in [("baseline", 0.0), ("mid", 1.4), ("high", 2.6)]:
            rows.append(
                {
                    "subject_id": f"S{subject_index + 1:02d}",
                    "condition": condition,
                    "score": baseline + offset + float(np.random.normal(0.0, 0.12)),
                }
            )
    frame = pd.DataFrame(rows)

    result = repeated_measures_anova(frame, dv="score", within="condition", subject="subject_id")

    assert result.test_name == "重复测量方差分析"
    assert result.statistic > 1.0
    assert result.p_value < 0.05
    assert result.effect_size > 0.1
    assert result.effect_size_name == "eta_sq"


def test_descriptive_table_columns() -> None:
    frame = pd.DataFrame(
        {
            "condition": ["careful", "careful", "skim", "skim"],
            "fixation_count": [10.0, 12.0, 8.0, 9.0],
            "path_length": [110.0, 108.0, 95.0, 98.0],
        }
    )

    result = descriptive_table(frame, group_col="condition", value_cols=["fixation_count", "path_length"])

    assert {"mean", "sd", "n", "ci_lower", "ci_upper"}.issubset(result.columns)
    assert set(result["condition"]) == {"careful", "skim"}


def test_compare_conditions_auto_selection() -> None:
    np.random.seed(11)
    subjects = [f"S{index:02d}" for index in range(1, 19)]
    normal_rows: list[dict[str, object]] = []
    baseline = np.random.normal(loc=10.0, scale=1.0, size=len(subjects))
    improvement = np.random.normal(loc=0.8, scale=0.2, size=len(subjects))
    for subject_id, careful_value, delta in zip(subjects, baseline, improvement, strict=True):
        normal_rows.append({"subject_id": subject_id, "condition": "careful", "metric_a": float(careful_value)})
        normal_rows.append({"subject_id": subject_id, "condition": "skim", "metric_a": float(careful_value + delta)})
    normal_frame = pd.DataFrame(normal_rows)

    normal_result = compare_conditions(
        normal_frame,
        condition_col="condition",
        metric_cols=["metric_a"],
        paired=True,
        subject_col="subject_id",
    )

    assert normal_result.loc[0, "test_name"] == "配对样本 t 检验"

    np.random.seed(19)
    skewed_frame = pd.DataFrame(
        {
            "condition": ["careful"] * 18 + ["skim"] * 18,
            "metric_b": np.concatenate(
                [
                    np.random.exponential(scale=1.0, size=18),
                    np.random.exponential(scale=2.6, size=18),
                ]
            ),
        }
    )

    skewed_result = compare_conditions(
        skewed_frame,
        condition_col="condition",
        metric_cols=["metric_b"],
        paired=False,
    )

    assert skewed_result.loc[0, "test_name"] == "Mann-Whitney U 检验"


def test_compare_conditions_granularity_check() -> None:
    frame = pd.DataFrame(
        {
            "subject_id": ["S01", "S01", "S01", "S02", "S02"],
            "condition": ["careful", "careful", "skim", "careful", "skim"],
            "fixation_count": [10.0, 11.0, 12.0, 9.0, 8.0],
        }
    )

    with pytest.raises(ValueError, match="输入必须是 subject × condition 粒度的汇总表"):
        compare_conditions(
            frame,
            condition_col="condition",
            metric_cols=["fixation_count"],
            paired=True,
            subject_col="subject_id",
        )


def test_apa_string_format() -> None:
    group1 = np.array([9.0, 10.0, 11.0, 12.0, 10.5], dtype=float)
    group2 = np.array([4.0, 5.0, 6.0, 5.5, 4.5], dtype=float)

    result = independent_t_test(group1, group2, var_name="velocity_mean")
    apa = result.to_apa_string()

    assert isinstance(result, StatTestResult)
    assert "t(" in apa
    assert "p" in apa
    assert "Cohen's d" in apa

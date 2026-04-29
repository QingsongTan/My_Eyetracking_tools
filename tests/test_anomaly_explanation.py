from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from gaze_toolkit.report_generator import (
    FeatureAnomaly,
    FeatureAnomalyExplanation,
    _detect_feature_anomalies,
    explain_feature_anomalies,
    generate_insight_report,
)


# ── U2: 规则引擎 ──────────────────────────────────────────────────────────────

def test_no_anomalies_when_all_normal():
    features = {
        "fixation_duration_mean": 250.0,
        "blink_rate_hz": 0.25,
        "valid_ratio": 0.92,
        "saccade_amplitude_mean": 4.0,
    }
    result = _detect_feature_anomalies(features)
    assert result == []


def test_single_critical_low_anomaly():
    features = {"fixation_duration_mean": 50.0}
    result = _detect_feature_anomalies(features)
    assert len(result) == 1
    a = result[0]
    assert a.feature == "fixation_duration_mean"
    assert a.direction == "low"
    assert a.severity == "critical"


def test_single_warning_high_anomaly():
    features = {"blink_rate_hz": 0.45}
    result = _detect_feature_anomalies(features)
    assert len(result) == 1
    assert result[0].direction == "high"
    assert result[0].severity == "warning"


def test_unknown_feature_is_silently_skipped():
    features = {"nonexistent_feature_xyz": 999.0}
    result = _detect_feature_anomalies(features)
    assert result == []


def test_invalid_value_is_skipped():
    features = {"fixation_duration_mean": float("nan")}
    # nan 仍是 float，会通过 float() 但比较结果可预期
    result = _detect_feature_anomalies(features)
    # nan 的比较全为 False，不应产生异常记录
    assert result == []


# ── U4: explain_feature_anomalies ────────────────────────────────────────────

def test_rule_based_returns_without_network():
    """use_llm=False 不应触发任何网络请求。"""
    features = {"fixation_duration_mean": 50.0, "blink_rate_hz": 0.25}
    with patch("gaze_toolkit.report_generator._call_llm") as mock_call:
        result = explain_feature_anomalies(features, use_llm=False)
        mock_call.assert_not_called()
    assert result.explanation_mode == "rule_based"
    assert result.model_used is None
    assert len(result.anomalies) == 1


def test_rule_based_all_normal_gives_no_anomalies():
    features = {
        "fixation_duration_mean": 250.0,
        "blink_rate_hz": 0.25,
        "valid_ratio": 0.95,
    }
    result = explain_feature_anomalies(features, use_llm=False)
    assert result.anomalies == []
    assert "正常" in result.cognitive_state_hypothesis


def test_llm_path_sets_mode_and_model():
    """LLM 路径成功时 explanation_mode == 'llm'。"""
    import json
    llm_response = json.dumps({
        "cognitive_state": "用户存在认知过载迹象",
        "ux_recommendations": ["简化界面", "减少信息密度"],
    })
    with patch("gaze_toolkit.report_generator._call_llm", return_value=llm_response):
        result = explain_feature_anomalies(
            {"fixation_duration_mean": 50.0},
            use_llm=True,
            llm_model="gpt-4o-mini",
            api_key="fake-key",
        )
    assert result.explanation_mode == "llm"
    assert result.model_used == "gpt-4o-mini"
    assert "认知过载" in result.cognitive_state_hypothesis
    assert len(result.ux_recommendations) == 2


def test_llm_failure_falls_back_to_rule_based():
    """LLM 抛异常时自动降级，不向上传播。"""
    with patch("gaze_toolkit.report_generator._call_llm", side_effect=RuntimeError("network error")):
        result = explain_feature_anomalies(
            {"fixation_duration_mean": 50.0},
            use_llm=True,
            api_key="fake-key",
        )
    assert result.explanation_mode == "rule_based"


# ── U5: generate_insight_report 签名兼容性 ───────────────────────────────────

def _minimal_features() -> dict[str, float]:
    return {
        "fixation_duration_mean": 250.0,
        "fixation_count": 20.0,
        "saccade_count": 15.0,
        "blink_rate_hz": 0.2,
        "valid_ratio": 0.95,
        "duration_ms": 10000.0,
        "path_length": 1200.0,
        "velocity_mean": 120.0,
        "pupil_baseline": 3.5,
    }


def test_report_section_count_unchanged_when_explain_off():
    """explain_anomalies=False 时 section 数量与原始相同。"""
    features = _minimal_features()
    report_without = generate_insight_report(features, explain_anomalies=False)
    report_default = generate_insight_report(features)
    assert len(report_without.sections) == len(report_default.sections)


def test_report_has_anomaly_section_when_enabled():
    """explain_anomalies=True 时多出 anomaly_explanation section。"""
    features = _minimal_features()
    report_off = generate_insight_report(features, explain_anomalies=False)
    report_on = generate_insight_report(features, explain_anomalies=True)
    assert len(report_on.sections) == len(report_off.sections) + 1
    section_types = [s.section_type for s in report_on.sections]
    assert "anomaly_explanation" in section_types


def test_anomaly_section_appears_before_llm_insight():
    """anomaly_explanation section 位于 llm_insight 之前。"""
    import json
    llm_response = json.dumps({
        "cognitive_state": "test",
        "ux_recommendations": [],
    })
    features = _minimal_features()
    with patch("gaze_toolkit.report_generator._call_llm", return_value=llm_response):
        report = generate_insight_report(
            features,
            explain_anomalies=True,
            use_llm=True,
            api_key="fake-key",
        )
    types = [s.section_type for s in report.sections]
    assert "anomaly_explanation" in types
    assert "llm_insight" in types
    assert types.index("anomaly_explanation") < types.index("llm_insight")

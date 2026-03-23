from __future__ import annotations

import importlib
import json
from unittest.mock import patch

from gaze_toolkit.aoi import AOIMetrics
from gaze_toolkit.report_generator import (
    InsightReport,
    _interpret_metric,
    generate_insight_report,
)
from gaze_toolkit.statistics import StatTestResult


def test_template_mode_returns_report() -> None:
    report = generate_insight_report(_sample_feature_data(), use_llm=False)

    assert isinstance(report, InsightReport)
    assert report.mode == "template"
    assert report.llm_model is None
    assert report.sections


def test_template_mode_section_order() -> None:
    report = generate_insight_report(
        _sample_feature_data(),
        stat_results=[_sample_stat_result()],
        aoi_metrics=_sample_aoi_metrics(),
        use_llm=False,
    )

    assert [section.section_type for section in report.sections] == [
        "overview",
        "quality",
        "features",
        "statistics",
        "aoi",
        "recommendation",
    ]


def test_template_mode_no_stat_results() -> None:
    report = generate_insight_report(_sample_feature_data(), stat_results=None, use_llm=False)

    statistics_section = next(section for section in report.sections if section.section_type == "statistics")
    assert "暂无统计检验结果" in statistics_section.body


def test_template_mode_no_aoi_metrics() -> None:
    report = generate_insight_report(_sample_feature_data(), aoi_metrics=None, use_llm=False)

    aoi_section = next(section for section in report.sections if section.section_type == "aoi")
    assert "暂无 AOI 指标" in aoi_section.body


def test_to_markdown_output() -> None:
    report = generate_insight_report(
        _sample_feature_data(),
        stat_results=[_sample_stat_result()],
        aoi_metrics=_sample_aoi_metrics(),
        use_llm=False,
    )

    markdown = report.to_markdown()

    assert isinstance(markdown, str)
    assert "# 眼动分析洞察报告" in markdown
    for section in report.sections:
        assert f"## {section.heading}" in markdown


def test_to_dict_serializable() -> None:
    report = generate_insight_report(_sample_feature_data(), use_llm=False)

    payload = report.to_dict()

    assert payload["mode"] == "template"
    json.dumps(payload, ensure_ascii=False)


def test_interpret_metric_thresholds() -> None:
    assert "偏低" in _interpret_metric("fixation_duration_mean", 100.0)
    assert "正常" in _interpret_metric("fixation_duration_mean", 260.0)
    assert "偏高" in _interpret_metric("fixation_duration_mean", 500.0)


def test_llm_mode_fallback_on_import_error() -> None:
    original_import_module = importlib.import_module

    def fake_import(name: str, *args: object, **kwargs: object):
        if name == "openai":
            raise ImportError("missing openai")
        return original_import_module(name, *args, **kwargs)

    with patch("gaze_toolkit.report_generator.importlib.import_module", side_effect=fake_import):
        report = generate_insight_report(
            _sample_feature_data(),
            use_llm=True,
            api_key="test-key",
        )

    assert report.mode == "template"
    assert report.sections[-1].section_type == "llm_insight"
    assert "LLM 洞察生成失败" in report.sections[-1].body


def test_llm_mode_fallback_on_api_error() -> None:
    with patch("gaze_toolkit.report_generator._call_openai", side_effect=RuntimeError("api failed")):
        report = generate_insight_report(
            _sample_feature_data(),
            use_llm=True,
            api_key="test-key",
        )

    assert report.mode == "template"
    assert report.sections[-1].section_type == "llm_insight"
    assert "回退到可复现的模板模式" in report.sections[-1].body


def test_report_with_scenario_context() -> None:
    report = generate_insight_report(
        _sample_feature_data(),
        scenario_context={
            "name": "手机设置页评测",
            "product": "HarmonyOS 设置页",
            "design_type": "组内设计",
        },
        use_llm=False,
    )

    overview = next(section for section in report.sections if section.section_type == "overview")
    assert "手机设置页评测" in overview.body
    assert "HarmonyOS 设置页" in overview.body


def _sample_feature_data() -> dict[str, float]:
    return {
        "valid_ratio": 0.942,
        "fixation_count": 24.0,
        "fixation_duration_mean": 280.0,
        "saccade_count": 41.0,
        "saccade_amplitude_mean": 5.1,
        "blink_rate_hz": 0.22,
        "pupil_baseline": 3.34,
    }


def _sample_stat_result() -> StatTestResult:
    return StatTestResult(
        test_name="独立样本 t 检验",
        statistic=2.31,
        p_value=0.032,
        effect_size=0.64,
        effect_size_name="Cohen's d",
        ci_lower=0.12,
        ci_upper=1.24,
        n=24,
        conclusion="条件间差异达到统计显著，且效应量处于中等水平。",
    )


def _sample_aoi_metrics() -> dict[str, AOIMetrics]:
    return {
        "搜索框": AOIMetrics(
            aoi_name="搜索框",
            first_fixation_time=180.0,
            total_dwell_time=1260.0,
            dwell_proportion=0.38,
            fixation_count=7,
            visit_count=4,
            revisit_count=3,
            mean_fixation_duration=180.0,
        ),
        "推荐卡片": AOIMetrics(
            aoi_name="推荐卡片",
            first_fixation_time=420.0,
            total_dwell_time=840.0,
            dwell_proportion=0.24,
            fixation_count=5,
            visit_count=2,
            revisit_count=1,
            mean_fixation_duration=168.0,
        ),
    }

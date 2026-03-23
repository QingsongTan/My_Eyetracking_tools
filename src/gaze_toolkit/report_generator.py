from __future__ import annotations

import importlib
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from gaze_toolkit.aoi import AOIMetrics
from gaze_toolkit.statistics import StatTestResult

_THRESHOLDS = {
    "fixation_duration_mean": {"low": 150, "high": 400},
    "blink_rate_hz": {"low": 0.1, "high": 0.4},
    "valid_ratio": {"poor": 0.5, "good": 0.9},
    "saccade_amplitude_mean": {"low": 2.0, "high": 8.0},
}

_TITLE_MAP = {
    "zh": "眼动分析洞察报告",
    "en": "Eye-Tracking Insight Report",
}

_SECTION_HEADINGS = {
    "zh": {
        "overview": "实验概览",
        "quality": "数据质量",
        "features": "核心指标摘要",
        "statistics": "统计检验结果",
        "aoi": "AOI 分析摘要",
        "recommendation": "建议",
        "llm_insight": "AI 洞察",
    },
    "en": {
        "overview": "Study Overview",
        "quality": "Data Quality",
        "features": "Feature Summary",
        "statistics": "Statistical Results",
        "aoi": "AOI Summary",
        "recommendation": "Recommendations",
        "llm_insight": "AI Insight",
    },
}

_METRIC_LABELS = {
    "zh": {
        "valid_ratio": "有效追踪率",
        "fixation_count": "注视次数",
        "fixation_duration_mean": "平均注视时长",
        "saccade_count": "扫视次数",
        "saccade_amplitude_mean": "平均扫视幅度",
        "blink_rate_hz": "眨眼频率",
        "pupil_baseline": "瞳孔基线",
    },
    "en": {
        "valid_ratio": "Valid tracking ratio",
        "fixation_count": "Fixation count",
        "fixation_duration_mean": "Mean fixation duration",
        "saccade_count": "Saccade count",
        "saccade_amplitude_mean": "Mean saccade amplitude",
        "blink_rate_hz": "Blink rate",
        "pupil_baseline": "Pupil baseline",
    },
}

_LLM_SYSTEM_PROMPT = """你是一位资深的人因工程研究员，擅长眼动数据分析和用户体验评估。
请基于以下眼动分析数据，撰写专业的研究洞察段落。

要求：
1. 使用学术但易读的语言
2. 指出数据中最值得关注的发现
3. 给出可操作的设计改进建议
4. 如果有统计检验结果，解读其实际意义（不只是看 p 值）
5. 控制在 {max_tokens} 字以内
"""


@dataclass
class ReportSection:
    """报告段落。"""

    heading: str
    body: str
    section_type: str


@dataclass
class InsightReport:
    """分析报告输出。"""

    title: str
    sections: list[ReportSection]
    generated_at: str
    mode: str
    llm_model: str | None

    def to_markdown(self) -> str:
        """输出完整 Markdown 文本。"""
        lines = [f"# {self.title}", ""]
        lines.append(f"- generated_at: {self.generated_at}")
        lines.append(f"- mode: {self.mode}")
        if self.llm_model:
            lines.append(f"- llm_model: {self.llm_model}")
        lines.append("")

        for section in self.sections:
            lines.append(f"## {section.heading}")
            lines.append(section.body.strip())
            lines.append("")
        return "\n".join(lines).strip() + "\n"

    def to_dict(self) -> dict[str, Any]:
        """输出 JSON-serializable 字典。"""
        return {
            "title": self.title,
            "sections": [asdict(section) for section in self.sections],
            "generated_at": self.generated_at,
            "mode": self.mode,
            "llm_model": self.llm_model,
        }


def generate_insight_report(
    feature_data: dict[str, float],
    *,
    scenario_context: dict[str, Any] | None = None,
    stat_results: list[StatTestResult] | None = None,
    aoi_metrics: dict[str, AOIMetrics] | None = None,
    quality_grade: str | None = None,
    use_llm: bool = False,
    llm_backend: str = "openai",
    llm_model: str = "gpt-4o-mini",
    api_key: str | None = None,
    language: str = "zh",
    max_tokens: int = 2000,
) -> InsightReport:
    """
    基于分析结果生成人因研究洞察报告。

    默认输出纯模板报告；启用 LLM 模式后，会在模板报告基础上追加自由文本洞察。
    """
    lang = _normalize_language(language)
    template_report = _build_template_report(
        feature_data=feature_data,
        scenario_context=scenario_context,
        stat_results=stat_results,
        aoi_metrics=aoi_metrics,
        quality_grade=quality_grade,
        language=lang,
    )
    if not use_llm:
        return template_report

    prompt = _build_llm_prompt(
        template_report=template_report,
        feature_data=feature_data,
        scenario_context=scenario_context,
        stat_results=stat_results,
        aoi_metrics=aoi_metrics,
        quality_grade=quality_grade,
        language=lang,
    )

    try:
        llm_text = _call_llm(
            prompt,
            backend=llm_backend,
            model=llm_model,
            api_key=api_key,
            max_tokens=max_tokens,
        )
    except Exception:
        failure_section = ReportSection(
            heading=_heading_for("llm_insight", lang),
            body=_llm_failure_message(lang),
            section_type="llm_insight",
        )
        return InsightReport(
            title=template_report.title,
            sections=[*template_report.sections, failure_section],
            generated_at=template_report.generated_at,
            mode="template",
            llm_model=None,
        )

    llm_section = ReportSection(
        heading=_heading_for("llm_insight", lang),
        body=llm_text.strip(),
        section_type="llm_insight",
    )
    return InsightReport(
        title=template_report.title,
        sections=[*template_report.sections, llm_section],
        generated_at=template_report.generated_at,
        mode="llm",
        llm_model=llm_model,
    )


def _build_template_report(
    *,
    feature_data: dict[str, float],
    scenario_context: dict[str, Any] | None,
    stat_results: list[StatTestResult] | None,
    aoi_metrics: dict[str, AOIMetrics] | None,
    quality_grade: str | None,
    language: str,
) -> InsightReport:
    sections = [
        ReportSection(
            heading=_heading_for("overview", language),
            body=_build_overview_body(scenario_context, language),
            section_type="overview",
        ),
        ReportSection(
            heading=_heading_for("quality", language),
            body=_build_quality_body(feature_data, quality_grade, language),
            section_type="quality",
        ),
        ReportSection(
            heading=_heading_for("features", language),
            body=_build_feature_body(feature_data, language),
            section_type="features",
        ),
        ReportSection(
            heading=_heading_for("statistics", language),
            body=_build_statistics_body(stat_results, language),
            section_type="statistics",
        ),
        ReportSection(
            heading=_heading_for("aoi", language),
            body=_build_aoi_body(aoi_metrics, language),
            section_type="aoi",
        ),
        ReportSection(
            heading=_heading_for("recommendation", language),
            body=_build_recommendation_body(feature_data, aoi_metrics, language),
            section_type="recommendation",
        ),
    ]
    return InsightReport(
        title=_TITLE_MAP[language],
        sections=sections,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        mode="template",
        llm_model=None,
    )


def _build_overview_body(scenario_context: dict[str, Any] | None, language: str) -> str:
    context = scenario_context or {}
    name = _as_text(context.get("name") or context.get("scenario_name"))
    product = _as_text(context.get("product"))
    design_type = _as_text(
        context.get("design_type")
        or context.get("design")
        or context.get("research_design_type")
    )

    if language == "en":
        if not any([name, product, design_type]):
            return "This report summarizes a single eye-tracking analysis and highlights the most actionable attention and usability patterns."
        parts = ["This report is based on"]
        parts.append(f'the "{name}" scenario' if name else "a single eye-tracking analysis")
        if product:
            parts.append(f"for the product {product}")
        if design_type:
            parts.append(f"under a {design_type} research design")
        return " ".join(parts).strip() + "."

    if not any([name, product, design_type]):
        return "本次报告面向单次眼动分析，聚焦整体注意分配、数据质量与可设计改进点。"

    segments = []
    if name:
        segments.append(f"本次分析基于“{name}”场景")
    else:
        segments.append("本次分析基于单次眼动分析")
    if product:
        segments.append(f"产品对象为{product}")
    if design_type:
        segments.append(f"研究设计类型为{design_type}")
    return "，".join(segments) + "。"


def _build_quality_body(
    feature_data: dict[str, float],
    quality_grade: str | None,
    language: str,
) -> str:
    valid_ratio = _safe_float(feature_data.get("valid_ratio"))
    grade = quality_grade or _quality_grade_from_ratio(valid_ratio, language)

    if valid_ratio is None:
        if language == "en":
            return f'Quality grade is "{grade}", but no valid tracking ratio was provided.'
        return f"当前质量等级为“{grade}”，但未提供有效追踪率，建议结合原始采样日志复核。"

    ratio_text = _format_metric_value("valid_ratio", valid_ratio, language)
    if valid_ratio < _THRESHOLDS["valid_ratio"]["poor"]:
        tail = (
            "The data loss is substantial, so downstream interpretation should be conservative."
            if language == "en"
            else "数据缺失较多，后续结论应谨慎解释。"
        )
    elif valid_ratio >= _THRESHOLDS["valid_ratio"]["good"]:
        tail = (
            "Tracking stability is strong enough to support downstream interpretation."
            if language == "en"
            else "追踪稳定性较好，可支持后续行为与界面解读。"
        )
    else:
        tail = (
            "The data is usable overall, although a small amount of missingness remains."
            if language == "en"
            else "整体数据可用，但仍存在一定缺失，需要结合任务上下文解读。"
        )

    if language == "en":
        return f'{_metric_label("valid_ratio", language)} is {ratio_text}, with quality grade "{grade}". {tail}'
    return f'{_metric_label("valid_ratio", language)}为{ratio_text}，质量等级为“{grade}”。{tail}'


def _build_feature_body(feature_data: dict[str, float], language: str) -> str:
    lines: list[str] = []
    fixation_count = _safe_float(feature_data.get("fixation_count"))
    if fixation_count is not None:
        if language == "en":
            lines.append(
                f"- {_metric_label('fixation_count', language)}: {fixation_count:.0f}. "
                "This reflects how often the participant established stable visual anchors during the task."
            )
        else:
            lines.append(
                f"- {_metric_label('fixation_count', language)}：{fixation_count:.0f} 次，"
                "反映任务过程中形成稳定视觉停留的频次。"
            )

    for metric_name in (
        "fixation_duration_mean",
        "saccade_count",
        "blink_rate_hz",
        "pupil_baseline",
        "saccade_amplitude_mean",
    ):
        metric_value = _safe_float(feature_data.get(metric_name))
        if metric_value is None:
            continue

        if metric_name in _THRESHOLDS:
            lines.append(f"- {_interpret_metric(metric_name, metric_value, language=language)}")
            continue

        if metric_name == "saccade_count":
            if language == "en":
                lines.append(
                    f"- {_metric_label(metric_name, language)}: {metric_value:.0f}. "
                    "Higher counts usually indicate more visual relocations across the interface."
                )
            else:
                lines.append(
                    f"- {_metric_label(metric_name, language)}：{metric_value:.0f} 次，"
                    "较高的扫视频次通常意味着界面中的视线切换更频繁。"
                )
        elif metric_name == "pupil_baseline":
            if language == "en":
                lines.append(
                    f"- {_metric_label(metric_name, language)}: {metric_value:.2f}. "
                    "This value can serve as a reference for later workload-related pupil changes."
                )
            else:
                lines.append(
                    f"- {_metric_label(metric_name, language)}：{metric_value:.2f}，"
                    "可作为后续认知负荷变化的基线参照。"
                )

    if lines:
        return "\n".join(lines)
    if language == "en":
        return "No core feature metrics were provided."
    return "未提供可用于解释的核心特征指标。"


def _build_statistics_body(stat_results: list[StatTestResult] | None, language: str) -> str:
    if not stat_results:
        return "No statistical test results are available yet." if language == "en" else "暂无统计检验结果。"

    lines = []
    for result in stat_results:
        apa_text = result.to_apa_string()
        conclusion = result.conclusion.strip()
        if language == "en":
            lines.append(f"- {result.test_name}: {apa_text}. {conclusion}")
        else:
            lines.append(f"- {result.test_name}：{apa_text}。{conclusion}")
    return "\n".join(lines)


def _build_aoi_body(aoi_metrics: dict[str, AOIMetrics] | None, language: str) -> str:
    if not aoi_metrics:
        return (
            "No AOI-level metrics are available, so regional attention allocation cannot be summarized."
            if language == "en"
            else "暂无 AOI 指标，暂时无法进一步总结区域级注意分配。"
        )

    metrics = list(aoi_metrics.values())
    top_dwell = max(metrics, key=lambda item: item.total_dwell_time)
    fastest_candidates = [item for item in metrics if item.first_fixation_time is not None]
    fastest = min(fastest_candidates, key=lambda item: item.first_fixation_time) if fastest_candidates else None
    top_revisit = max(metrics, key=lambda item: item.revisit_count)

    sentences: list[str] = []
    if language == "en":
        sentences.append(
            f'Attention is concentrated most on "{top_dwell.aoi_name}" '
            f'({top_dwell.total_dwell_time:.1f} ms dwell, {top_dwell.dwell_proportion:.1%} of total dwell).'
        )
        if fastest is not None:
            sentences.append(
                f'The earliest first fixation landed on "{fastest.aoi_name}" '
                f'(TTFF {fastest.first_fixation_time:.1f} ms).'
            )
        if top_revisit.revisit_count > 0:
            sentences.append(
                f'"{top_revisit.aoi_name}" shows the highest revisit count ({top_revisit.revisit_count}), '
                "which may indicate repeated confirmation or local comprehension friction."
            )
        return " ".join(sentences)

    sentences.append(
        f'用户注意力主要集中在“{top_dwell.aoi_name}”区域，累计驻留 {top_dwell.total_dwell_time:.1f} ms，'
        f'约占总驻留的 {top_dwell.dwell_proportion:.1%}。'
    )
    if fastest is not None:
        sentences.append(
            f'首次注意最早落在“{fastest.aoi_name}”区域（TTFF {fastest.first_fixation_time:.1f} ms）。'
        )
    if top_revisit.revisit_count > 0:
        sentences.append(
            f'“{top_revisit.aoi_name}”的回视次数最高（{top_revisit.revisit_count} 次），'
            "可能存在信息核对或理解阻塞。"
        )
    return "".join(sentences)


def _build_recommendation_body(
    feature_data: dict[str, float],
    aoi_metrics: dict[str, AOIMetrics] | None,
    language: str,
) -> str:
    recommendations: list[str] = []

    valid_ratio = _safe_float(feature_data.get("valid_ratio"))
    fixation_duration = _safe_float(feature_data.get("fixation_duration_mean"))
    blink_rate = _safe_float(feature_data.get("blink_rate_hz"))
    saccade_amplitude = _safe_float(feature_data.get("saccade_amplitude_mean"))

    if valid_ratio is not None and valid_ratio < _THRESHOLDS["valid_ratio"]["poor"]:
        recommendations.append(
            "先优化采集环境与校准流程，避免数据缺失过高影响后续解释。"
            if language == "zh"
            else "Improve calibration and capture conditions first, because severe data loss weakens downstream interpretation."
        )

    if fixation_duration is not None and fixation_duration > _THRESHOLDS["fixation_duration_mean"]["high"]:
        recommendations.append(
            "平均注视时长偏高，建议简化信息层级、减少需要长时间停留理解的界面元素。"
            if language == "zh"
            else "Mean fixation duration is elevated; consider simplifying the information hierarchy and reducing elements that require prolonged inspection."
        )
    elif fixation_duration is not None and fixation_duration < _THRESHOLDS["fixation_duration_mean"]["low"]:
        recommendations.append(
            "平均注视时长偏低，建议强化关键内容的视觉强调，避免用户过快扫过重要信息。"
            if language == "zh"
            else "Mean fixation duration is low; strengthen visual emphasis for critical information so it is not skimmed too quickly."
        )

    if blink_rate is not None and blink_rate > _THRESHOLDS["blink_rate_hz"]["high"]:
        recommendations.append(
            "眨眼频率偏高，可优先检查任务负荷、文本密度或操作步骤是否过于紧张。"
            if language == "zh"
            else "Blink rate is elevated; review task load, text density, or interaction pacing for potential overload."
        )

    if saccade_amplitude is not None and saccade_amplitude > _THRESHOLDS["saccade_amplitude_mean"]["high"]:
        recommendations.append(
            "扫视跨度偏大，建议加强版面分组与关键控件聚合，降低跨区域搜索成本。"
            if language == "zh"
            else "Saccade amplitude is high; improve grouping and cluster key controls to reduce cross-screen search cost."
        )

    if aoi_metrics:
        highest_revisit = max(aoi_metrics.values(), key=lambda item: item.revisit_count)
        if highest_revisit.revisit_count > 0:
            recommendations.append(
                f'优先优化“{highest_revisit.aoi_name}”区域的信息层级或标签表达，减少反复确认。'
                if language == "zh"
                else f'Prioritize clearer hierarchy or labeling in "{highest_revisit.aoi_name}" to reduce repeated checking.'
            )

        slow_entry_candidates = [item for item in aoi_metrics.values() if item.first_fixation_time is not None]
        if slow_entry_candidates:
            slowest_entry = max(slow_entry_candidates, key=lambda item: item.first_fixation_time)
            if (slowest_entry.first_fixation_time or 0.0) > 1500:
                recommendations.append(
                    f'“{slowest_entry.aoi_name}”较晚才获得首次注意，建议提升其入口可见性或前置提示。'
                    if language == "zh"
                    else f'"{slowest_entry.aoi_name}" receives attention late; increase its visual discoverability or introduce earlier cues.'
                )

    if len(recommendations) < 2:
        recommendations.append(
            "建议结合任务完成率、主观量表或访谈结果做交叉验证，以确认眼动模式背后的真实原因。"
            if language == "zh"
            else "Triangulate these findings with task success, subjective ratings, or interviews to validate the underlying causes."
        )
    if len(recommendations) < 3:
        recommendations.append(
            "建议在下一轮迭代中保留关键视觉层级的一致性，再针对高摩擦区域做定点优化。"
            if language == "zh"
            else "Keep the core visual hierarchy stable in the next iteration, then target the highest-friction areas for refinement."
        )

    return "\n".join(f"- {item}" for item in recommendations[:3])


def _build_llm_prompt(
    *,
    template_report: InsightReport,
    feature_data: dict[str, float],
    scenario_context: dict[str, Any] | None,
    stat_results: list[StatTestResult] | None,
    aoi_metrics: dict[str, AOIMetrics] | None,
    quality_grade: str | None,
    language: str,
) -> str:
    payload = {
        "language": language,
        "quality_grade": quality_grade,
        "scenario_context": scenario_context or {},
        "feature_data": feature_data,
        "stat_results": [
            {
                "test_name": result.test_name,
                "apa": result.to_apa_string(),
                "conclusion": result.conclusion,
                "p_value": result.p_value,
                "effect_size": result.effect_size,
                "effect_size_name": result.effect_size_name,
            }
            for result in (stat_results or [])
        ],
        "aoi_metrics": {
            name: {
                "first_fixation_time": metric.first_fixation_time,
                "total_dwell_time": metric.total_dwell_time,
                "dwell_proportion": metric.dwell_proportion,
                "fixation_count": metric.fixation_count,
                "visit_count": metric.visit_count,
                "revisit_count": metric.revisit_count,
                "mean_fixation_duration": metric.mean_fixation_duration,
            }
            for name, metric in (aoi_metrics or {}).items()
        },
        "template_report_markdown": template_report.to_markdown(),
    }

    if language == "en":
        return (
            "Please write a concise but professional eye-tracking insight paragraph in English.\n"
            "Do not invent data that is not present.\n\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )
    return (
        "请基于以下结构化数据和模板报告，在不虚构未提供结果的前提下，"
        "输出一段更像研究者撰写的眼动洞察总结。\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def _call_llm(
    prompt: str,
    *,
    backend: str,
    model: str,
    api_key: str | None,
    max_tokens: int,
) -> str:
    """统一的 LLM 调用入口。"""
    backend_name = backend.strip().lower()
    if backend_name == "openai":
        return _call_openai(prompt, model=model, api_key=api_key, max_tokens=max_tokens)
    if backend_name == "anthropic":
        return _call_anthropic(prompt, model=model, api_key=api_key, max_tokens=max_tokens)
    if backend_name == "custom":
        return _call_custom(prompt, model=model, api_key=api_key, max_tokens=max_tokens)
    raise ValueError(f"Unsupported llm backend: {backend}")


def _call_openai(prompt: str, *, model: str, api_key: str | None, max_tokens: int) -> str:
    try:
        openai_module = importlib.import_module("openai")
    except ImportError as exc:
        raise ImportError("OpenAI backend requires `openai>=1.0`. Please run: pip install openai>=1.0") from exc

    resolved_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_key:
        raise RuntimeError("OpenAI API key is missing. Provide api_key or set OPENAI_API_KEY.")

    client_cls = getattr(openai_module, "OpenAI", None)
    if client_cls is None:
        raise RuntimeError("Installed openai package does not expose `OpenAI`.")

    client = client_cls(api_key=resolved_key, timeout=30.0)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _LLM_SYSTEM_PROMPT.format(max_tokens=max_tokens)},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content
    text = _coerce_text(content)
    if not text:
        raise RuntimeError("OpenAI response did not contain text content.")
    return text


def _call_anthropic(prompt: str, *, model: str, api_key: str | None, max_tokens: int) -> str:
    try:
        anthropic_module = importlib.import_module("anthropic")
    except ImportError as exc:
        raise ImportError("Anthropic backend requires `anthropic`. Please run: pip install anthropic") from exc

    resolved_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not resolved_key:
        raise RuntimeError("Anthropic API key is missing. Provide api_key or set ANTHROPIC_API_KEY.")

    client_cls = getattr(anthropic_module, "Anthropic", None)
    if client_cls is None:
        raise RuntimeError("Installed anthropic package does not expose `Anthropic`.")

    client = client_cls(api_key=resolved_key, timeout=30.0)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=_LLM_SYSTEM_PROMPT.format(max_tokens=max_tokens),
        messages=[{"role": "user", "content": prompt}],
    )
    text_parts = [block.text for block in getattr(response, "content", []) if hasattr(block, "text")]
    text = "\n".join(part.strip() for part in text_parts if part and part.strip())
    if not text:
        raise RuntimeError("Anthropic response did not contain text content.")
    return text


def _call_custom(prompt: str, *, model: str, api_key: str | None, max_tokens: int) -> str:
    try:
        requests_module = importlib.import_module("requests")
    except ImportError as exc:
        raise ImportError("Custom backend requires `requests`. Please run: pip install requests") from exc

    endpoint = os.getenv("GAZE_LLM_ENDPOINT")
    if not endpoint:
        raise RuntimeError("Custom backend requires GAZE_LLM_ENDPOINT to be set.")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    response = requests_module.post(
        endpoint,
        headers=headers,
        json={
            "model": model,
            "system_prompt": _LLM_SYSTEM_PROMPT.format(max_tokens=max_tokens),
            "prompt": prompt,
            "max_tokens": max_tokens,
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    text = _extract_custom_text(payload)
    if not text:
        raise RuntimeError("Custom backend response did not contain text content.")
    return text


def _extract_custom_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload.strip()
    if isinstance(payload, dict):
        for key in ("text", "content", "output"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    text = _coerce_text(content)
                    if text:
                        return text
                content = first.get("content")
                text = _coerce_text(content)
                if text:
                    return text
    return ""


def _interpret_metric(name: str, value: float, language: str = "zh") -> str:
    numeric = _safe_float(value)
    lang = _normalize_language(language)
    if numeric is None:
        return (
            f"{_metric_label(name, lang)} is unavailable."
            if lang == "en"
            else f"{_metric_label(name, lang)}暂无可解释数值。"
        )

    if name == "valid_ratio":
        low_threshold = _THRESHOLDS[name]["poor"]
        high_threshold = _THRESHOLDS[name]["good"]
    else:
        metric_threshold = _THRESHOLDS.get(name)
        if not metric_threshold:
            if lang == "en":
                return f"{_metric_label(name, lang)} is {numeric:.2f}, with no built-in heuristic threshold."
            return f"{_metric_label(name, lang)}为{numeric:.2f}，当前未内置通用阈值。"
        low_threshold = metric_threshold["low"]
        high_threshold = metric_threshold["high"]

    if numeric < low_threshold:
        level = "偏低" if lang == "zh" else "low"
        meaning = _metric_meaning(name, "low", lang)
    elif numeric > high_threshold:
        level = "偏高" if lang == "zh" else "high"
        meaning = _metric_meaning(name, "high", lang)
    else:
        level = "正常" if lang == "zh" else "normal"
        meaning = _metric_meaning(name, "normal", lang)

    value_text = _format_metric_value(name, numeric, lang)
    if lang == "en":
        return f"{_metric_label(name, lang)} is {value_text}, which is {level}. {meaning}"
    return f"{_metric_label(name, lang)}为{value_text}，{level}。{meaning}"


def _metric_meaning(name: str, level: str, language: str) -> str:
    if language == "en":
        messages = {
            "fixation_duration_mean": {
                "low": "Visual processing appears brief and possibly scan-oriented.",
                "normal": "Visual processing tempo is relatively balanced.",
                "high": "Participants may need more time to interpret local content.",
            },
            "blink_rate_hz": {
                "low": "Task pacing appears stable and interruption is limited.",
                "normal": "Blink rhythm remains within a typical observation band.",
                "high": "Fatigue or cognitive load may be elevated.",
            },
            "valid_ratio": {
                "low": "Data quality is weak and interpretation should stay cautious.",
                "normal": "Data quality is usable with some missingness.",
                "high": "Tracking quality is stable enough for interpretation.",
            },
            "saccade_amplitude_mean": {
                "low": "Visual search stays within a relatively compact area.",
                "normal": "Search span is within a moderate range.",
                "high": "Attention shifts cover a wide spatial range across the interface.",
            },
        }
    else:
        messages = {
            "fixation_duration_mean": {
                "low": "视觉加工停留较短，可能更偏向快速浏览。",
                "normal": "信息提取节奏相对平衡。",
                "high": "用户可能需要更长时间理解局部内容。",
            },
            "blink_rate_hz": {
                "low": "任务节奏整体稳定，中断较少。",
                "normal": "眨眼节律处于常见观察范围。",
                "high": "可能存在疲劳上升或认知负荷偏高的情况。",
            },
            "valid_ratio": {
                "low": "数据质量偏弱，解读时需保持谨慎。",
                "normal": "数据整体可用，但仍有一定缺失。",
                "high": "追踪质量稳定，适合进一步解释。",
            },
            "saccade_amplitude_mean": {
                "low": "视觉搜索主要集中在较紧凑的局部区域。",
                "normal": "视线搜索跨度处于中等范围。",
                "high": "注意力切换覆盖了更大的界面空间。",
            },
        }
    return messages.get(name, {}).get(level, "")


def _metric_label(name: str, language: str) -> str:
    return _METRIC_LABELS[language].get(name, name)


def _heading_for(section_type: str, language: str) -> str:
    return _SECTION_HEADINGS[language][section_type]


def _quality_grade_from_ratio(valid_ratio: float | None, language: str) -> str:
    if valid_ratio is None:
        return "未知" if language == "zh" else "unknown"
    if valid_ratio >= 0.9:
        return "优" if language == "zh" else "excellent"
    if valid_ratio >= 0.75:
        return "良" if language == "zh" else "good"
    if valid_ratio >= 0.5:
        return "可用" if language == "zh" else "usable"
    return "建议剔除" if language == "zh" else "poor"


def _format_metric_value(name: str, value: float, language: str) -> str:
    if name == "valid_ratio":
        return f"{value:.1%}"
    if name == "fixation_duration_mean":
        return f"{value:.1f} ms"
    if name == "blink_rate_hz":
        return f"{value:.2f} Hz"
    if name == "saccade_amplitude_mean":
        return f"{value:.2f} deg"
    if name in {"fixation_count", "saccade_count"}:
        return f"{value:.0f}"
    if name == "pupil_baseline":
        return f"{value:.2f}"
    return f"{value:.2f}"


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text if text and text.lower() != "nan" else ""


def _normalize_language(language: str) -> str:
    return "en" if str(language).strip().lower() == "en" else "zh"


def _llm_failure_message(language: str) -> str:
    if language == "en":
        return (
            "LLM insight generation failed, so the report has gracefully fallen back to the reproducible template mode below."
        )
    return "LLM 洞察生成失败，系统已自动回退到可复现的模板模式，以下内容为规则化报告。"


def _coerce_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, str) and item.strip():
                text_parts.append(item.strip())
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    text_parts.append(text.strip())
            elif hasattr(item, "text"):
                text = getattr(item, "text")
                if isinstance(text, str) and text.strip():
                    text_parts.append(text.strip())
        return "\n".join(text_parts)
    return ""

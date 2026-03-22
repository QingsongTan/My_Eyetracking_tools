"""产品评测研究场景模板模块。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from gaze_toolkit.aoi import AOI, define_aoi

SCENARIOS_DIR = Path(__file__).resolve().parent.parent.parent / "configs" / "scenarios"


@dataclass
class AOIRegion:
    """场景任务中的矩形 AOI 区域定义。"""

    name: str
    region: tuple[float, float, float, float]


@dataclass
class ScenarioTask:
    """单个研究任务与其 AOI 配置。"""

    id: str
    description: str
    aoi_regions: list[AOIRegion] = field(default_factory=list)


@dataclass
class ResearchDesign:
    """研究设计与指标说明。"""

    type: str
    iv: str
    dv: dict[str, list[str]]
    sample_size: str
    counterbalancing: str


@dataclass
class ScenarioTemplate:
    """完整的产品评测研究场景模板。"""

    name: str
    product: str
    research_goal: str
    research_design: ResearchDesign
    tasks: list[ScenarioTask]
    analysis_plan: dict[str, list[str]]
    huawei_relevance: str


def list_scenarios() -> list[str]:
    """列出所有可用场景的名称（文件 stem）。"""
    if not SCENARIOS_DIR.exists():
        return []
    return sorted(path.stem for path in SCENARIOS_DIR.glob("*.yaml"))


def load_scenario(name: str) -> ScenarioTemplate:
    """从 YAML 文件加载场景模板。"""
    scenario_path = SCENARIOS_DIR / f"{name}.yaml"
    if not scenario_path.exists():
        raise FileNotFoundError(f"未找到场景 `{name}`。搜索路径：{scenario_path}")

    with scenario_path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}

    if not isinstance(payload, dict):
        raise ValueError(f"场景 `{name}` 的 YAML 结构无效。搜索路径：{scenario_path}")

    scenario_meta = payload.get("scenario")
    research_meta = payload.get("research_design")
    task_items = payload.get("tasks", [])
    analysis_plan = payload.get("analysis_plan", {})
    huawei_relevance = payload.get("huawei_relevance", "")

    if not isinstance(scenario_meta, dict):
        raise ValueError(f"场景 `{name}` 缺少 `scenario` 配置。搜索路径：{scenario_path}")
    if not isinstance(research_meta, dict):
        raise ValueError(f"场景 `{name}` 缺少 `research_design` 配置。搜索路径：{scenario_path}")
    if not isinstance(task_items, list):
        raise ValueError(f"场景 `{name}` 的 `tasks` 必须是列表。搜索路径：{scenario_path}")
    if not isinstance(analysis_plan, dict):
        raise ValueError(f"场景 `{name}` 的 `analysis_plan` 必须是字典。搜索路径：{scenario_path}")

    tasks = [_parse_task(task_item, scenario_name=name, scenario_path=scenario_path) for task_item in task_items]
    research_design = _parse_research_design(research_meta, scenario_name=name, scenario_path=scenario_path)

    return ScenarioTemplate(
        name=str(scenario_meta.get("name", name)),
        product=str(scenario_meta.get("product", "")),
        research_goal=str(scenario_meta.get("research_goal", "")),
        research_design=research_design,
        tasks=tasks,
        analysis_plan=_normalize_plan_items(analysis_plan),
        huawei_relevance=str(huawei_relevance),
    )


def get_scenario_aois(scenario: ScenarioTemplate, task_id: str) -> list[AOI]:
    """从场景模板提取指定任务的 AOI 列表。"""
    for task in scenario.tasks:
        if task.id == task_id:
            return [define_aoi(region.name, *region.region) for region in task.aoi_regions]
    raise ValueError(f"场景 `{scenario.name}` 中不存在任务 `{task_id}`。")


def _parse_task(task_item: object, *, scenario_name: str, scenario_path: Path) -> ScenarioTask:
    if not isinstance(task_item, dict):
        raise ValueError(f"场景 `{scenario_name}` 的任务定义无效。搜索路径：{scenario_path}")

    task_id = str(task_item.get("id", "")).strip()
    description = str(task_item.get("description", "")).strip()
    region_items = task_item.get("aoi_regions", [])
    if not task_id or not description:
        raise ValueError(f"场景 `{scenario_name}` 的任务缺少 id 或 description。搜索路径：{scenario_path}")
    if not isinstance(region_items, list):
        raise ValueError(
            f"场景 `{scenario_name}` 的任务 `{task_id}` 中 `aoi_regions` 必须是列表。搜索路径：{scenario_path}"
        )

    regions: list[AOIRegion] = []
    for region_item in region_items:
        regions.append(_parse_region(region_item, scenario_name=scenario_name, task_id=task_id, scenario_path=scenario_path))
    return ScenarioTask(id=task_id, description=description, aoi_regions=regions)


def _parse_region(region_item: object, *, scenario_name: str, task_id: str, scenario_path: Path) -> AOIRegion:
    if not isinstance(region_item, dict):
        raise ValueError(
            f"场景 `{scenario_name}` 的任务 `{task_id}` 中 AOI 定义无效。搜索路径：{scenario_path}"
        )

    name = str(region_item.get("name", "")).strip()
    coordinates = region_item.get("region")
    if not name or not isinstance(coordinates, list) or len(coordinates) != 4:
        raise ValueError(
            f"场景 `{scenario_name}` 的任务 `{task_id}` 中 AOI `{name or '<unknown>'}` 坐标无效。"
            f" 搜索路径：{scenario_path}"
        )

    x_min, y_min, x_max, y_max = (float(value) for value in coordinates)
    return AOIRegion(name=name, region=(x_min, y_min, x_max, y_max))


def _parse_research_design(research_meta: dict[str, object], *, scenario_name: str, scenario_path: Path) -> ResearchDesign:
    dv_meta = research_meta.get("dv", {})
    if not isinstance(dv_meta, dict):
        raise ValueError(f"场景 `{scenario_name}` 的 `research_design.dv` 必须是字典。搜索路径：{scenario_path}")

    normalized_dv: dict[str, list[str]] = {}
    for key, values in dv_meta.items():
        if isinstance(values, list):
            normalized_dv[str(key)] = [str(value) for value in values]
        else:
            raise ValueError(
                f"场景 `{scenario_name}` 的 `research_design.dv.{key}` 必须是列表。搜索路径：{scenario_path}"
            )

    return ResearchDesign(
        type=str(research_meta.get("type", "")),
        iv=str(research_meta.get("iv", "")),
        dv=normalized_dv,
        sample_size=str(research_meta.get("sample_size", "")),
        counterbalancing=str(research_meta.get("counterbalancing", "")),
    )


def _normalize_plan_items(analysis_plan: dict[str, object]) -> dict[str, list[str]]:
    normalized: dict[str, list[str]] = {}
    for key, values in analysis_plan.items():
        if isinstance(values, list):
            normalized[str(key)] = [str(value) for value in values]
        else:
            normalized[str(key)] = [str(values)]
    return normalized

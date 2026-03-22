from __future__ import annotations

import pytest

from gaze_toolkit.aoi import AOI
from gaze_toolkit.scenarios import (
    ScenarioTemplate,
    get_scenario_aois,
    list_scenarios,
    load_scenario,
)


def test_list_scenarios_returns_names() -> None:
    scenario_names = list_scenarios()

    assert scenario_names
    assert "phone_settings_evaluation" in scenario_names


def test_load_scenario_phone_settings() -> None:
    scenario = load_scenario("phone_settings_evaluation")

    assert isinstance(scenario, ScenarioTemplate)
    assert scenario.name == "手机设置页信息架构评测"
    assert scenario.product == "手机"
    assert len(scenario.tasks) == 3
    assert scenario.tasks[0].id == "T1"


def test_scenario_has_tasks_with_aois() -> None:
    scenario = load_scenario("phone_settings_evaluation")

    assert scenario.tasks
    assert scenario.tasks[0].aoi_regions
    assert scenario.tasks[0].aoi_regions[0].name == "顶部搜索栏"


def test_get_scenario_aois_returns_aoi_objects() -> None:
    scenario = load_scenario("phone_settings_evaluation")

    aois = get_scenario_aois(scenario, "T1")

    assert aois
    assert all(isinstance(aoi, AOI) for aoi in aois)
    assert all(aoi.region_type == "rectangle" for aoi in aois)


def test_load_nonexistent_scenario_raises() -> None:
    with pytest.raises(FileNotFoundError, match="不存在|未找到场景|搜索路径"):
        load_scenario("not_a_real_scenario")

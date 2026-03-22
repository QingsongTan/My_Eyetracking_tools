from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib.path import Path


@dataclass
class AOI:
    """兴趣区域定义。"""

    name: str
    region: object
    region_type: str = "rectangle"


@dataclass
class AOIMetrics:
    """单个 AOI 的核心指标。"""

    aoi_name: str
    first_fixation_time: Optional[float]
    total_dwell_time: float
    dwell_proportion: float
    fixation_count: int
    visit_count: int
    revisit_count: int
    mean_fixation_duration: float


def define_aoi(name: str, x_min: float, y_min: float, x_max: float, y_max: float) -> AOI:
    """定义矩形 AOI。"""
    x0, x1 = sorted((float(x_min), float(x_max)))
    y0, y1 = sorted((float(y_min), float(y_max)))
    return AOI(name=name, region=(x0, y0, x1, y1), region_type="rectangle")


def define_polygon_aoi(name: str, vertices: list[tuple[float, float]]) -> AOI:
    """定义多边形 AOI。"""
    if len(vertices) < 3:
        raise ValueError("Polygon AOI requires at least 3 vertices.")
    normalized = [(float(x), float(y)) for x, y in vertices]
    return AOI(name=name, region=normalized, region_type="polygon")


def assign_fixations_to_aoi(fixation_df: pd.DataFrame, aois: list[AOI]) -> pd.DataFrame:
    """将 fixation 表按 AOI 顺序分配到第一个命中的兴趣区域。"""
    required_columns = {"centroid_x", "centroid_y"}
    missing = sorted(required_columns - set(fixation_df.columns))
    if missing:
        raise ValueError(f"fixation_df is missing required columns: {missing}")

    assigned = fixation_df.copy()
    assigned["aoi_name"] = None
    if assigned.empty or not aois:
        return assigned

    valid_points = assigned[["centroid_x", "centroid_y"]].notna().all(axis=1)
    points = assigned.loc[valid_points, ["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    unassigned = assigned.loc[valid_points, "aoi_name"].isna()

    for aoi in aois:
        if points.size == 0:
            break
        matches = _match_aoi(points, aoi)
        if matches.size == 0:
            continue

        target_index = assigned.index[valid_points][matches & unassigned.to_numpy()]
        if len(target_index) == 0:
            continue
        assigned.loc[target_index, "aoi_name"] = aoi.name
        unassigned = assigned.loc[valid_points, "aoi_name"].isna()

    return assigned


def compute_aoi_metrics(
    fixations_with_aoi: pd.DataFrame,
    aois: list[AOI],
    total_duration: float,
) -> dict[str, AOIMetrics]:
    """计算每个 AOI 的 TTFF、驻留、访问与回视指标。"""
    if "aoi_name" not in fixations_with_aoi.columns:
        raise ValueError("fixations_with_aoi must contain `aoi_name` column.")

    required_columns = {"start_time_ms", "duration_ms"}
    missing = sorted(required_columns - set(fixations_with_aoi.columns))
    if missing:
        raise ValueError(f"fixations_with_aoi is missing required columns: {missing}")

    sequence = fixations_with_aoi["aoi_name"].tolist()
    baseline_start = (
        float(fixations_with_aoi["start_time_ms"].iloc[0]) if not fixations_with_aoi.empty else None
    )
    safe_total_duration = max(float(total_duration), 0.0)
    metrics: dict[str, AOIMetrics] = {}

    for aoi in aois:
        subset = fixations_with_aoi.loc[fixations_with_aoi["aoi_name"] == aoi.name]
        fixation_count = int(len(subset))
        total_dwell_time = float(subset["duration_ms"].sum()) if fixation_count else 0.0
        dwell_proportion = total_dwell_time / safe_total_duration if safe_total_duration > 0 else 0.0
        mean_fixation_duration = total_dwell_time / fixation_count if fixation_count > 0 else 0.0
        visit_count = _count_visits(sequence, target_name=aoi.name)
        revisit_count = max(0, visit_count - 1)

        if fixation_count > 0 and baseline_start is not None:
            first_fixation_time = float(subset["start_time_ms"].iloc[0] - baseline_start)
        else:
            first_fixation_time = None

        metrics[aoi.name] = AOIMetrics(
            aoi_name=aoi.name,
            first_fixation_time=first_fixation_time,
            total_dwell_time=total_dwell_time,
            dwell_proportion=float(dwell_proportion),
            fixation_count=fixation_count,
            visit_count=visit_count,
            revisit_count=revisit_count,
            mean_fixation_duration=float(mean_fixation_duration),
        )

    return metrics


def compute_transition_matrix(fixations_with_aoi: pd.DataFrame, aoi_names: list[str]) -> pd.DataFrame:
    """基于连续非空 AOI 序列计算行归一化转移矩阵。"""
    if "aoi_name" not in fixations_with_aoi.columns:
        raise ValueError("fixations_with_aoi must contain `aoi_name` column.")

    ordered_names = list(aoi_names)
    if not ordered_names:
        return pd.DataFrame(dtype=float)

    sequence = pd.Series(fixations_with_aoi["aoi_name"], dtype="object").dropna().reset_index(drop=True)
    collapsed = sequence.loc[sequence.ne(sequence.shift())].reset_index(drop=True)

    if len(collapsed) < 2:
        return pd.DataFrame(0.0, index=ordered_names, columns=ordered_names)

    transition_pairs = pd.DataFrame(
        {
            "source": pd.Categorical(collapsed.iloc[:-1], categories=ordered_names),
            "target": pd.Categorical(collapsed.iloc[1:], categories=ordered_names),
        }
    )
    counts = pd.crosstab(transition_pairs["source"], transition_pairs["target"])
    counts = counts.reindex(index=ordered_names, columns=ordered_names, fill_value=0)

    row_totals = counts.sum(axis=1).replace(0, np.nan)
    normalized = counts.div(row_totals, axis=0).fillna(0.0)
    return normalized.astype(float)


def _match_aoi(points: np.ndarray, aoi: AOI) -> np.ndarray:
    if aoi.region_type == "rectangle":
        x_min, y_min, x_max, y_max = aoi.region
        return (
            (points[:, 0] >= float(x_min))
            & (points[:, 0] <= float(x_max))
            & (points[:, 1] >= float(y_min))
            & (points[:, 1] <= float(y_max))
        )
    if aoi.region_type == "polygon":
        polygon = Path(np.asarray(aoi.region, dtype=float))
        return polygon.contains_points(points, radius=1e-9)
    raise ValueError(f"Unsupported AOI region type: {aoi.region_type}")


def _count_visits(sequence: list[object], target_name: str) -> int:
    visit_count = 0
    previous_name: object = None
    for current_name in sequence:
        if current_name == target_name and previous_name != target_name:
            visit_count += 1
        previous_name = current_name
    return visit_count

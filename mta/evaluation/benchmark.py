"""评估模块：用模拟器的 ground truth 评估归因算法偏差。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from mta.data import Journey
from mta.models.base import BaseAttributionModel
from mta.simulator import SimulationGroundTruth


def _truth_credit(truth: SimulationGroundTruth | dict[str, float]) -> dict[str, float]:
    if isinstance(truth, dict):
        total = sum(truth.values())
        return {k: v / total for k, v in truth.items()} if total > 0 else dict.fromkeys(truth, 0.0)
    return truth.normalized_credit()


def evaluate_against_truth(
    result_scores: dict[str, float],
    truth: SimulationGroundTruth | dict[str, float],
) -> pd.DataFrame:
    """比较归因输出与真实渠道贡献（反事实移除效应）。

    指标：
    - attributed / ground_truth：归因占比 vs 真实占比
    - abs_error：占比绝对误差
    """
    pred_total = sum(result_scores.values())
    truth_credit = _truth_credit(truth)
    rows = []
    for ch in sorted(set(result_scores) | set(truth_credit)):
        pred = result_scores.get(ch, 0.0) / pred_total if pred_total > 0 else 0.0
        true = truth_credit.get(ch, 0.0)
        rows.append(
            {
                "channel": ch,
                "attributed": pred,
                "ground_truth": true,
                "abs_error": abs(pred - true),
            }
        )
    return pd.DataFrame(rows)


def benchmark(
    models: Sequence[BaseAttributionModel],
    journeys: Sequence[Journey],
    truth: SimulationGroundTruth | dict[str, float],
) -> pd.DataFrame:
    """批量跑多个模型并汇总每个模型的平均绝对误差（MAE）。"""
    rows = []
    for model in models:
        res = model.attribute(journeys)
        total = sum(res.scores.values()) or 1.0
        truth_credit = _truth_credit(truth)
        mae = float(
            np.mean(
                [
                    abs(res.scores.get(ch, 0.0) / total - truth_credit.get(ch, 0.0))
                    for ch in set(res.scores) | set(truth_credit)
                ]
            )
        )
        rows.append({"model": model.name, "mae_vs_truth": mae})
    return (
        pd.DataFrame(rows)
        .sort_values("mae_vs_truth")
        .reset_index(drop=True)
    )

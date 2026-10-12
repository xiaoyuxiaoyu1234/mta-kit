"""启发式归因模型：首触/末触/线性/位置/时间衰减。"""

from __future__ import annotations

import math
from collections.abc import Sequence

from mta.data import AttributionResult, Journey
from mta.models.base import BaseAttributionModel, register


def _accumulate(weights_per_journey: Sequence[Sequence[float]],
                journeys: Sequence[Journey],
                channels: Sequence[str]) -> AttributionResult:
    scores = {c: 0.0 for c in channels}
    for w, j in zip(weights_per_journey, journeys):
        if not j.converted:
            continue
        for tp, weight in zip(j.touchpoints, w):
            scores[tp] += weight * j.revenue
    return AttributionResult(scores=scores, model="")


@register
class LastTouch(BaseAttributionModel):
    """末次触点归因：全部功劳归转化前最后一个渠道。"""

    name = "last_touch"

    def attribute(self, journeys: Sequence[Journey]) -> AttributionResult:
        channels = self._channels(journeys)
        weights = []
        for j in journeys:
            w = [0.0] * len(j.touchpoints)
            w[-1] = 1.0
            weights.append(w)
        result = _accumulate(weights, journeys, channels)
        result.model = self.name
        return result


@register
class FirstTouch(BaseAttributionModel):
    """首次触点归因：全部功劳归第一个渠道。"""

    name = "first_touch"

    def attribute(self, journeys: Sequence[Journey]) -> AttributionResult:
        channels = self._channels(journeys)
        weights = []
        for j in journeys:
            w = [0.0] * len(j.touchpoints)
            w[0] = 1.0
            weights.append(w)
        result = _accumulate(weights, journeys, channels)
        result.model = self.name
        return result


@register
class Linear(BaseAttributionModel):
    """线性归因：功劳均分给路径上所有触点。"""

    name = "linear"

    def attribute(self, journeys: Sequence[Journey]) -> AttributionResult:
        channels = self._channels(journeys)
        weights = []
        for j in journeys:
            n = len(j.touchpoints)
            weights.append([1.0 / n] * n)
        result = _accumulate(weights, journeys, channels)
        result.model = self.name
        return result


@register
class PositionBased(BaseAttributionModel):
    """U 型 / 位置归因：首末各 40%，中间均分 20%。

    Parameters
    ----------
    first_weight, last_weight:
        首末触点权重（默认 0.4 / 0.4）。
    """

    name = "position_based"

    def __init__(self, first_weight: float = 0.4, last_weight: float = 0.4):
        self.first_weight = first_weight
        self.last_weight = last_weight

    def attribute(self, journeys: Sequence[Journey]) -> AttributionResult:
        channels = self._channels(journeys)
        weights = []
        for j in journeys:
            n = len(j.touchpoints)
            if n == 1:
                weights.append([1.0])
            elif n == 2:
                mid = self.first_weight + self.last_weight
                weights.append([mid / 2, mid / 2])
            else:
                mid_each = (1 - self.first_weight - self.last_weight) / (n - 2)
                w = [self.first_weight] + [mid_each] * (n - 2) + [self.last_weight]
                weights.append(w)
        result = _accumulate(weights, journeys, channels)
        result.model = self.name
        return result


@register
class TimeDecay(BaseAttributionModel):
    """时间衰减归因：越靠近转化的触点权重越高。

    weight ∝ exp(-lambda * (距转化的触点数 - 1))
    """

    name = "time_decay"

    def __init__(self, half_life: float = 2.0):
        if half_life <= 0:
            raise ValueError("half_life 必须为正")
        self.half_life = half_life

    def attribute(self, journeys: Sequence[Journey]) -> AttributionResult:
        channels = self._channels(journeys)
        lam = math.log(2) / self.half_life
        weights = []
        for j in journeys:
            n = len(j.touchpoints)
            # 距转化按"触点间隔"计：最后一个触点距离为 0
            raw = [math.exp(-lam * (n - 1 - i)) for i in range(n)]
            s = sum(raw)
            weights.append([r / s for r in raw])
        result = _accumulate(weights, journeys, channels)
        result.model = self.name
        return result

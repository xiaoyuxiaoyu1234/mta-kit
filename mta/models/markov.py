"""Markov Chain 归因（移除效应 / Removal Effect）。

将用户路径建模为离散状态转移图：
    start -> ch1 -> ch2 -> ... -> conversion / null

渠道 c 的移除效应 = 完整图的转化概率 - 移除 c 后（经过 c 的边全部删除）
从 start 到 conversion 的转化概率。各渠道按移除效应占比分配转化功劳。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from mta.data import AttributionResult, Journey
from mta.models.base import BaseAttributionModel, register

START = "(start)"
CONV = "(conversion)"
NULL = "(null)"


@register
class MarkovAttribution(BaseAttributionModel):
    """基于一阶马尔可夫链 + 移除效应的归因。

    Parameters
    ----------
    order:
        使用几阶马尔可夫链（1 = 只看当前状态；k = 看前 k 个状态）。
        默认 1 阶（经典 Removal Effect 做法）。
    """

    name = "markov"

    def __init__(self, order: int = 1):
        if order < 1:
            raise ValueError("order 必须 >= 1")
        self.order = order

    # ---------- 构建转移计数 ----------
    def _transition_counts(self, journeys: Sequence[Journey]) -> Counter:
        counter: Counter = Counter()
        order = self.order
        for j in journeys:
            tps = list(dict.fromkeys(j.touchpoints)) if self._dedup else j.touchpoints
            states = [START] + tps + [CONV if j.converted else NULL]
            # 滑动窗口构造 k 阶状态
            for i in range(1, len(states)):
                prev_states = states[max(0, i - order):i]
                from_state = tuple(prev_states) if order > 1 else prev_states[-1]
                counter[(from_state, states[i])] += 1
        return counter

    _dedup = False

    # ---------- 转化概率（线性方程组求解） ----------
    def _conversion_probability(
        self, counter: Counter, removed: str | None = None
    ) -> float:
        """解 p(s) = Σ p(s')·P(s|s') 的吸收链方程，返回 start -> conversion 概率。

        采用迭代法（power iteration on absorbing chain），简单且稳健。
        """
        # 聚合出转移概率
        out_count: dict[object, Counter] = {}
        for (src, dst), n in counter.items():
            if removed is not None and (
                src == removed and dst == removed
            ):
                pass
            if removed is not None and (src == removed or dst == removed):
                continue  # 移除该渠道：删除所有涉及它的转移
            out_count.setdefault(src, Counter())[dst] += n

        probs: dict[object, dict[object, float]] = {}
        for src, dsts in out_count.items():
            total = sum(dsts.values())
            probs[src] = {d: n / total for d, n in dsts.items()}

        # 移除模式：被删转移的流量全部流入 NULL（用户流失），
        # 且各源节点的出边概率按"原始总流量"分母计算（不重新归一化），
        # 否则未转化流量会被错误放大为转化。
        if removed is not None:
            for src, dsts in out_count.items():
                denom = sum(
                    n
                    for (s2, _), n in counter.items()
                    if s2 == src
                )
                if denom <= 0:
                    continue
                trans: dict[object, float] = {}
                for dst, n in dsts.items():
                    trans[dst] = n / denom
                # 被移除渠道作为目的地的流量 -> NULL
                removed_mass = sum(
                    n / denom
                    for (s2, d2), n in counter.items()
                    if s2 == src and d2 == removed
                )
                trans[NULL] = trans.get(NULL, 0.0) + removed_mass
                probs[src] = trans

        # 迭代求解
        values: dict[object, float] = {START: 0.0, CONV: 1.0, NULL: 0.0}
        for _ in range(200):
            new_values = dict(values)
            delta = 0.0
            for src, trans in probs.items():
                if src in (CONV, NULL):
                    continue
                v = sum(values.get(d, 0.0) * p for d, p in trans.items())
                delta = max(delta, abs(v - values.get(src, 0.0)))
                new_values[src] = v
            values = new_values
            if delta < 1e-10:
                break
        return values.get(START, 0.0)

    def attribute(self, journeys: Sequence[Journey]) -> AttributionResult:
        self._dedup = False
        channels = self._channels(journeys)
        counter = self._transition_counts(journeys)
        p_full = self._conversion_probability(counter)

        if p_full <= 0:
            return AttributionResult(
                {c: 0.0 for c in channels}, model=self.name
            )

        removal: dict[str, float] = {}
        for c in channels:
            p_without = self._conversion_probability(counter, removed=c)
            removal[c] = max(0.0, (p_full - p_without) / p_full)

        return AttributionResult(scores=removal, model=self.name)

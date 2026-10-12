"""Shapley Value 归因（联盟博弈论方法）。

将每个渠道视为"玩家"，路径中渠道集合的子集视为"联盟"，
特征函数 v(S) = 包含渠道集 S 的转化数（或转化价值），
渠道 i 的 Shapley 值即其对转化的边际贡献期望。

精确计算复杂度 O(2^n)，默认当渠道数 <= 12 时使用精确算法，
否则自动切换到基于随机排列的蒙特卡洛采样估计。
"""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Sequence

from mta.data import AttributionResult, Journey
from mta.models.base import BaseAttributionModel, register


@register
class ShapleyAttribution(BaseAttributionModel):
    """基于 Shapley Value 的多触点归因。

    Parameters
    ----------
    order_normalize:
        是否对路径去重排序（把顺序敏感性消除，视为集合）。
        经典 Shapley MTA 文献常用 ordered 或 unordered 两种口径，
        默认 unordered（集合口径）。
    exact_threshold:
        渠道数超过该值时切换为蒙特卡洛估计。
    n_samples:
        蒙特卡洛采样的排列抽样次数。
    seed:
        随机种子（蒙特卡洛用）。
    """

    name = "shapley"

    def __init__(
        self,
        order_normalize: bool = True,
        exact_threshold: int = 12,
        n_samples: int = 100_000,
        seed: int = 42,
    ):
        self.order_normalize = order_normalize
        self.exact_threshold = exact_threshold
        self.n_samples = n_samples
        self.seed = seed

    # ---------- 特征函数 ----------
    def _characteristic(self) -> None:
        """v(S) = Σ 转化价值 × [路径触点集合 ⊆ S]。

        含义：若渠道联盟 S 已覆盖某条转化路径的全部触点，
        则该转化在联盟 S 下即可达成。这样保证 Shapley 总量
        恰好等于总转化价值（效率公理）。
        """
        r: dict[frozenset, float] = {}
        for j in self.journeys:
            if not j.converted:
                continue
            tps = j.touchpoints
            if self.order_normalize:
                tps = sorted(set(tps))
            key = frozenset(dict.fromkeys(tps))
            r[key] = r.get(key, 0.0) + j.revenue
        self._r = r

    def _val(self, s: frozenset) -> float:
        return sum(v for t, v in self._r.items() if t <= s)

    # ---------- 精确计算 ----------
    def _exact(self, channels: Sequence[str]) -> dict[str, float]:
        shapley = {c: 0.0 for c in channels}
        n = len(channels)
        for c in channels:
            others = [x for x in channels if x != c]
            for k in range(n):
                for subset in itertools.combinations(others, k):
                    s_with = frozenset(subset) | {c}
                    s_without = frozenset(subset)
                    marginal = self._val(s_with) - self._val(s_without)
                    weight = (math.factorial(k) * math.factorial(n - k - 1)
                              / math.factorial(n))
                    shapley[c] += weight * marginal
        return shapley

    # ---------- 蒙特卡洛 ----------
    def _montecarlo(self, channels: Sequence[str]) -> dict[str, float]:
        rng = random.Random(self.seed)
        sums = {c: 0.0 for c in channels}
        for _ in range(self.n_samples):
            perm = list(channels)
            rng.shuffle(perm)
            s: frozenset = frozenset()
            prev = self._val(s)
            for c in perm:
                s = s | {c}
                cur = self._val(s)
                sums[c] += cur - prev
                prev = cur
        n = self.n_samples
        return {c: v / n for c, v in sums.items()}

    # ---------- 接口 ----------
    def attribute(self, journeys: Sequence[Journey]) -> AttributionResult:
        self.journeys = list(journeys)
        channels = self._channels(journeys)
        self._characteristic()
        if len(channels) <= self.exact_threshold:            scores = self._exact(channels)
        else:
            scores = self._montecarlo(channels)
        return AttributionResult(scores=scores, model=self.name)

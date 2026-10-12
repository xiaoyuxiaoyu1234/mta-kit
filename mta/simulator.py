"""模拟数据生成器：mta-kit 的核心差异化能力。

与散装 notebook 不同，本生成器内置**已知的真实渠道贡献机制**：
用户按渠道各自的转化率"吸收"转化，因此每个渠道的真实因果贡献
（ground truth）可以精确计算，用来评估各归因算法的偏差。

机制说明
--------
- 每个用户从随机（或指定起始分布）渠道进入
- 每一步：以该渠道的转化率 p_conv 转化（终止）；以 p_exit 离开
  （生成一条未转化路径）；否则按转移概率矩阵跳到下一个渠道
- 渠道的真实贡献 ≈ 该渠道直接触发的转化数（模拟器计数可得）
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ChannelSpec:
    """单个渠道的行为参数。"""

    name: str
    p_convert: float            # 到达该渠道后直接转化的概率
    p_exit: float = 0.05        # 到达该渠道后流失的概率
    entry_share: float = 0.25   # 作为首次触点的入口占比（会被归一化）


@dataclass
class SimulationGroundTruth:
    """模拟产生的真实贡献（用于评估归因算法）。"""

    direct_conversions: dict[str, float]   # 各渠道直接触发的转化数
    total_conversions: int
    channel_credit: dict[str, float] = field(default_factory=dict)

    def normalized_credit(self) -> dict[str, float]:
        source = self.channel_credit or self.direct_conversions
        total = sum(source.values())
        if total <= 0:
            return {k: 0.0 for k in source}
        return {k: v / total for k, v in source.items()}


class JourneySimulator:
    """带因果机制的转化路径模拟器。"""

    def __init__(
        self,
        channels: list[ChannelSpec],
        transition: dict[str, dict[str, float]] | None = None,
        max_touchpoints: int = 6,
        seed: int = 42,
    ):
        self.channels = {c.name: c for c in channels}
        self.max_touchpoints = max_touchpoints
        self.rng = random.Random(seed)

        total_entry = sum(c.entry_share for c in channels)
        self._entry = {c.name: c.entry_share / total_entry for c in channels}

        if transition is None:
            # 默认：均匀跳转到其他渠道
            names = list(self.channels)
            self.transition = {
                a: {b: 1.0 / (len(names) - 1) for b in names if b != a}
                for a in names
            }
        else:
            self.transition = transition

    def _next_channel(self, current: str) -> str | None:
        trans = self.transition.get(current, {})
        r = self.rng.random()
        cum = 0.0
        for dst, p in trans.items():
            cum += p
            if r <= cum:
                return dst
        return None

    def generate(self, n_users: int = 50_000) -> tuple[pd.DataFrame, SimulationGroundTruth]:
        direct_conv = {name: 0.0 for name in self.channels}
        rows: list[dict] = []

        for _ in range(n_users):
            # 入口
            r = self.rng.random()
            cum = 0.0
            current = list(self.channels)[-1]
            for name, p in self._entry.items():
                cum += p
                if r <= cum:
                    current = name
                    break

            path: list[str] = []
            converted = False
            for _step in range(self.max_touchpoints):
                path.append(current)
                spec = self.channels[current]
                r2 = self.rng.random()
                if r2 < spec.p_convert:
                    converted = True
                    direct_conv[current] += 1.0
                    break
                if r2 < spec.p_convert + spec.p_exit:
                    break  # 流失
                nxt = self._next_channel(current)
                if nxt is None or nxt == current:
                    break
                current = nxt

            rows.append(
                {
                    "user_id": len(rows),
                    "paths": path,
                    "converted": converted,
                    "revenue": 1.0,
                }
            )

        df = pd.DataFrame(rows)
        truth = SimulationGroundTruth(
            direct_conversions=direct_conv,
            total_conversions=int(df["converted"].sum()),
        )
        return df, truth

    # ---------- 反事实 ground truth ----------
    def counterfactual_credit(
        self, n_users: int = 100_000, seed_offset: int = 1000
    ) -> dict[str, float]:
        """反事实渠道贡献：将渠道 c 的致转能力关闭（p_convert=0 且
        到达即流失）后重跑模拟，转化概率的相对下降即为 c 的移除效应。

        这是本库评估归因算法的 ground truth：它度量的是渠道的
        因果贡献，而非"最后触点在谁身上"。
        """

        base_p = self._baseline_conversion(n_users, seed_offset)
        if base_p <= 0:
            return {name: 0.0 for name in self.channels}

        credit: dict[str, float] = {}
        for i, name in enumerate(self.channels):
            saved = self.channels[name]
            neutral = ChannelSpec(
                name=name, p_convert=0.0, p_exit=1.0, entry_share=saved.entry_share
            )
            self.channels[name] = neutral
            p_without = self._baseline_conversion(n_users, seed_offset + 1 + i)
            self.channels[name] = saved
            credit[name] = max(0.0, (base_p - p_without) / base_p)
        return credit

    def _baseline_conversion(self, n_users: int, seed: int) -> float:
        saved_rng = self.rng
        self.rng = random.Random(seed)
        try:
            converted = 0
            for _ in range(n_users):
                if self._simulate_one():
                    converted += 1
            return converted / n_users
        finally:
            self.rng = saved_rng

    def _simulate_one(self) -> bool:
        """模拟单个用户，返回是否转化（不记录路径）。"""
        r = self.rng.random()
        cum = 0.0
        current = list(self.channels)[-1]
        for name, p in self._entry.items():
            cum += p
            if r <= cum:
                current = name
                break

        for _step in range(self.max_touchpoints):
            spec = self.channels[current]
            r2 = self.rng.random()
            if r2 < spec.p_convert:
                return True
            if r2 < spec.p_convert + spec.p_exit:
                return False
            nxt = self._next_channel(current)
            if nxt is None or nxt == current:
                return False
            current = nxt
        return False

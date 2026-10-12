"""深度学习归因：Additive Attribution Model（简化版 DeepMind 风格）。

思路：把路径中每个渠道触点 embedding 后做 attention/加性聚合，
输出该路径的转化概率；训练后，用梯度/积分梯度把转化概率分解回各渠道，
得到归因分数。本实现为不依赖 PyTorch 的轻量版：

- 特征：路径中各渠道的出现次数（one-hot 计数向量的加性和）
- 模型：单层逻辑回归（可换成 MLP / Transformer）
- 归因：将 logit 按各渠道的线性贡献占比分解（加性模型下梯度可解析计算）

这是"深度学习归因"最小可用内核，接口与统计模型一致，方便后续
替换为基于 PyTorch 的 Attention 归因（如 DNAMMT 风格的 RNN+反事实）。
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from mta.data import AttributionResult, Journey
from mta.models.base import BaseAttributionModel, register


@register
class DeepAdditiveAttribution(BaseAttributionModel):
    """加性深度归因模型（无第三方深度学习依赖的最小实现）。

    p(convert | path) = sigmoid(b + Σ_c w_c * count_c(path))

    归因分数：对每条转化路径，渠道 c 的边际贡献为
    p_path * w_c * count_c（软分配），对所有转化路径求和。
    """

    name = "deep_additive"

    def __init__(
        self,
        lr: float = 0.1,
        epochs: int = 300,
        l2: float = 1e-4,
        seed: int = 42,
    ):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.seed = seed

    def _featurize(self, j: Journey, channels: Sequence[str]) -> dict[str, float]:
        counts: dict[str, float] = {}
        for tp in j.touchpoints:
            counts[tp] = counts.get(tp, 0.0) + 1.0
        return counts

    def attribute(self, journeys: Sequence[Journey]) -> AttributionResult:
        channels = self._channels(journeys)
        idx = {c: i for i, c in enumerate(channels)}
        rng = random.Random(self.seed)

        # 初始化
        w = [rng.uniform(-0.01, 0.01) for _ in channels]
        b = 0.0

        # 训练（全批量梯度上升，最大化对数似然）
        feats = [(self._featurize(j, channels), 1.0 if j.converted else 0.0,
                  j.revenue) for j in journeys]
        for _ in range(self.epochs):
            gw = [0.0] * len(channels)
            gb = 0.0
            for counts, y, _rev in feats:
                z = b + sum(w[idx[c]] * v for c, v in counts.items())
                p = 1.0 / (1.0 + math.exp(-z))
                err = (y - p)  # 梯度方向
                gb += err
                for c, v in counts.items():
                    gw[idx[c]] += err * v
            n = len(feats)
            for i in range(len(channels)):
                w[i] += self.lr * (gw[i] / n - self.l2 * w[i])
            b += self.lr * gb / n

        # 归因：每条转化路径按 p * w_c * count_c 软分配
        scores = {c: 0.0 for c in channels}
        for counts, y, rev in feats:
            if y == 0:
                continue
            z = b + sum(w[idx[c]] * v for c, v in counts.items())
            p = 1.0 / (1.0 + math.exp(-z))
            for c, v in counts.items():
                scores[c] += max(0.0, p * w[idx[c]] * v) * rev

        return AttributionResult(scores=scores, model=self.name)

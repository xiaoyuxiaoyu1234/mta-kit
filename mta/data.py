"""数据结构：用户转化路径（Journey）。"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Journey:
    """单条用户转化路径。

    Parameters
    ----------
    touchpoints:
        按时间顺序排列的渠道触点序列，如 ["search", "social", "display"]。
    converted:
        该路径是否发生转化。
    revenue:
        可选，转化产生的收入（用于收入归因）。
    """

    touchpoints: list[str]
    converted: bool = True
    revenue: float = 1.0

    def __post_init__(self) -> None:
        if not self.touchpoints:
            raise ValueError("touchpoints 不能为空")


@dataclass
class AttributionResult:
    """归因结果：各渠道获得的归因权重/转化量/收入。"""

    scores: dict[str, float]
    model: str = ""

    def normalized(self) -> AttributionResult:
        total = sum(self.scores.values())
        if total <= 0:
            return AttributionResult(
                {k: 0.0 for k in self.scores}, self.model
            )
        return AttributionResult(
            {k: v / total for k, v in self.scores.items()}, self.model
        )

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {"channel": list(self.scores), "credit": list(self.scores.values())}
        )
        df["model"] = self.model
        return df


def journeys_from_frame(df: pd.DataFrame) -> list[Journey]:
    """从 DataFrame 构建路径列表。

    期望列：``paths``（渠道序列，list 或逗号分隔字符串）、
    ``converted``（0/1）、可选 ``revenue``。
    """
    journeys: list[Journey] = []
    for _, row in df.iterrows():
        path = row["paths"]
        if isinstance(path, str):
            path = [p.strip() for p in path.split(",") if p.strip()]
        journeys.append(
            Journey(
                touchpoints=list(path),
                converted=bool(row.get("converted", True)),
                revenue=float(row.get("revenue", 1.0)),
            )
        )
    return journeys

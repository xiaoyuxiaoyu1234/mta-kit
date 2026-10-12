"""Transformer 遮蔽反事实归因模型测试。"""

import pytest

from mta.data import Journey
from mta.models import TransformerAttentionAttribution

torch = pytest.importorskip("torch", reason="未安装 PyTorch 时跳过深度归因测试")


def make_journeys():
    # 信号明确：B 出现在转化路径的紧邻转化位置
    return [
        Journey(["A", "B"], converted=True),
        Journey(["A", "B"], converted=True),
        Journey(["B"], converted=True),
        Journey(["A"], converted=True),
        Journey(["A", "B"], converted=False),
        Journey(["A"], converted=False),
        Journey(["B", "A"], converted=False),
    ]


def test_scores_nonnegative_and_channels_complete():
    model = TransformerAttentionAttribution(epochs=8, d_model=16, nhead=2)
    res = model.attribute(make_journeys())
    assert set(res.scores) == {"A", "B"}
    assert all(v >= 0 for v in res.scores.values())
    assert sum(res.scores.values()) > 0


def test_unconverted_only_returns_zero():
    model = TransformerAttentionAttribution(epochs=2)
    res = model.attribute([Journey(["A"], converted=False)])
    assert res.scores["A"] == 0.0


def test_better_channel_gets_more_credit():
    # B 在所有转化路径都出现且紧邻转化，应比 A 拿到更多功劳
    model = TransformerAttentionAttribution(epochs=15)
    res = model.attribute(make_journeys())
    assert res.scores["B"] > res.scores["A"]

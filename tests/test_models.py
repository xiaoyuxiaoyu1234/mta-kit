import math

import pytest

from mta.data import Journey
from mta.models import (
    DeepAdditiveAttribution,
    FirstTouch,
    LastTouch,
    Linear,
    MarkovAttribution,
    PositionBased,
    ShapleyAttribution,
    TimeDecay,
    get_model,
)


def make_journeys():
    return [
        Journey(["search", "social", "display"], converted=True),
        Journey(["search"], converted=True),
        Journey(["social", "display"], converted=True),
        Journey(["display", "social", "search"], converted=True),
        Journey(["search", "social"], converted=False),  # 未转化：不参与归因
    ]


ALL_MODELS = [
    LastTouch,
    FirstTouch,
    Linear,
    PositionBased,
    TimeDecay,
    ShapleyAttribution,
    MarkovAttribution,
    DeepAdditiveAttribution,
]


@pytest.mark.parametrize("cls", ALL_MODELS)
def test_scores_are_nonnegative(cls):
    res = cls().attribute(make_journeys())
    assert all(v >= 0 for v in res.scores.values())


@pytest.mark.parametrize("cls", ALL_MODELS)
def test_channels_complete(cls):
    res = cls().attribute(make_journeys())
    assert set(res.scores) == {"search", "social", "display"}


def test_last_touch_concentrates_on_final_channel():
    res = LastTouch().attribute(make_journeys()).normalized()
    # 末触点：display(2), search(2), display... 末位为 display x2, search x2
    # 路径1 末位 display；2 search；3 display；4 search
    assert res.scores["social"] == 0.0
    assert math.isclose(res.scores["display"], 0.5)
    assert math.isclose(res.scores["search"], 0.5)


def test_first_touch():
    res = FirstTouch().attribute(make_journeys()).normalized()
    # 首触点分布：search x2, social x1, display x1
    assert math.isclose(res.scores["search"], 0.5)
    assert math.isclose(res.scores["social"], 0.25)
    assert math.isclose(res.scores["display"], 0.25)


def test_shapley_sums_reasonable():
    res = ShapleyAttribution().attribute(make_journeys())
    total = sum(res.scores.values())
    assert total > 0
    # Shapley 总量应等于所有转化价值之和 = 4
    assert math.isclose(total, 4.0, rel_tol=1e-6)


def test_markov_removal_positive():
    res = MarkovAttribution().attribute(make_journeys())
    assert all(v >= 0 for v in res.scores.values())
    assert sum(res.scores.values()) > 0


def test_get_model_registry():
    model = get_model("shapley", seed=1)
    assert isinstance(model, ShapleyAttribution)
    with pytest.raises(KeyError):
        get_model("nope")


def test_empty_path_rejected():
    with pytest.raises(ValueError):
        Journey([])


def test_unconverted_ignored():
    only_unconverted = [Journey(["search"], converted=False)]
    res = LastTouch().attribute(only_unconverted)
    assert res.scores["search"] == 0.0

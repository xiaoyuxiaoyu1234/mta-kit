"""Criteo 加载器单元测试（合成迷你 tsv.gz，不联网）。"""

import gzip
import os

import pandas as pd
import pytest

from mta.datasets.criteo import COLUMNS, load_criteo


@pytest.fixture
def mini_criteo(tmp_path):
    """构造 2 个用户、2 个 campaign、2 次转化的迷你数据（22 列）。"""
    def row(**kw):
        base = dict.fromkeys(COLUMNS, 0)
        base.update(
            timestamp=-1, uid=0, campaign=0, conversion=0, conversion_timestamp=-1,
            conversion_id=-1, attribution=0, click=1, click_pos=-1, click_nb=1,
            cost=0.1, cpo=-1, time_since_last_click=-1,
        )
        base.update(kw)
        return [base[c] for c in COLUMNS]

    rows = [
        row(timestamp=1, uid=100, campaign=11, click_nb=1),
        row(timestamp=2, uid=100, campaign=22, conversion=1,
            conversion_timestamp=5, conversion_id=900, attribution=1,
            click_pos=1, click_nb=2, cpo=1.0, time_since_last_click=3),
        row(timestamp=10, uid=200, campaign=11),
        row(timestamp=11, uid=200, campaign=22),
        row(timestamp=12, uid=200, campaign=11, conversion=1,
            conversion_timestamp=15, conversion_id=901, click_nb=3),
    ]
    df = pd.DataFrame(rows, columns=COLUMNS)
    # 补齐 cat 列值类型
    path = tmp_path / "mini.tsv.gz"
    with gzip.open(path, "wt") as f:
        df.to_csv(f, sep="\t", header=False, index=False)
    return str(path)


def test_load_criteo_reconstructs_journeys(mini_criteo):
    data = load_criteo(path=mini_criteo, n_rows=None)
    # 两次转化 → 两条路径
    assert data.n_conversions == 2
    # 用户 100：campaign 11 -> 22（转化 900）；用户 200：11 -> 22 -> 11（转化 901）
    paths = sorted(tuple(j.touchpoints) for j in data.journeys)
    assert ("c11", "c22") in paths
    assert ("c11", "c22", "c11") in paths
    assert all(j.converted for j in data.journeys)


def test_load_criteo_campaign_map(mini_criteo):
    data = load_criteo(path=mini_criteo, n_rows=None)
    assert set(data.campaign_map.values()) == {"c11", "c22"}


@pytest.mark.criteo
def test_real_criteo_integration():
    pytest.importorskip("pandas")
    if os.environ.get("RUN_CRITEO") != "1":
        pytest.skip("设置 RUN_CRITEO=1 才运行真实数据集成测试")
    path = "data/criteo_attribution_dataset.tsv.gz"
    if not (os.path.exists(path) and os.path.getsize(path) > 100_000_000):
        pytest.skip("Criteo 数据集未下载（集成测试）")
    data = load_criteo(path=path, n_rows=500_000, max_campaigns=5)
    assert data.n_conversions > 100

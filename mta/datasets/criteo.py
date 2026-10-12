"""Criteo Attribution Modeling for Bidding 数据集加载器。

数据集：Diemert et al., 2017（AdKDD），CC-BY-NC-SA 4.0
- 16.5M 展示、45K 转化、700 个匿名 campaign、30 天真实投放日志
- 官方地址 https://huggingface.co/datasets/criteo/criteo-attribution-dataset
  （国内可用 hf-mirror.com 镜像）

数据到转化路径的映射
--------------------
每行是一次展示（impression）。构建路径的关键点：

1. 只保留发生转化的展示所属的转化事件（conversion_id != -1）
2. 对每个 conversion_id，收集该用户（uid）在该转化时间戳之前的
   全部展示历史，按时间排序，campaign 序列即触点路径
3. 渠道 = 匿名 campaign id（映射为 c0, c1, ... 缩短名称）
4. **不要**直接把 impression 级 conversion 列当转化标签 ——
   必须以 conversion_id 重组（Diemert 论文明确指出的坑）

由于归因数据集没有因果 ground truth，Criteo 用于"真实数据上的
分布验证"：比较不同模型给出的渠道贡献结构与合理性检查
（如 last_touch 与 Criteo 官方 attribution 列的一致率）。
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass

import pandas as pd

from mta.data import Journey

CRITEO_URL = (
    "https://huggingface.co/datasets/criteo/criteo-attribution-dataset"
    "/resolve/main/criteo_attribution_dataset.tsv.gz"
)
CRITEO_MIRROR_URL = (
    "https://hf-mirror.com/datasets/criteo/criteo-attribution-dataset"
    "/resolve/main/criteo_attribution_dataset.tsv.gz"
)

COLUMNS = [
    "timestamp", "uid", "campaign", "conversion", "conversion_timestamp",
    "conversion_id", "attribution", "click", "click_pos", "click_nb",
    "cost", "cpo", "time_since_last_click",
] + [f"cat{i}" for i in range(1, 10)]


@dataclass
class CriteoData:
    journeys: list[Journey]
    df: pd.DataFrame              # 原始（抽样后）数据
    campaign_map: dict[int, str]  # campaign id -> 渠道名
    n_conversions: int


def download_criteo(
    path: str = "data/criteo_attribution_dataset.tsv.gz",
    mirror: bool = True,
) -> str:
    """下载 Criteo 数据集（约 623MB）。已存在则跳过。"""
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    url = CRITEO_MIRROR_URL if mirror else CRITEO_URL
    print(f"downloading Criteo dataset from {url} ...")
    urllib.request.urlretrieve(url, path)
    return path


def load_criteo(
    path: str = "data/criteo_attribution_dataset.tsv.gz",
    n_rows: int | None = 2_000_000,
    max_clicks_per_path: int = 12,
    max_campaigns: int | None = None,
) -> CriteoData:
    """加载 Criteo 数据并重组为转化路径。

    Parameters
    ----------
    path:
        tsv.gz 文件路径（可先用 :func:`download_criteo` 下载）。
    n_rows:
        读取的行数上限（全量 16.5M 行约需 2.4GB 内存）。
    max_clicks_per_path:
        每条路径最多保留的触点数（取最近的 N 次展示）。
    max_campaigns:
        只保留出现次数最多的前 K 个 campaign，其余归入 c_other。
    """
    df = pd.read_csv(
        path, sep="\t", names=COLUMNS, nrows=n_rows, compression="gzip" if path.endswith(".gz") else None
    )

    # 只保留点击展示（impression→路径以点击为触点；纯展示无行为信号）
    df = df[df["click"] == 1]

    # campaign 长尾合并
    if max_campaigns:
        top = df["campaign"].value_counts().head(max_campaigns).index
        df["campaign"] = df["campaign"].where(df["campaign"].isin(top), -1)
    ids = sorted(df["campaign"].unique())
    campaign_map = {int(c): (f"c{int(c)}" if c >= 0 else "c_other") for c in ids}

    journeys: list[Journey] = []
    # 按用户聚合展示，再按转化切分
    for uid, g in df.groupby("uid", sort=False):
        g = g.sort_values("timestamp")
        # 该用户涉及的转化时间
        user_convs = g[g["conversion_id"] != -1][["conversion_id", "conversion_timestamp"]]
        user_convs = user_convs.drop_duplicates()
        for _, row in user_convs.iterrows():
            conv_ts = row["conversion_timestamp"]
            # 转化前的展示触点（含转化展示本身）
            hist = g[g["timestamp"] <= conv_ts].tail(max_clicks_per_path)
            if hist.empty:
                continue
            tps = [campaign_map[int(c)] for c in hist["campaign"]]
            journeys.append(
                Journey(touchpoints=tps, converted=True, revenue=1.0)
            )

    return CriteoData(
        journeys=journeys,
        df=df,
        campaign_map=campaign_map,
        n_conversions=len(journeys),
    )

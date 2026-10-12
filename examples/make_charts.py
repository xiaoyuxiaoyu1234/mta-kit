"""生成 README 配图：模型偏差基准图 + 渠道贡献对比图。

运行：python examples/make_charts.py
输出：docs/benchmark.png、docs/channel_credit.png
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from mta.data import Journey
from mta.evaluation import benchmark
from mta.models import (
    DeepAdditiveAttribution,
    FirstTouch,
    LastTouch,
    Linear,
    MarkovAttribution,
    PositionBased,
    ShapleyAttribution,
    TimeDecay,
)
from mta.simulator import ChannelSpec, JourneySimulator

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "sans-serif"]


def build_data():
    channels = [
        ChannelSpec("搜索广告", p_convert=0.08, p_exit=0.05, entry_share=0.40),
        ChannelSpec("信息流",   p_convert=0.05, p_exit=0.08, entry_share=0.30),
        ChannelSpec("短视频",   p_convert=0.02, p_exit=0.10, entry_share=0.20),
        ChannelSpec("再营销",   p_convert=0.15, p_exit=0.03, entry_share=0.10),
    ]
    sim = JourneySimulator(channels, seed=2026)
    df, _ = sim.generate(20_000)
    truth_credit = sim.counterfactual_credit(50_000)
    journeys = [
        Journey(row.paths, row.converted, row.revenue) for row in df.itertuples()
    ]
    models = [
        MarkovAttribution(),
        ShapleyAttribution(),
        Linear(),
        TimeDecay(),
        PositionBased(),
        FirstTouch(),
        LastTouch(),
        DeepAdditiveAttribution(epochs=200),
    ]
    return journeys, truth_credit, models


def main() -> None:
    import os

    os.makedirs("docs", exist_ok=True)
    journeys, truth_credit, models = build_data()

    # ---- 图 1：模型偏差基准（MAE 横向条形图）----
    table = benchmark(models, journeys, truth_credit)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    colors = ["#2ecc71" if m < 0.05 else "#e74c3c" if m > 0.08 else "#f39c12"
              for m in table["mae_vs_truth"]]
    ax.barh(table["model"], table["mae_vs_truth"], color=colors)
    ax.set_xlabel("MAE vs. counterfactual ground truth（越低越准）")
    ax.set_title("归因模型准确性基准（模拟数据，4 渠道）")
    for i, v in enumerate(table["mae_vs_truth"]):
        ax.text(v + 0.003, i, f"{v:.4f}", va="center", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("docs/benchmark.png", dpi=150)
    print("saved docs/benchmark.png")

    # ---- 图 2：各渠道贡献对比（归因 vs 真实）----
    total = sum(truth_credit.values())
    truth_norm = {k: v / total for k, v in truth_credit.items()}
    picks = [LastTouch(), ShapleyAttribution(), MarkovAttribution()]
    rows = []
    for m in picks:
        res = m.attribute(journeys).normalized()
        for ch in truth_norm:
            rows.append(
                {"model": m.name, "channel": ch,
                 "credit": res.scores.get(ch, 0.0) / (sum(res.scores.values()) or 1)}
            )
    rows.append({"model": "ground_truth", "channel": ch, "credit": 0.0})
    plot_df = pd.DataFrame(rows)
    for ch, credit in truth_norm.items():
        plot_df.loc[len(plot_df)] = {
            "model": "ground_truth", "channel": ch, "credit": credit
        }

    fig, ax = plt.subplots(figsize=(8, 4.2))
    model_names = [m.name for m in picks] + ["ground_truth"]
    chans = list(truth_norm)
    width = 0.2
    import numpy as np

    x = np.arange(len(chans))
    for i, mn in enumerate(model_names):
        vals = [plot_df[(plot_df.model == mn) & (plot_df.channel == c)]["credit"].sum()
                for c in chans]
        style = {"hatch": "//", "color": "#7f8c8d"} if mn == "ground_truth" else {}
        ax.bar(x + (i - 1.5) * width, vals, width, label=mn, **style)
    ax.set_xticks(x, list(chans))
    ax.set_ylabel("归因占比")
    ax.set_title("各渠道归因结果 vs 反事实真实贡献")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("docs/channel_credit.png", dpi=150)
    print("saved docs/channel_credit.png")


if __name__ == "__main__":
    main()

"""端到端演示：模拟数据 -> 多模型归因 -> 与真实贡献对比基准。

运行：python examples/quickstart.py
"""

import pandas as pd

from mta.evaluation import benchmark, evaluate_against_truth
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


def main() -> None:
    pd.set_option("display.width", 120)

    # 1. 定义渠道：每个渠道有真实不同的转化能力（这是 ground truth 来源）
    channels = [
        ChannelSpec("搜索广告", p_convert=0.08, p_exit=0.05, entry_share=0.40),
        ChannelSpec("信息流",   p_convert=0.05, p_exit=0.08, entry_share=0.30),
        ChannelSpec("短视频",   p_convert=0.02, p_exit=0.10, entry_share=0.20),
        ChannelSpec("再营销",   p_convert=0.15, p_exit=0.03, entry_share=0.10),
    ]
    sim = JourneySimulator(channels, seed=2026)
    df, truth = sim.generate(n_users=20_000)

    n_conv = truth.total_conversions
    print(f"模拟用户 {len(df)}，转化 {n_conv}（转化率 {n_conv/len(df):.2%}）\n")

    # 3. 反事实 ground truth：关闭某渠道致转能力后转化率下降多少
    truth_credit = sim.counterfactual_credit(n_users=50_000)
    print("=== 反事实真实渠道贡献 ===")
    total = sum(truth_credit.values())
    for k, v in sorted(truth_credit.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v/total:.2%}")
    print()

    # 4. 跑所有模型
    journeys = [
        __import__("mta.data", fromlist=["Journey"]).Journey(
            row.paths, row.converted, row.revenue
        )
        for row in df.itertuples()
    ]
    models = [
        LastTouch(),
        FirstTouch(),
        Linear(),
        PositionBased(),
        TimeDecay(),
        ShapleyAttribution(),
        MarkovAttribution(),
        DeepAdditiveAttribution(epochs=200),
    ]

    # 5. 基准：谁最接近真实渠道贡献？
    table = benchmark(models, journeys, truth_credit)
    print("=== 模型偏差基准（MAE，越低越好）===")
    print(table.to_string(index=False), "\n")

    # 6. 展示细节：末触 vs Shapley vs 真实
    for model in [LastTouch(), ShapleyAttribution(), MarkovAttribution()]:
        res = model.attribute(journeys).normalized()
        detail = evaluate_against_truth(res.scores, truth_credit)
        print(f"=== {model.name} ===")
        print(detail.to_string(index=False), "\n")


if __name__ == "__main__":
    main()

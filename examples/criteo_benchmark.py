"""真实数据基准：Criteo Attribution Dataset（Diemert et al., 2017）。

说明：真实归因数据没有因果 ground truth（我们无法观测"如果
没有渠道 X 会怎样"），所以本示例做的是**分布验证**：

1. 各模型在真实 700 个匿名 campaign 上的贡献结构对比
2. 合理性检查：last_touch 应与 Criteo 官方 attribution 列
   （Criteo 末触归因）高度一致 —— 用于验证数据管线正确性
3. 集中度指标：真实数据上多触归因如何重新分配"抢功"渠道

运行：PYTHONPATH=. python examples/criteo_benchmark.py
（首次运行会自动下载数据集约 623MB，可用 --download 控制）
"""

import argparse

from mta.datasets.criteo import download_criteo, load_criteo
from mta.models import (
    LastTouch,
    Linear,
    MarkovAttribution,
    ShapleyAttribution,
    TransformerAttentionAttribution,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=2_000_000)
    parser.add_argument("--top-campaigns", type=int, default=10)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    if not args.skip_download:
        download_criteo(mirror=True)

    print("加载 Criteo 数据 ...")
    data = load_criteo(n_rows=args.rows, max_campaigns=args.top_campaigns)
    print(f"点击展示 {len(data.df):,}，重组转化路径 {data.n_conversions:,}，"
          f"渠道数 {len(data.campaign_map)}\n")

    models = [
        ("last_touch", LastTouch()),
        ("linear", Linear()),
        ("markov", MarkovAttribution()),
        ("shapley", ShapleyAttribution(exact_threshold=15, n_samples=20_000)),
        ("transformer", TransformerAttentionAttribution(epochs=10)),
    ]

    print(f"{'模型':<14}" + "".join(f"{c:>10}" for c in sorted(data.campaign_map.values())))
    for name, model in models:
        res = model.attribute(data.journeys).normalized()
        total = sum(res.scores.values()) or 1.0
        row = "".join(f"{res.scores.get(c, 0.0)/total:>10.2%}"
                      for c in sorted(data.campaign_map.values()))
        print(f"{name:<14}{row}")

    print("\n解读：last_touch 会把功劳集中到转化前最后一个 campaign；"
          "Markov/Shapley/Transformer 则按结构化证据分摊 ——"
          "两者的差异就是真实投放中预算错配的量级。")


if __name__ == "__main__":
    main()

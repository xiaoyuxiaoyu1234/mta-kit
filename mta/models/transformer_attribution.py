"""基于 PyTorch 的深度归因模型：Transformer 编码器 + 遮蔽反事实归因。

架构（DNAMMT 风格的最小内核）：
    触点 embedding → Transformer 编码器（捕捉触点间交互）
    → attention 池化 → 转化概率 p(convert | path)

归因方法：**遮蔽反事实（occlusion counterfactual）**
对每条转化路径，逐个移除触点 t，用模型计算 p(转化 | 路径 - t)，
渠道贡献 = Σ p_full - p_masked（负值截断为 0）。
这把"深度归因"从纯梯度启发式升级为模型内的因果反事实，
与库内 Markov 移除效应在方法论上对齐。

依赖：torch（未安装时给出清晰报错，不影响库其他功能）。
"""

from __future__ import annotations

from collections.abc import Sequence

from mta.data import AttributionResult, Journey
from mta.models.base import BaseAttributionModel, register

try:
    import torch
    from torch import nn
except ImportError as _exc:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _TORCH_ERR = _exc


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "TransformerAttentionAttribution 需要 PyTorch：pip install torch"
        ) from _TORCH_ERR


@register
class TransformerAttentionAttribution(BaseAttributionModel):
    """Transformer + 遮蔽反事实的深度归因模型。

    Parameters
    ----------
    d_model:
        embedding / 隐藏维度。
    nhead:
        Transformer 注意力头数。
    num_layers:
        Transformer 编码器层数。
    epochs / batch_size / lr:
        训练超参（CPU 即可训练，路径短、词表小）。
    seed:
        随机种子。
    """

    name = "transformer_occlusion"

    def __init__(
        self,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 1,
        dim_feedforward: int = 64,
        epochs: int = 15,
        batch_size: int = 256,
        lr: float = 1e-3,
        seed: int = 42,
        device: str = "cpu",
    ):
        _require_torch()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.device = device

    # ---------------- 模型 ----------------
    def _build(self, vocab_size: int, max_len: int) -> nn.Module:
        class Net(nn.Module):
            def __init__(dself):
                super().__init__()
                dself.emb = nn.Embedding(vocab_size, self.d_model)
                dself.pos = nn.Embedding(max_len, self.d_model)
                layer = nn.TransformerEncoderLayer(
                    d_model=self.d_model, nhead=self.nhead,
                    dim_feedforward=self.dim_feedforward,
                    batch_first=True, dropout=0.1,
                )
                dself.encoder = nn.TransformerEncoder(layer, self.num_layers)
                dself.attn = nn.Linear(self.d_model, 1)
                dself.head = nn.Linear(self.d_model, 1)

            def forward(dself, x):  # x: (B, L) int64, -1 为 padding
                pos = torch.arange(x.size(1), device=x.device)
                mask = x == -1
                h = dself.emb(x.clamp(min=0)) + dself.pos(pos)[None, :, :]
                h = dself.encoder(h, src_key_padding_mask=mask)
                w = torch.softmax(dself.attn(h).masked_fill(mask[..., None], -1e9)
                                  .squeeze(-1), dim=-1)  # (B, L)
                pooled = (h * w[..., None]).sum(dim=1)
                return dself.head(pooled).squeeze(-1)  # logits (B,)

        return Net().to(self.device)

    # ---------------- 训练 ----------------
    def _fit(self, seqs: list[list[int]], labels: list[int]) -> nn.Module:
        torch.manual_seed(self.seed)
        vocab_size = max(max(s) for s in seqs) + 1
        max_len = max(len(s) for s in seqs)
        self._vocab_size = vocab_size

        def pad(s: list[int]) -> list[int]:
            return s + [-1] * (max_len - len(s))

        x_all = torch.tensor([pad(s) for s in seqs], dtype=torch.long)
        y_all = torch.tensor(labels, dtype=torch.float32)
        model = self._build(vocab_size, max_len)
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
        lossf = nn.BCEWithLogitsLoss()

        n = len(seqs)
        for _epoch in range(self.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                x, y = x_all[idx], y_all[idx]
                opt.zero_grad()
                loss = lossf(model(x), y)
                loss.backward()
                opt.step()
        self._max_len = max_len
        return model

    @staticmethod
    def _encode(journey: Journey, channel_idx: dict[str, int]) -> list[int]:
        return [channel_idx[tp] for tp in journey.touchpoints]

    # ---------------- 归因 ----------------
    def attribute(self, journeys: Sequence[Journey]) -> AttributionResult:
        _require_torch()
        channels = self._channels(journeys)
        channel_idx = {c: i for i, c in enumerate(channels)}

        seqs = [self._encode(j, channel_idx) for j in journeys]
        labels = [int(j.converted) for j in journeys]
        if sum(labels) == 0:
            return AttributionResult({c: 0.0 for c in channels}, model=self.name)

        model = self._fit(seqs, labels)
        model.eval()
        base_rate = sum(labels) / len(labels)

        def prob(seq: list[int]) -> float:
            if not seq:  # 空路径：无法过 Transformer，用训练集基准转化率
                return base_rate
            x = torch.tensor([seq], dtype=torch.long)
            with torch.no_grad():
                return torch.sigmoid(model(x)).item()

        # 遮蔽反事实：逐触点移除，贡献 = p_full - p_masked
        scores = {c: 0.0 for c in channels}
        inv = {i: c for c, i in channel_idx.items()}
        for j, seq in zip(journeys, seqs):
            if not j.converted:
                continue
            p_full = prob(seq)
            for t in range(len(seq)):
                masked = seq[:t] + seq[t + 1:]
                delta = p_full - prob(masked)
                if delta > 0:
                    scores[inv[seq[t]]] += delta * j.revenue

        return AttributionResult(scores=scores, model=self.name)

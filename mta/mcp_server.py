"""mta-kit MCP Server —— 让 Claude / Cursor 等编码 Agent 直接调用归因分析。

零第三方依赖：手写 stdio JSON-RPC 2.0（Model Context Protocol），
无需安装 fastmcp。运行方式：

    python -m mta.mcp_server

在 Claude Code / Cursor / Claude Desktop 的 MCP 配置中加入：

    {
      "mcpServers": {
        "mta-kit": {
          "command": "python",
          "args": ["-m", "mta.mcp_server"],
          "cwd": "/path/to/mta-kit"
        }
      }
    }

暴露的工具：
    - simulate_and_benchmark：模拟路径数据并对比全部归因模型的准确性
    - run_attribution：对给定转化路径列表运行指定归因模型
"""

from __future__ import annotations

import json
import sys
from typing import Any

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
    TransformerAttentionAttribution,
    get_model,
)
from mta.simulator import ChannelSpec, JourneySimulator

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "mta-kit", "version": "0.1.0"}

MODEL_FACTORIES = {
    "last_touch": LastTouch,
    "first_touch": FirstTouch,
    "linear": Linear,
    "position_based": PositionBased,
    "time_decay": TimeDecay,
    "shapley": ShapleyAttribution,
    "markov": MarkovAttribution,
    "deep_additive": DeepAdditiveAttribution,
    "transformer_occlusion": TransformerAttentionAttribution,
}

TOOLS: list[dict[str, Any]] = [
    {
        "name": "simulate_and_benchmark",
        "description": (
            "模拟多触点转化路径数据（每个渠道带真实转化概率），"
            "运行所有 8 种归因模型，返回各模型相对反事实真实贡献的 MAE 基准。"
            "用于回答'哪种归因模型更准'类问题。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "channels": {
                    "type": "array",
                    "description": "渠道列表",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "p_convert": {"type": "number",
                                          "description": "到达后直接转化的概率 0~1"},
                            "p_exit": {"type": "number", "description": "流失概率 0~1"},
                            "entry_share": {"type": "number",
                                            "description": "入口占比（相对值即可）"},
                        },
                        "required": ["name", "p_convert"],
                    },
                },
                "n_users": {"type": "integer", "description": "模拟用户数，默认 20000"},
            },
            "required": ["channels"],
        },
    },
    {
        "name": "run_attribution",
        "description": (
            "对给定转化路径列表运行指定归因模型，返回各渠道归因权重。"
            "model 可选: last_touch, first_touch, linear, position_based, "
            "time_decay, shapley, markov, deep_additive。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "journeys": {
                    "type": "array",
                    "description": "转化路径列表；每项为 [touchpoints, converted]",
                    "items": {
                        "type": "array",
                        "items": {},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                },
                "model": {"type": "string", "description": "模型名，默认 shapley"},
            },
            "required": ["journeys"],
        },
    },
]


# ---------------- 工具实现 ----------------

def tool_simulate_and_benchmark(args: dict[str, Any]) -> str:
    channels = [
        ChannelSpec(
            name=c["name"],
            p_convert=float(c["p_convert"]),
            p_exit=float(c.get("p_exit", 0.05)),
            entry_share=float(c.get("entry_share", 0.25)),
        )
        for c in args["channels"]
    ]
    n_users = int(args.get("n_users", 20_000))
    if len(channels) < 2:
        raise ValueError("至少需要 2 个渠道")
    if any(not (0 < c.p_convert < 1) for c in channels):
        raise ValueError("p_convert 必须在 (0,1) 区间")

    sim = JourneySimulator(channels, seed=2026)
    df, _ = sim.generate(n_users)
    truth = sim.counterfactual_credit(min(50_000, n_users * 2))
    journeys = [Journey(r.paths, r.converted, r.revenue) for r in df.itertuples()]

    models = [f() for f in MODEL_FACTORIES.values()]
    table = benchmark(models, journeys, truth)

    truth_total = sum(truth.values())
    lines = [f"模拟用户 {len(df)}，反事实真实贡献："]
    for k in sorted(truth, key=lambda x: -truth[x]):
        lines.append(f"  {k}: {truth[k] / truth_total:.2%}")
    lines.append("")
    lines.append("模型准确性基准（MAE 越低越准）：")
    lines.append(table.to_string(index=False))
    return "\n".join(lines)


def tool_run_attribution(args: dict[str, Any]) -> str:
    model_name = args.get("model", "shapley")
    if model_name not in MODEL_FACTORIES:
        raise ValueError(f"未知模型 '{model_name}'，可选: {sorted(MODEL_FACTORIES)}")
    journeys: list[Journey] = []
    for item in args["journeys"]:
        tps, converted = item[0], item[1]
        journeys.append(Journey(list(tps), bool(converted)))
    model = get_model(model_name)
    res = model.attribute(journeys).normalized()
    return json.dumps(
        {"model": model_name, "n_journeys": len(journeys), "credits": res.scores},
        ensure_ascii=False,
        indent=2,
    )


TOOL_IMPL = {
    "simulate_and_benchmark": tool_simulate_and_benchmark,
    "run_attribution": tool_run_attribution,
}

# ---------------- JSON-RPC / MCP 骨架 ----------------

def handle_request(req: dict[str, Any]) -> dict[str, Any] | None:
    method = req.get("method", "")
    rid = req.get("id")

    def ok(result: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": rid, "result": result}

    def err(code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}}

    if method == "initialize":
        return ok(
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": SERVER_INFO,
            }
        )
    if method == "notifications/initialized":
        return None  # 通知，无需响应
    if method == "tools/list":
        return ok({"tools": TOOLS})
    if method == "tools/call":
        params = req.get("params", {})
        name = params.get("name", "")
        try:
            impl = TOOL_IMPL.get(name)
            if impl is None:
                return err(-32602, f"未知工具: {name}")
            text = impl(params.get("arguments", {}))
            return ok(
                {"content": [{"type": "text", "text": text}], "isError": False}
            )
        except Exception as exc:  # noqa: BLE001 — 错误回传给客户端
            return ok(
                {"content": [{"type": "text", "text": f"错误: {exc}"}], "isError": True}
            )
    if method == "ping":
        return ok({})
    if rid is not None:
        return err(-32601, f"未知方法: {method}")
    return None


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = handle_request(req)
        if resp is not None:
            sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()

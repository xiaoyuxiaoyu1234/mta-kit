"""MCP Server JSON-RPC 协议测试（不起子进程，直接调 handle_request）。"""

import json

from mta.mcp_server import handle_request


def test_initialize():
    resp = handle_request({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert resp["result"]["serverInfo"]["name"] == "mta-kit"
    assert "protocolVersion" in resp["result"]


def test_tools_list():
    resp = handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    names = {t["name"] for t in resp["result"]["tools"]}
    assert {"simulate_and_benchmark", "run_attribution"} <= names


def test_run_attribution_roundtrip():
    req = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "run_attribution",
            "arguments": {
                "model": "shapley",
                "journeys": [[["A", "B"], True], [["A"], True], [["B"], False]],
            },
        },
    }
    resp = handle_request(req)
    assert resp["result"]["isError"] is False
    payload = json.loads(resp["result"]["content"][0]["text"])
    assert abs(sum(payload["credits"].values()) - 1.0) < 1e-9


def test_unknown_tool_is_error_content():
    req = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {"name": "nope", "arguments": {}},
    }
    resp = handle_request(req)
    # 未知工具属于协议级错误（-32602）
    assert resp["error"]["code"] == -32602


def test_unknown_method():
    resp = handle_request({"jsonrpc": "2.0", "id": 5, "method": "wat"})
    assert resp["error"]["code"] == -32601


def test_notification_returns_none():
    assert handle_request(
        {"jsonrpc": "2.0", "method": "notifications/initialized"}
    ) is None

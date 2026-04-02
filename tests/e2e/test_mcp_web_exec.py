from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp", reason="e2e tests require the optional 'mcp' dependency")

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


@pytest.mark.skip(reason="web_search and run_experiment tools removed from classic surface")
@pytest.mark.asyncio
async def test_web_search_stub_and_run_experiment(tmp_repo: Path, tmp_berry_home: Path):
    # Configure Berry to allow web + exec for this test and use the offline stub search provider.
    cfg = {
        "allow_write": False,
        "allow_exec": True,
        "allow_web": True,
        "allow_web_private": False,
        "allowed_roots": [],
        "enforce_verification": False,
        "require_plan_approval": False,
        "exec_network_mode": "deny_if_possible",
        "exec_allowed_commands": list(
            {
                # Always allow the current python executable name for cross-platform tests.
                Path(sys.executable).name,
                "python",
                "python3",
            }
        ),
        "web_search_provider": "stub",
        "web_search_stub_results": [
            {
                "url": "https://example.com/",
                "title": "Example",
                "snippet": "stubbed",
            }
        ],
    }
    (tmp_berry_home / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    src_dir = Path(__file__).resolve().parents[2] / "src"
    env = {
        **os.environ,
        "BERRY_HOME": str(tmp_berry_home),
        "BERRY_PROJECT_ROOT": str(tmp_repo),
        "PYTHONPATH": str(src_dir) + (os.pathsep + os.environ.get("PYTHONPATH", ""))
        if os.environ.get("PYTHONPATH")
        else str(src_dir),
        # Keep verification deterministic in tests.
        "OPENAI_API_KEY": "",
    }
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "berry.mcp_server", "--transport", "stdio", "--project-root", str(tmp_repo)],
        env=env,
        cwd=str(tmp_repo),
    )

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Create a run so we can do per-run permission handshakes.
            start = await session.call_tool(
                "start_run",
                {"problem_statement": "Test problem", "deliverable": "Test deliverable"},
            )
            assert start.isError is False
            run_id = start.structuredContent["result"]["run_id"]

            # Web access handshake + stub search
            req_web = await session.call_tool("request_web_access", {"run_id": run_id})
            assert req_web.isError is False
            web_token = req_web.structuredContent["result"]["web_access_token"]

            grant_web = await session.call_tool(
                "grant_web_access", {"run_id": run_id, "web_access_token": web_token}
            )
            assert grant_web.isError is False

            ws = await session.call_tool("web_search", {"run_id": run_id, "query": "anything"})
            assert ws.isError is False
            out = ws.structuredContent["result"]
            assert out["provider"] == "stub"
            assert out["results"] and out["results"][0]["url"].startswith("https://")

            # Exec access handshake + experiment
            req_exec = await session.call_tool("request_exec_access", {"run_id": run_id})
            assert req_exec.isError is False
            exec_token = req_exec.structuredContent["result"]["exec_access_token"]

            grant_exec = await session.call_tool(
                "grant_exec_access", {"run_id": run_id, "exec_access_token": exec_token}
            )
            assert grant_exec.isError is False

            exp = await session.call_tool(
                "run_experiment",
                {
                    "run_id": run_id,
                    "args": [sys.executable, "-c", "print('hello')"],
                    "timeout_s": 10,
                },
            )
            assert exp.isError is False
            exp_out = exp.structuredContent["result"]
            assert exp_out["ok"] is True
            assert exp_out.get("sid")
            # Always report sandbox facts (even if no sandbox is available).
            assert "network_isolated" in exp_out
            assert "sandbox_method" in exp_out

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp", reason="e2e tests require the optional 'mcp' dependency")

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


@pytest.mark.asyncio
async def test_mcp_server_tools_and_prompts(tmp_repo: Path, tmp_berry_home: Path):
    """Test that the classic MCP server exposes only approved tools and prompts."""
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

            # Verify approved tools are present
            tools_res = await session.list_tools()
            tool_names = {t.name for t in tools_res.tools}

            # Core verification tools
            assert "detect_hallucination" in tool_names
            assert "audit_trace_budget" in tool_names

            # Run management tools
            assert "start_run" in tool_names
            assert "load_run" in tool_names
            assert "get_deliverable" in tool_names

            # Evidence span tools
            assert "add_span" in tool_names
            assert "add_file_span" in tool_names
            assert "record_attempt" in tool_names
            assert "list_attempts" in tool_names
            assert "distill_span" in tool_names
            assert "list_spans" in tool_names
            assert "get_span" in tool_names
            assert "search_spans" in tool_names

            # Verify non-approved tools are NOT present (classic server only)
            assert "berry_health" not in tool_names
            assert "read_repo_file" not in tool_names
            assert "write_repo_file" not in tool_names
            assert "list_repo_files" not in tool_names

            # Verify prompts are available
            prompts_res = await session.list_prompts()
            prompt_names = {p.name for p in prompts_res.prompts}
            assert len(prompt_names) > 0  # At least some prompts should be registered

            # Verification tools should fail closed when OPENAI_API_KEY not configured.
            verify = await session.call_tool(
                "detect_hallucination",
                {
                    "answer": "Hello world. [S0]",
                    "spans": [{"sid": "S0", "text": "Hello world."}],
                    "max_claims": 5,
                },
            )
            assert verify.isError is False
            out = verify.structuredContent["result"]
            assert out["flagged"] is True
            err = str(out.get("error") or "")
            assert "OPENAI_API_KEY" in err


@pytest.mark.asyncio
async def test_mcp_server_run_and_spans(tmp_repo: Path, tmp_berry_home: Path):
    """Test run management and span operations in the classic MCP server."""
    src_dir = Path(__file__).resolve().parents[2] / "src"
    env = {
        **os.environ,
        "BERRY_HOME": str(tmp_berry_home),
        "BERRY_PROJECT_ROOT": str(tmp_repo),
        "PYTHONPATH": str(src_dir) + (os.pathsep + os.environ.get("PYTHONPATH", ""))
        if os.environ.get("PYTHONPATH")
        else str(src_dir),
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

            # Start a new run
            rr = await session.call_tool(
                "start_run",
                {
                    "problem_statement": "Test problem statement",
                    "deliverable": "Test deliverable",
                },
            )
            assert rr.isError is False
            result = rr.structuredContent["result"]
            run_id = result["run_id"]
            assert run_id
            assert result["problem_sid"]
            assert result["deliverable_sid"]

            # Get deliverable
            dr = await session.call_tool("get_deliverable", {"run_id": run_id})
            assert dr.isError is False
            assert dr.structuredContent["result"]["text"] == "Test deliverable"

            # Add a text span
            sr = await session.call_tool(
                "add_span",
                {
                    "run_id": run_id,
                    "text": "This is evidence text.",
                    "source": "test",
                },
            )
            assert sr.isError is False
            span_sid = sr.structuredContent["result"]["sid"]
            assert span_sid

            # List spans
            lr = await session.call_tool("list_spans", {"run_id": run_id})
            assert lr.isError is False
            spans = lr.structuredContent["result"]["spans"]
            assert len(spans) >= 3  # problem, deliverable, and our span

            # Record and list attempts
            ar = await session.call_tool(
                "record_attempt",
                {
                    "run_id": run_id,
                    "claim_id": "ROOT_CAUSE",
                    "hypothesis": "Failure is caused by bad config parsing",
                    "action": "Ran the smallest repro and captured stderr",
                    "budget_minutes": 5,
                    "input_sids": [span_sid],
                    "output_sids": [span_sid],
                    "audit_status": "needs_more_evidence",
                    "decision": "continue",
                    "next_step": "Inspect config loader",
                },
            )
            assert ar.isError is False
            attempt_id = ar.structuredContent["result"]["attempt_id"]
            assert attempt_id == "A0"

            la = await session.call_tool("list_attempts", {"run_id": run_id})
            assert la.isError is False
            attempts = la.structuredContent["result"]["attempts"]
            assert len(attempts) == 1
            assert attempts[0]["claim_id"] == "ROOT_CAUSE"

            # Get span
            gr = await session.call_tool("get_span", {"run_id": run_id, "sid": span_sid})
            assert gr.isError is False
            assert gr.structuredContent["result"]["text"] == "This is evidence text."

            # Search spans
            searchr = await session.call_tool(
                "search_spans",
                {"run_id": run_id, "query": "evidence"},
            )
            assert searchr.isError is False
            results = searchr.structuredContent["result"]["results"]
            assert len(results) > 0
            assert any(r["sid"] == span_sid for r in results)

            # Add file span (using README.md from tmp_repo)
            readme = tmp_repo / "README.md"
            readme.write_text("# Test Repo\n\nThis is a test repository.\n", encoding="utf-8")

            fsr = await session.call_tool(
                "add_file_span",
                {
                    "run_id": run_id,
                    "path": str(readme),
                    "start_line": 1,
                    "end_line": 3,
                },
            )
            assert fsr.isError is False
            file_span_sid = fsr.structuredContent["result"]["sid"]
            assert file_span_sid

            # Get file span content
            fgr = await session.call_tool("get_span", {"run_id": run_id, "sid": file_span_sid})
            assert fgr.isError is False
            assert "Test Repo" in fgr.structuredContent["result"]["text"]

            # add_file_span should fail closed for paths outside project_root
            outside = tmp_repo.parent / "outside.txt"
            outside.write_text("secret\n", encoding="utf-8")
            denied = await session.call_tool(
                "add_file_span",
                {
                    "run_id": run_id,
                    "path": str(outside),
                    "start_line": 1,
                    "end_line": 1,
                },
            )
            assert denied.isError is True
            assert denied.content
            assert "File read not allowed" in denied.content[0].text

            # Distill span
            dist = await session.call_tool(
                "distill_span",
                {
                    "run_id": run_id,
                    "parent_sid": file_span_sid,
                    "pattern": "Test",
                },
            )
            assert dist.isError is False
            distilled_sid = dist.structuredContent["result"]["sid"]
            assert distilled_sid

            # Load run (should work for existing run)
            load_r = await session.call_tool("load_run", {"run_id": run_id})
            assert load_r.isError is False
            assert load_r.structuredContent["result"]["run_id"] == run_id

            # Persistence side-effects should include machine-readable ledgers.
            run_dir = tmp_berry_home / "runs" / run_id
            assert (run_dir / "evidence.tsv").exists()
            assert (run_dir / "attempts.tsv").exists()

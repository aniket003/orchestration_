from __future__ import annotations

import asyncio
import io
import json
import sys
from typing import Any

import app.cli as cli


def final_result() -> dict[str, Any]:
    return {
        "status": "final_subqueries_ready",
        "thread_id": "thread-1",
        "intent": {
            "intent_summary": "Create a voltage requirement.",
            "work_product": "Voltage Requirement",
            "confidence": "High",
        },
        "final_subquery_plan": {
            "final_subqueries": [
                {
                    "slot_pack_id": "LV_SENS__COMP_DEF__SYS2__V1",
                    "feature": "LV Sensing (Voltage Measurement)",
                    "slot_type": "Component Definition",
                    "channel": "Technical",
                    "final_subquery": "Retrieve LV voltage component definition.",
                    "subquery": "Retrieve LV voltage component definition.",
                    "rag_query": "LV voltage component definition",
                    "keyword_query": '"LV voltage" AND definition',
                    "graph_query_intent": "Traverse LV sensing component definitions.",
                }
            ]
        },
        "slot_subquery_plan": {
            "subqueries_by_slot": {
                "LV_SENS__COMP_DEF__SYS2__V1": {
                    "draft_subquery": "Draft LV voltage component definition."
                }
            }
        },
    }


def test_cli_runs_agent_without_server(monkeypatch: Any, capsys: Any) -> None:
    async def fake_run_requirement_agent(
        requirement: str,
        thread_id: str | None = None,
        auto_answer_clarifications: bool = False,
    ) -> dict[str, Any]:
        assert requirement == "The system shall report voltage."
        assert thread_id is None
        assert auto_answer_clarifications is True
        return final_result()

    monkeypatch.setattr(cli, "run_requirement_agent", fake_run_requirement_agent)

    exit_code = asyncio.run(
        cli.run(["--auto-answer", "The system shall report voltage."])
    )

    output = capsys.readouterr().out
    parsed = json.loads(output)
    assert exit_code == 0
    assert parsed["status"] == "final_subqueries_ready"
    final_subquery = parsed["final_output"]["final_subqueries_by_slot"][
        "LV_SENS__COMP_DEF__SYS2__V1"
    ]
    assert final_subquery["final_subquery"] == (
        "Retrieve LV voltage component definition."
    )
    assert "subquery" not in final_subquery
    assert "slot_subquery_plan" not in parsed


def test_cli_resumes_clarification_in_same_process(
    monkeypatch: Any,
    capsys: Any,
) -> None:
    async def fake_run_requirement_agent(
        requirement: str,
        thread_id: str | None = None,
        auto_answer_clarifications: bool = False,
    ) -> dict[str, Any]:
        return {
            "status": "awaiting_clarification",
            "thread_id": "thread-1",
            "clarification_questions": ["Which voltage?"],
            "intent": {
                "intent_summary": "Need voltage details.",
                "work_product": "Voltage Requirement",
                "confidence": "Medium",
            },
        }

    async def fake_resume_requirement_agent(
        thread_id: str,
        clarification_answers: dict[str, Any] | None = None,
        auto_answer_clarifications: bool | None = None,
    ) -> dict[str, Any]:
        assert thread_id == "thread-1"
        assert clarification_answers == {"Which voltage?": "12 V output voltage"}
        return final_result()

    monkeypatch.setattr(cli, "run_requirement_agent", fake_run_requirement_agent)
    monkeypatch.setattr(cli, "resume_requirement_agent", fake_resume_requirement_agent)
    monkeypatch.setattr(sys, "stdin", io.StringIO("12 V output voltage\n"))

    exit_code = asyncio.run(cli.run(["The system shall report voltage."]))

    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert exit_code == 0
    assert "Clarification required:" in captured.err
    assert parsed["status"] == "final_subqueries_ready"


def test_cli_summary_output(monkeypatch: Any, capsys: Any) -> None:
    async def fake_run_requirement_agent(
        requirement: str,
        thread_id: str | None = None,
        auto_answer_clarifications: bool = False,
    ) -> dict[str, Any]:
        return final_result()

    monkeypatch.setattr(cli, "run_requirement_agent", fake_run_requirement_agent)

    exit_code = asyncio.run(
        cli.run(["--summary", "--auto-answer", "The system shall report voltage."])
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "status: final_subqueries_ready" in output
    assert "Retrieve LV voltage component definition." in output

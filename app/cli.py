from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from agents.requirement_agent import AgentExecutionError
from agents.requirement_agent import resume_requirement_agent
from agents.requirement_agent import run_requirement_agent
from core.config import LLMConfigurationError
from core.config import get_agent_settings


PAUSED_STATUSES = {"awaiting_clarification", "awaiting_slot_clarification"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the requirement agent without starting the HTTP server.",
    )
    parser.add_argument(
        "requirement",
        nargs="*",
        help="Requirement text. Omit when using --file or piping stdin.",
    )
    parser.add_argument(
        "-f",
        "--file",
        type=Path,
        help="Read the requirement from a text file.",
    )
    parser.add_argument(
        "--auto-answer",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Let the agent answer clarification questions automatically. Defaults to "
            "AGENT_AUTO_ANSWER_CLARIFICATIONS from .env."
        ),
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Exit instead of prompting when the agent asks clarification questions.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full raw final state as JSON.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a human-readable summary instead of a dictionary.",
    )
    return parser


def _read_requirement(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    has_text_arg = bool(args.requirement)
    if args.file and has_text_arg:
        parser.error("Provide either requirement text or --file, not both.")

    if args.file:
        return args.file.read_text(encoding="utf-8").strip()

    if has_text_arg:
        return " ".join(args.requirement).strip()

    if not sys.stdin.isatty():
        return sys.stdin.read().strip()

    parser.error("Provide requirement text, --file, or pipe text on stdin.")
    raise AssertionError("unreachable")


def _write_status(message: str, json_mode: bool) -> None:
    stream = sys.stderr if json_mode else sys.stdout
    print(message, file=stream)


def _prompt_for_answers(questions: list[str], json_mode: bool) -> dict[str, str]:
    answers: dict[str, str] = {}
    stream = sys.stderr if json_mode else sys.stdout
    print("\nClarification required:", file=stream)

    for question in questions:
        print(f"\n{question}", file=stream)
        print("Answer: ", end="", file=stream, flush=True)
        answer = sys.stdin.readline()
        if answer == "":
            raise RuntimeError(
                "Clarification was required, but no answer was available on stdin."
            )
        answers[question] = answer.strip()

    return answers


def _print_json(result: dict[str, Any]) -> None:
    print(json.dumps(result, indent=2, default=str))


FINAL_SUBQUERY_OUTPUT_FIELDS = (
    "feature",
    "slot_type",
    "slot_pack_id",
    "channel",
    "retrieval_objective",
    "rag_query",
    "keyword_query",
    "graph_query_intent",
    "search_containers",
    "required_evidence",
    "filters",
    "retrieval_mode",
    "final_subquery",
    "applied_clarification_answer",
    "unresolved_items",
    "ready_for_retrieval",
    "knowledge_base_paths",
)


def _clean_final_subquery(subquery: dict[str, Any]) -> dict[str, Any]:
    cleaned = {
        field: subquery[field]
        for field in FINAL_SUBQUERY_OUTPUT_FIELDS
        if field in subquery
    }
    if "knowledge_base_paths" not in cleaned and "knowledge_paths" in subquery:
        cleaned["knowledge_base_paths"] = subquery["knowledge_paths"]
    if "final_subquery" not in cleaned and "subquery" in subquery:
        cleaned["final_subquery"] = subquery["subquery"]
    return cleaned


def _final_output_dict(result: dict[str, Any]) -> dict[str, Any]:
    intent = result.get("intent") or {}
    final_plan = result.get("final_subquery_plan") or {}
    token_usage = result.get("token_usage") or {}
    final_subqueries = [
        _clean_final_subquery(subquery)
        for subquery in final_plan.get("final_subqueries", [])
    ]
    final_subqueries_by_slot = {
        str(subquery["slot_pack_id"]): subquery
        for subquery in final_subqueries
        if subquery.get("slot_pack_id")
    }

    output: dict[str, Any] = {
        "status": result.get("status", "completed"),
        "thread_id": result.get("thread_id"),
        "intent": {
            "summary": intent.get("intent_summary"),
            "work_product": intent.get("work_product"),
            "objective": intent.get("objective"),
            "confidence": intent.get("confidence"),
            "improved_requirement_query": intent.get("improved_requirement_query"),
        },
        "final_output": {
            "answer_mapping_summary": final_plan.get("answer_mapping_summary"),
            "final_subqueries_by_slot": final_subqueries_by_slot,
        },
        "token_usage": token_usage.get("total", token_usage or None),
    }

    if result.get("clarification_questions"):
        output["clarification_questions"] = result["clarification_questions"]
    if result.get("slot_subquery_plan") and not final_subqueries_by_slot:
        output["slot_subquery_plan"] = result["slot_subquery_plan"]

    return output


def _print_summary(result: dict[str, Any]) -> None:
    print(f"status: {result.get('status', 'completed')}")
    if result.get("thread_id"):
        print(f"thread_id: {result['thread_id']}")

    intent = result.get("intent") or {}
    if intent:
        print(f"\nintent: {intent.get('intent_summary') or '-'}")
        print(f"work_product: {intent.get('work_product') or '-'}")
        print(f"confidence: {intent.get('confidence') or '-'}")

    final_plan = result.get("final_subquery_plan") or {}
    final_subqueries = final_plan.get("final_subqueries") or []
    if final_subqueries:
        print("\nfinal_subqueries:")
        for index, subquery in enumerate(final_subqueries, start=1):
            slot = subquery.get("slot_pack_id") or "-"
            title = subquery.get("slot_type") or "slot"
            channel = subquery.get("channel") or "Uncategorized"
            print(f"\n{index}. {title} | {channel} | {slot}")
            print(subquery.get("final_subquery") or "-")

            rag_query = subquery.get("rag_query")
            keyword_query = subquery.get("keyword_query")
            graph_query_intent = subquery.get("graph_query_intent")
            if rag_query:
                print(f"rag_query: {rag_query}")
            if keyword_query:
                print(f"keyword_query: {keyword_query}")
            if graph_query_intent:
                print(f"graph_query_intent: {graph_query_intent}")
    else:
        print("\nNo final_subquery_plan was returned.")

    token_total = (result.get("token_usage") or {}).get("total")
    if token_total:
        print(f"\ntoken_usage: {json.dumps(token_total, default=str)}")


async def run(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    requirement = _read_requirement(args, parser)
    if not requirement:
        parser.error("Requirement must not be empty.")

    auto_answer = (
        args.auto_answer
        if args.auto_answer is not None
        else get_agent_settings().agent_auto_answer_clarifications
    )
    structured_output = args.json or not args.summary

    try:
        result = await run_requirement_agent(
            requirement,
            auto_answer_clarifications=auto_answer,
        )

        while result.get("status") in PAUSED_STATUSES:
            if args.no_interactive:
                _write_status(
                    "Agent paused for clarification. Re-run without "
                    "--no-interactive or use --auto-answer.",
                    structured_output,
                )
                if args.summary:
                    _print_summary(result)
                elif args.json:
                    _print_json(result)
                else:
                    _print_json(_final_output_dict(result))
                return 2

            questions = result.get("clarification_questions") or []
            answers = _prompt_for_answers([str(q) for q in questions], structured_output)
            result = await resume_requirement_agent(
                str(result["thread_id"]),
                answers,
                auto_answer_clarifications=auto_answer,
            )
    except (ValueError, RuntimeError, AgentExecutionError, LLMConfigurationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.summary:
        _print_summary(result)
    elif args.json:
        _print_json(result)
    else:
        _print_json(_final_output_dict(result))
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(run()))


if __name__ == "__main__":
    main()

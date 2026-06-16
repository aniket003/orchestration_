import logging
from uuid import uuid4
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from openai import APIConnectionError, APIStatusError, APITimeoutError
from openai import LengthFinishReasonError

from agents.nodes.create_slot_subqueries import create_slot_subqueries
from agents.nodes.finalize_slot_subqueries import finalize_slot_subqueries
from agents.nodes.understand_requirement_intent import understand_requirement_intent
from agents.nodes.wait_for_clarification import wait_for_clarification
from agents.nodes.wait_for_slot_clarification import wait_for_slot_clarification
from agents.state import RequirementState


logger = logging.getLogger("requirement_agent.graph")


def route_after_intent(state: RequirementState) -> str:
    intent = state.get("intent") or {}
    if intent.get("clarification_questions"):
        return "wait_for_clarification"
    return "create_slot_subqueries"


def route_after_slot_subqueries(state: RequirementState) -> str:
    plan = state.get("slot_subquery_plan") or {}
    if plan.get("requires_clarification"):
        return "wait_for_slot_clarification"
    return "finalize_slot_subqueries"


def build_requirement_graph() -> Any:
    builder = StateGraph(RequirementState)
    builder.add_node("understand_requirement_intent", understand_requirement_intent)
    builder.add_node("wait_for_clarification", wait_for_clarification)
    builder.add_node("create_slot_subqueries", create_slot_subqueries)
    builder.add_node("wait_for_slot_clarification", wait_for_slot_clarification)
    builder.add_node("finalize_slot_subqueries", finalize_slot_subqueries)
    builder.add_edge(START, "understand_requirement_intent")
    builder.add_conditional_edges(
        "understand_requirement_intent",
        route_after_intent,
        {
            "wait_for_clarification": "wait_for_clarification",
            "create_slot_subqueries": "create_slot_subqueries",
        },
    )
    builder.add_edge("wait_for_clarification", "create_slot_subqueries")
    builder.add_conditional_edges(
        "create_slot_subqueries",
        route_after_slot_subqueries,
        {
            "wait_for_slot_clarification": "wait_for_slot_clarification",
            "finalize_slot_subqueries": "finalize_slot_subqueries",
        },
    )
    builder.add_edge("wait_for_slot_clarification", "finalize_slot_subqueries")
    builder.add_edge("finalize_slot_subqueries", END)
    return builder.compile(checkpointer=MemorySaver())


requirement_graph = build_requirement_graph()


class AgentExecutionError(RuntimeError):
    pass


def _config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def _public_result(result: dict[str, Any], thread_id: str) -> dict[str, Any]:
    interrupts = result.pop("__interrupt__", None)
    if interrupts:
        interrupt_value = interrupts[0].value
        state = dict(interrupt_value.get("state") or {})
        state["thread_id"] = thread_id
        state["status"] = interrupt_value.get("status", "awaiting_clarification")
        state["clarification_questions"] = interrupt_value.get(
            "clarification_questions", []
        )
        return state

    result["thread_id"] = thread_id
    return result


async def _invoke_graph(input_value: Any, thread_id: str, failure_context: str) -> Any:
    try:
        return await requirement_graph.ainvoke(input_value, _config(thread_id))
    except LengthFinishReasonError as exc:
        raise AgentExecutionError(
            f"The model response exceeded its completion budget while {failure_context}. "
            "Try again with a narrower request or shorter clarification answers."
        ) from exc
    except APITimeoutError as exc:
        raise AgentExecutionError(
            f"The model request timed out while {failure_context}. Try again, or reduce "
            "the number of selected slots/clarification detail for this run."
        ) from exc
    except APIConnectionError as exc:
        raise AgentExecutionError(
            f"The model connection failed while {failure_context}. Try again when the "
            "Azure OpenAI endpoint is reachable."
        ) from exc
    except APIStatusError as exc:
        raise AgentExecutionError(
            f"The model API returned HTTP {exc.status_code} while {failure_context}."
        ) from exc


async def run_requirement_agent(
    requirement: str,
    thread_id: str | None = None,
) -> dict[str, Any]:
    cleaned_requirement = requirement.strip()
    if not cleaned_requirement:
        raise ValueError("Requirement must not be empty.")
    graph_thread_id = thread_id or str(uuid4())
    logger.info("graph=requirement_agent thread_id=%s status=started", graph_thread_id)
    result = await _invoke_graph(
        {"requirement": cleaned_requirement},
        graph_thread_id,
        "parsing requirement intent",
    )
    logger.info(
        "graph=requirement_agent thread_id=%s status=completed_or_paused",
        graph_thread_id,
    )
    return _public_result(result, graph_thread_id)


async def resume_requirement_agent(
    thread_id: str,
    clarification_answers: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not thread_id.strip():
        raise ValueError("thread_id must not be empty.")
    logger.info("graph=requirement_agent thread_id=%s status=resuming", thread_id)
    result = await _invoke_graph(
        Command(resume=clarification_answers or {}),
        thread_id,
        "creating slot subqueries",
    )
    logger.info("graph=requirement_agent thread_id=%s status=completed", thread_id)
    return _public_result(result, thread_id)

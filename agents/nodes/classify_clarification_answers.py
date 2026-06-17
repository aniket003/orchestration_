import json
import logging
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from agents.nodes.llm_usage import invoke_structured_with_usage
from agents.state import RequirementState


logger = logging.getLogger("requirement_agent.nodes")


class AutoClarificationAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(description="Exact clarification question being answered.")
    answer: str = Field(
        description=(
            "Concise answer to use as clarification input. If the source state lacks "
            "a firm value, state a conservative assumption explicitly."
        )
    )
    confidence: Literal["High", "Medium", "Low"] = Field(
        description="Confidence that the answer is supported by the current state."
    )
    rationale: str = Field(
        description="Brief reason why this answer is acceptable for continuing."
    )


class AutoClarificationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(description="How the clarification answers were derived.")
    answers: list[AutoClarificationAnswer] = Field(min_length=1, max_length=10)


AUTO_CLARIFICATION_SYSTEM_PROMPT = """You are a classifier node in an automotive requirements-engineering LangGraph workflow.

Your job is to answer clarification questions internally so the graph can continue without pausing for the user.

Rules:
- Answer every provided clarification question exactly once.
- Preserve the exact question text in each answer object.
- Use only the requirement, normalized requirement, intent, resolved slots, draft subqueries, knowledge paths, and any existing clarification answers from the provided state.
- Do not invent precise numeric thresholds, timing, operating modes, safety levels, interfaces, or calibration values as facts.
- If a required detail is not explicitly available, provide a conservative assumption and mark confidence Low or Medium.
- Keep answers short and directly usable by the next graph node.
- Prefer bracketed placeholders only when even a reasonable assumption would be unsafe.
- This node must move the workflow forward; do not ask new questions.
"""


async def _auto_answer_questions(
    *,
    state: RequirementState,
    questions: list[str],
    stage: str,
) -> tuple[AutoClarificationPlan, dict[str, Any]]:
    payload = {
        "stage": stage,
        "questions": questions,
        "state": {
            "requirement": state.get("requirement"),
            "normalized_requirement": state.get("normalized_requirement"),
            "intent": state.get("intent") or {},
            "clarification_answers": state.get("clarification_answers") or {},
            "slot_subquery_plan": state.get("slot_subquery_plan") or {},
            "slot_clarification_answers": state.get("slot_clarification_answers") or {},
            "status": state.get("status"),
        },
    }
    plan, token_usage = await invoke_structured_with_usage(
        state=state,
        node=stage,
        schema=AutoClarificationPlan,
        messages=[
            SystemMessage(content=AUTO_CLARIFICATION_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    "Answer these clarification questions from the current state:\n"
                    f"{json.dumps(payload, indent=2, ensure_ascii=False)}"
                )
            ),
        ],
    )
    return plan, token_usage


def _answers_by_question(plan: AutoClarificationPlan) -> dict[str, str]:
    return {item.question: item.answer for item in plan.answers}


async def classify_clarification_answers(
    state: RequirementState,
) -> RequirementState:
    logger.info("node=classify_clarification_answers status=started")
    intent = state.get("intent") or {}
    questions = intent.get("clarification_questions") or []
    if not questions:
        logger.info("node=classify_clarification_answers status=skipped")
        return {"clarification_answers": {}}

    plan, token_usage = await _auto_answer_questions(
        state=state,
        questions=questions[:10],
        stage="requirement_intent_clarifications",
    )
    plan_data = plan.model_dump()
    logger.info(
        "node=classify_clarification_answers status=completed answers=%s",
        len(plan_data["answers"]),
    )
    return {
        "clarification_answers": _answers_by_question(plan),
        "auto_clarification_answers": plan_data,
        "token_usage": token_usage,
        "status": "auto_clarifications_generated",
    }


async def classify_slot_clarification_answers(
    state: RequirementState,
) -> RequirementState:
    logger.info("node=classify_slot_clarification_answers status=started")
    plan = state.get("slot_subquery_plan") or {}
    questions = plan.get("next_clarification_questions") or []
    if not questions:
        logger.info("node=classify_slot_clarification_answers status=skipped")
        return {"slot_clarification_answers": {}}

    auto_plan, token_usage = await _auto_answer_questions(
        state=state,
        questions=questions[:10],
        stage="slot_subquery_clarifications",
    )
    plan_data = auto_plan.model_dump()
    logger.info(
        "node=classify_slot_clarification_answers status=completed answers=%s",
        len(plan_data["answers"]),
    )
    return {
        "slot_clarification_answers": _answers_by_question(auto_plan),
        "auto_slot_clarification_answers": plan_data,
        "token_usage": token_usage,
        "status": "auto_slot_clarifications_generated",
    }

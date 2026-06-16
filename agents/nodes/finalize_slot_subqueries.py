import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from agents.knowledge.knowledge_base import enrich_subqueries_with_knowledge_paths
from agents.state import RequirementState
from core.llm import get_chat_model


logger = logging.getLogger("requirement_agent.nodes")


class FinalSlotSubquery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature: str
    slot_type: str
    slot_pack_id: str
    channel: str = Field(
        description=(
            "The subquery channel from node 2, such as Technical or "
            "Compliance and methods."
        )
    )
    final_subquery: str = Field(
        description="Final standalone subquery after applying slot clarifications."
    )
    applied_clarification_answer: str | None = Field(
        description="Clarification answer applied to this final subquery, if any."
    )
    unresolved_items: list[str] = Field(
        description="Any remaining unknown values that must stay as placeholders."
    )
    ready_for_generation: bool


class FinalSubqueryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer_mapping_summary: str
    final_subqueries: list[FinalSlotSubquery] = Field(min_length=1, max_length=100)


FINALIZE_SLOT_SUBQUERIES_SYSTEM_PROMPT = """You are node 3 in an automotive requirements-engineering LangGraph workflow.

You receive:
- node 1 intent and requirement context
- node 2 draft slot subqueries, keyed by slot_pack_id
- nearest knowledge_paths for each slot
- slot_clarification_answers mapped to each slot_pack_id

Your job is to create the final slot-level subqueries.

Rules:
- Generate exactly one final_subquery for each slot in subqueries_by_slot.
- Apply the mapped clarification answer for that slot when available.
- Preserve known facts and do not invent missing values.
- If information remains unknown, keep explicit bracketed placeholders.
- Preserve each slot's feature, slot_type, slot_pack_id, and channel exactly.
- Use knowledge_paths as retrieval/planning context, but do not fabricate facts from path names alone.
- ready_for_generation is true only when the final subquery is specific enough for the downstream requirement-generation node.
"""


def map_slot_clarification_answers(
    slot_subquery_plan: dict[str, Any],
    raw_answers: dict[str, Any],
) -> dict[str, Any]:
    subqueries_by_slot = slot_subquery_plan.get("subqueries_by_slot") or {}
    mapped: dict[str, Any] = {}
    shared_response = raw_answers.get("user_response")

    for slot_pack_id, package in subqueries_by_slot.items():
        question = (package.get("clarification_question") or {}).get("question")
        answer = None
        if slot_pack_id in raw_answers:
            answer = raw_answers[slot_pack_id]
        elif question and question in raw_answers:
            answer = raw_answers[question]
        elif shared_response:
            answer = shared_response

        mapped[slot_pack_id] = {
            "question": question,
            "answer": answer,
        }

    return mapped


async def finalize_slot_subqueries(state: RequirementState) -> RequirementState:
    logger.info("node=finalize_slot_subqueries status=started")
    slot_subquery_plan = state.get("slot_subquery_plan") or {}
    subqueries_by_slot = slot_subquery_plan.get("subqueries_by_slot") or {}
    if not subqueries_by_slot:
        raise ValueError("Cannot finalize subqueries without subqueries_by_slot.")

    mapped_answers = map_slot_clarification_answers(
        slot_subquery_plan,
        state.get("slot_clarification_answers") or {},
    )
    payload = {
        "previous_state": {
            "requirement": state.get("requirement"),
            "normalized_requirement": state.get("normalized_requirement"),
            "intent": state.get("intent") or {},
        },
        "slot_subquery_plan": slot_subquery_plan,
        "mapped_slot_clarification_answers": mapped_answers,
    }

    structured_model = get_chat_model().with_structured_output(
        FinalSubqueryPlan,
        method="json_schema",
        strict=True,
    )
    final_plan = await structured_model.ainvoke(
        [
            SystemMessage(content=FINALIZE_SLOT_SUBQUERIES_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    "Create final slot subqueries from this state:\n"
                    f"{json.dumps(payload, indent=2, ensure_ascii=False)}"
                )
            ),
        ]
    )

    plan_data = final_plan.model_dump()
    plan_data["mapped_slot_clarification_answers"] = mapped_answers
    plan_data["final_subqueries"] = [
        {
            **subquery,
            "channel": subqueries_by_slot[subquery["slot_pack_id"]].get(
                "channel",
                subquery["channel"],
            ),
            "subquery": subquery["final_subquery"],
            "knowledge_paths": subqueries_by_slot[subquery["slot_pack_id"]][
                "knowledge_paths"
            ],
        }
        for subquery in plan_data["final_subqueries"]
    ]
    plan_data = enrich_subqueries_with_knowledge_paths(plan_data)
    plan_data["final_subqueries_by_slot"] = {
        subquery["slot_pack_id"]: subquery for subquery in plan_data["final_subqueries"]
    }

    logger.info(
        "node=finalize_slot_subqueries status=completed final_subqueries=%s",
        len(plan_data["final_subqueries"]),
    )
    return {
        "final_subquery_plan": plan_data,
        "status": "final_subqueries_ready",
    }

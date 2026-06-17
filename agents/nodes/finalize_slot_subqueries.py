import json
import logging
import re
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from agents.knowledge.knowledge_base import enrich_subqueries_with_knowledge_paths
from agents.nodes.llm_usage import invoke_structured_with_usage
from agents.state import RequirementState


logger = logging.getLogger("requirement_agent.nodes")


PLACEHOLDER_PATTERN = re.compile(r"\[([^\[\]]+)\]")


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
    retrieval_objective: str = Field(
        description=(
            "What evidence this slot needs to retrieve from technical sources. Do not "
            "use bracketed placeholders."
        )
    )
    rag_query: str = Field(
        description=(
            "Natural-language semantic query for vector/RAG retrieval against "
            "technical documents. Do not use bracketed placeholders."
        )
    )
    keyword_query: str = Field(
        description=(
            "Boolean/keyword-style query for lexical search. Include exact terms, "
            "signal names, standards, units, thresholds, and synonyms when available. "
            "Do not use bracketed placeholders."
        )
    )
    graph_query_intent: str = Field(
        description=(
            "Graph database traversal intent in plain language, including likely "
            "entities, relationships, and properties to search. Do not use bracketed "
            "placeholders."
        )
    )
    search_containers: list[str] = Field(
        description=(
            "Knowledge-base containers/paths or document groups that should be "
            "searched first."
        ),
        min_length=1,
        max_length=10,
    )
    required_evidence: list[str] = Field(
        description=(
            "Types of source evidence expected from retrieval, such as thresholds, "
            "interfaces, diagnostic reactions, timing tables, calibration data, or "
            "verification criteria."
        ),
        min_length=1,
        max_length=10,
    )
    filters: list[str] = Field(
        description=(
            "Metadata filters or constraints for retrieval, such as component, system "
            "level, ASIL, interface, document type, or requirement level."
        ),
        max_length=10,
    )
    retrieval_mode: Literal["rag", "keyword", "graph", "hybrid"] = Field(
        description="Preferred first-pass retrieval mode for this subquery."
    )
    final_subquery: str = Field(
        description=(
            "Single retrieval-ready subquery combining the semantic, keyword, graph, "
            "and evidence intent for downstream orchestration. Do not use bracketed "
            "placeholders."
        )
    )
    applied_clarification_answer: str | None = Field(
        description="Clarification answer applied to this final subquery, if any."
    )
    unresolved_items: list[str] = Field(
        description="Any remaining unknown values that retrieval should try to resolve."
    )
    ready_for_retrieval: bool


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

Your job is to create final slot-level retrieval subqueries.

These are NOT requirement-generation prompts. They are retrieval/search plans that
will be sent to RAG/vector search, graph database traversal, keyword search, or
hybrid retrieval to find relevant technical source documents.

Rules:
- Generate exactly one final_subquery for each slot in subqueries_by_slot.
- Apply the mapped clarification answer for that slot when available.
- Preserve known facts and do not invent missing values.
- If information remains unknown, make it an unresolved retrieval target rather than inventing a value.
- Preserve each slot's feature, slot_type, slot_pack_id, and channel exactly.
- Use knowledge_paths as retrieval/planning context, but do not fabricate facts from path names alone.
- ready_for_retrieval is true when the subquery is specific enough to run against RAG, graph DB, keyword search, or hybrid retrieval.
- Prefer retrieval_mode "hybrid" unless one mode is clearly dominant.
- Do not put bracketed placeholders like [CAN signal name], [threshold], or [TBD] in retrieval_objective, rag_query, keyword_query, graph_query_intent, final_subquery, search_containers, required_evidence, or filters.
- Convert missing values into clean searchable concepts in unresolved_items, for example "CAN signal name", "scaling", "ASIL target", "FTTI", "diagnostic debounce time".
- Retrieval queries should be low-noise search instructions. Search for missing concepts; do not include placeholder syntax.

Each final subquery must include:
- retrieval_objective: the evidence to find.
- rag_query: semantic query for vector/RAG search.
- keyword_query: lexical query with important exact terms and synonyms.
- graph_query_intent: graph DB traversal intent, not executable Cypher unless known.
- search_containers: nearest knowledge paths/container names to search first.
- required_evidence: what retrieved documents must contain to be useful.
- filters: metadata constraints such as component, requirement level, ASIL, interface, standard, document type, or system element.
- final_subquery: a compact retrieval-ready instruction combining the above.
"""


def _remove_placeholder_brackets(value: Any) -> Any:
    if isinstance(value, str):
        return PLACEHOLDER_PATTERN.sub(r"\1", value)
    if isinstance(value, list):
        return [_remove_placeholder_brackets(item) for item in value]
    return value


def _clean_retrieval_placeholder_noise(subquery: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "retrieval_objective",
        "rag_query",
        "keyword_query",
        "graph_query_intent",
        "search_containers",
        "required_evidence",
        "filters",
        "final_subquery",
        "unresolved_items",
    )
    cleaned = dict(subquery)
    for field in fields:
        if field in cleaned:
            cleaned[field] = _remove_placeholder_brackets(cleaned[field])
    return cleaned


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

    final_plan, token_usage = await invoke_structured_with_usage(
        state=state,
        node="finalize_slot_subqueries",
        schema=FinalSubqueryPlan,
        messages=[
            SystemMessage(content=FINALIZE_SLOT_SUBQUERIES_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    "Create final slot subqueries from this state:\n"
                    f"{json.dumps(payload, indent=2, ensure_ascii=False)}"
                )
            ),
        ],
    )

    plan_data = final_plan.model_dump()
    plan_data["mapped_slot_clarification_answers"] = mapped_answers
    plan_data["final_subqueries"] = [
        {
            **_clean_retrieval_placeholder_noise(subquery),
            "channel": subqueries_by_slot[subquery["slot_pack_id"]].get(
                "channel",
                subquery["channel"],
            ),
            "subquery": _remove_placeholder_brackets(subquery["final_subquery"]),
            "ready_for_generation": subquery["ready_for_retrieval"],
            "knowledge_paths": subqueries_by_slot[subquery["slot_pack_id"]][
                "knowledge_paths"
            ],
            "knowledge_base_paths": subqueries_by_slot[subquery["slot_pack_id"]][
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
        "token_usage": token_usage,
        "status": "final_subqueries_ready",
    }

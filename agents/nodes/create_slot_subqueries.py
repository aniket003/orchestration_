import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from agents.knowledge.knowledge_base import enrich_subqueries_with_knowledge_paths
from agents.knowledge.knowledge_base import enrich_slots_with_knowledge_paths
from agents.nodes.llm_usage import invoke_structured_with_usage
from agents.state import RequirementState


logger = logging.getLogger("requirement_agent.nodes")


class SlotClarificationQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(description="A concise question specific to this slot.")
    reason: str = Field(
        description="Why this answer is required for this slot's subquery."
    )
    knowledge_path: str = Field(
        description=(
            "The exact nearest AI knowledge-base path/container that should be used "
            "to answer or verify this question."
        )
    )
    container_names: list[str] = Field(
        description=(
            "Exact container names from the knowledge_path that are relevant to this "
            "question, ordered from broadest to most specific."
        ),
        min_length=1,
        max_length=8,
    )


class SlotSubquery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature: str = Field(
        description="Exact Feature / System Element value from the resolved slot."
    )
    slot_type: str = Field(description="Exact slot type from the resolved slot.")
    slot_pack_id: str = Field(description="Exact slot pack ID from the resolved slot.")
    channel: str = Field(
        description=(
            "Subquery channel, for example Technical, Compliance and methods, "
            "Safety, Diagnostics, Communication, Calibration, Verification, "
            "Performance, or Security."
        )
    )
    objective: str = Field(
        description="What this slot-specific downstream query must accomplish."
    )
    subquery: str = Field(
        description=(
            "A standalone downstream subquery for this slot. Preserve facts and use "
            "bracketed placeholders for unknown values."
        )
    )
    known_context: list[str] = Field(
        description="Facts available for this slot from the prior state and answers."
    )
    missing_information: list[str] = Field(
        description="Information still missing for this slot."
    )
    clarification_questions: list[SlotClarificationQuestion] = Field(
        description=(
            "At most one slot-specific question that should be answered before final "
            "subquery generation."
        ),
        max_length=1,
    )
    ready_for_generation: bool = Field(
        description="True only when this slot subquery is sufficiently specific."
    )


class SlotSubqueryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer_integration_summary: str = Field(
        description="How user clarification answers were applied."
    )
    subqueries: list[SlotSubquery] = Field(min_length=1, max_length=100)
    requires_clarification: bool = Field(
        description="True when any slot still needs user clarification."
    )
    next_clarification_questions: list[str] = Field(
        description="Deduplicated user-facing questions across all slots.",
        max_length=10,
    )


SLOT_SUBQUERY_SYSTEM_PROMPT = """You are node 2 in an automotive requirements-engineering LangGraph workflow.

You receive the complete state from node 1 plus any clarification answers supplied by the user.
Your job is to convert the resolved technical matrix slots into downstream slot-specific subqueries.

Input contract:
- previous_state contains the original requirement, normalized requirement, intent analysis, and resolved relevant_slots.
- clarification_answers may be empty or may contain user answers to prior clarification questions.
- relevant_slots are already resolved from the source matrix and include feature, slot_type, slot_pack_id, themes, and nearest knowledge_paths from the AI knowledge base.

Core rules:
- Preserve all facts from previous_state and clarification_answers.
- Use clarification_answers to fill missing context when the answer clearly applies.
- Use each slot's knowledge_paths as retrieval/planning context when writing that slot's subquery.
- Do not invent thresholds, timing, operating modes, interfaces, safety levels, standards, fault reactions, or verification criteria.
- If a value is still missing, keep it as an explicit bracketed placeholder in the subquery.
- Generate exactly one subquery object for each relevant slot.
- Each subquery must be standalone enough for a downstream requirement-generation node.
- Each subquery must focus on its own slot_type and themes, while carrying shared context such as component, feature, safety level, and requirement level.
- Ask at most one slot-specific clarification question per subquery, only when the answer materially changes that slot's final generated subquery.
- Prefer concrete per-slot questions over repeating the same generic question everywhere.
- Deduplicate repeated questions in next_clarification_questions.
- ready_for_generation for a slot is true only if that slot has enough measurable detail to draft a precise, testable requirement without assumptions.
- Limit next_clarification_questions to the 10 most important questions across the whole plan.

Subquery channel and follow-up question policy:
- Assign a channel to every subquery.
- Use "Compliance and methods" when the subquery must determine rules, standards, methods, work products, ASPICE expectations, ISO 26262 expectations, verification evidence, or required compliance approach.
- Use "Technical" when the subquery must determine engineering behavior, measurement details, operating modes, interfaces, timing, diagnostics, calibration, or implementation-relevant constraints.
- You may use a more specific concise channel such as Safety, Diagnostics, Communication, Calibration, Verification, Performance, or Security when it is more accurate than Technical.
- For every slot where critical information is missing, ask at most one very important follow-up question.
- Every follow-up question must explicitly mention the nearest knowledge-base path or exact container names from that slot's knowledge_paths where the answer should be searched or verified.
- Populate clarification_questions[].knowledge_path with one exact path from the slot's knowledge_paths.
- Populate clarification_questions[].container_names by splitting that path into its meaningful containers.
- Do not ask follow-up questions for low-value details that can safely remain as bracketed placeholders.

Subquery structure guidance:
- Component Definition: include measurement point, signal definition, range, accuracy/resolution, units, and interface assumptions/placeholders.
- Functional Behavior: include trigger, modes, input-output behavior, initialization/default behavior, and update behavior.
- Communication Spec: include interface, message/signal name, scaling, freshness, timeout, and status/validity behavior.
- Diagnostics / Monitoring: include monitored entity, fault conditions, debounce/detection timing, reporting, and reaction placeholders.
- Functional Safety: include ASIL/safety context, safe/degraded state, FTTI/reaction time, independence, and evidence placeholders.
- Verification / Validation: include test method, acceptance criteria, setup, corner cases, and evidence artifacts.
- Calibration / Configuration: include configurable parameters, ranges/defaults, persistence, access control, and validation.
- Performance / Timing: include latency, update period, jitter, transient response, resource constraints, and verification approach.
- Security: include threat/security context only when the slot exists and the requirement objective needs it.
"""


async def create_slot_subqueries(state: RequirementState) -> RequirementState:
    logger.info("node=create_slot_subqueries status=started")
    intent = state.get("intent") or {}
    relevant_slots = intent.get("relevant_slots") or []
    if not relevant_slots:
        raise ValueError(
            "Cannot create slot subqueries without resolved relevant_slots."
        )

    enriched_slots = enrich_slots_with_knowledge_paths(relevant_slots)
    enriched_intent = dict(intent)
    enriched_intent["relevant_slots"] = enriched_slots
    payload = {
        "previous_state": {
            "requirement": state.get("requirement"),
            "normalized_requirement": state.get("normalized_requirement"),
            "intent": enriched_intent,
            "status": state.get("status"),
        },
        "clarification_answers": state.get("clarification_answers") or {},
    }

    plan, token_usage = await invoke_structured_with_usage(
        state={**state, "intent": enriched_intent},
        node="create_slot_subqueries",
        schema=SlotSubqueryPlan,
        messages=[
            SystemMessage(content=SLOT_SUBQUERY_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    "Create slot subqueries from this state:\n"
                    f"{json.dumps(payload, indent=2, ensure_ascii=False)}"
                )
            ),
        ],
    )
    plan_data = enrich_subqueries_with_knowledge_paths(plan.model_dump())
    subqueries_by_slot = {}
    next_questions = []
    seen_questions: set[str] = set()
    for subquery in plan_data.get("subqueries", []):
        slot_pack_id = subquery["slot_pack_id"]
        subqueries_by_slot[slot_pack_id] = {
            "slot": {
                "feature": subquery["feature"],
                "slot_type": subquery["slot_type"],
                "slot_pack_id": slot_pack_id,
            },
            "channel": subquery["channel"],
            "knowledge_paths": subquery["knowledge_paths"],
            "draft_subquery": subquery["subquery"],
            "known_context": subquery["known_context"],
            "missing_information": subquery["missing_information"],
            "clarification_question": (
                subquery["clarification_questions"][0]
                if subquery.get("clarification_questions")
                else None
            ),
            "ready_for_generation": subquery["ready_for_generation"],
        }
        for clarification in subquery.get("clarification_questions", []):
            question = clarification.get("question")
            if question and question not in seen_questions:
                next_questions.append(question)
                seen_questions.add(question)

    plan_data["subqueries_by_slot"] = subqueries_by_slot
    plan_data["next_clarification_questions"] = next_questions[:10]
    plan_data["requires_clarification"] = bool(next_questions)
    logger.info(
        "node=create_slot_subqueries status=completed subqueries=%s "
        "requires_clarification=%s knowledge_path_matches=%s",
        len(plan_data["subqueries"]),
        plan_data["requires_clarification"],
        sum(len(subquery["knowledge_paths"]) for subquery in plan_data["subqueries"]),
    )

    return {
        "intent": enriched_intent,
        "slot_subquery_plan": plan_data,
        "token_usage": token_usage,
        "status": (
            "awaiting_slot_clarification"
            if plan_data["requires_clarification"]
            else "slot_subqueries_ready"
        ),
    }

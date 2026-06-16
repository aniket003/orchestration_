import logging
import re
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from agents.knowledge.technical_matrix import (
    get_matrix_index_for_prompt,
    resolve_slot_references,
)
from agents.state import RequirementState
from core.llm import get_chat_model


logger = logging.getLogger("requirement_agent.nodes")


class SelectedSlotReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature: str = Field(
        description="Exact Feature / System Element value from the technical matrix."
    )
    slot_type: str = Field(
        description="Exact slot column name from the technical matrix."
    )


class InferredSlotReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature: str = Field(
        description=(
            "Feature or system element name for an inferred slot when no exact matrix "
            "slot covers required downstream context."
        )
    )
    slot_type: str = Field(description="Short, specific inferred slot type name.")
    slot_pack_id: str = Field(
        description=(
            "Generated stable ID using uppercase words separated by underscores and "
            "ending with __INFERRED__V1."
        ),
        pattern=r"^[A-Z0-9_]+__INFERRED__V1$",
    )
    themes: list[str] = Field(
        description="Three to five concrete themes this inferred slot should cover.",
        min_length=1,
        max_length=5,
    )
    rationale: str = Field(
        description=(
            "Why this inferred slot is necessary and why the matrix slots are not "
            "sufficient."
        )
    )


class RequirementIntentAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    work_product: str = Field(
        description=(
            "A short, specific engineering artifact name generated from the user's "
            "objective, such as 'LV Supply Voltage CAN Reporting Requirement'."
        )
    )
    objective: str = Field(
        description="The concrete engineering outcome the user wants to achieve."
    )
    intent_summary: str = Field(
        description="One precise sentence describing what the user wants to achieve."
    )
    known_context: list[str] = Field(
        description="Technical facts explicitly stated by the user."
    )
    missing_information: list[str] = Field(
        description="Information required before a precise requirement can be created."
    )
    clarification_questions: list[str] = Field(
        description="One to seven concise questions for the user.",
        max_length=7,
    )
    query_is_complete: bool
    improved_requirement_query: str = Field(
        description=(
            "A complete engineering-focused version of the user's query. Preserve known "
            "facts and use explicit placeholders for missing information."
        )
    )
    selected_slots: list[SelectedSlotReference] = Field(
        description=(
            "Only exact matrix slots that are directly relevant to the requirement or "
            "required missing context."
        ),
        min_length=1,
        max_length=10,
    )
    inferred_slots: list[InferredSlotReference] = Field(
        description=(
            "Additional non-matrix slots created only when required context is not "
            "covered by available matrix slots."
        ),
        max_length=5,
    )
    ready_for_requirement_generation: bool
    confidence: Literal["High", "Medium", "Low"]


INTENT_SYSTEM_PROMPT = """You are the first node in an automotive requirements-engineering workflow.

Analyze the user's request so later LangGraph nodes can create a precise, testable engineering requirement.
The requirement is for the automotive domain.

Primary duties:
1. Understand the user's engineering intent.
2. Preserve every technical fact explicitly supplied by the user.
3. Identify the information still missing for a precise, testable requirement.
4. Select every technical matrix feature that is needed for downstream requirement generation, including features that are not directly named by the user but are required by your engineering reasoning.

Intent extraction rules:
- Do not invent components, thresholds, timing, operating modes, interfaces, safety levels, standards, or other technical facts.
- Generate one short, specific work_product name from the user's objective.
- Do not select from a predefined work-product list or return a generic category when a more specific artifact name is possible.
- Derive the objective and intent without adding unstated engineering facts.
- ready_for_requirement_generation is true only when the request is unambiguous and contains enough measurable detail to draft a testable requirement without assumptions.

Clarification rules:
- Ask only questions whose answers materially affect the final requirement.
- Ask between one and five questions when information is missing; ask none when the request is already precise.
- Questions should cover relevant gaps such as system/component boundary, trigger, behavior, output, operating conditions, measurable acceptance criteria, timing, fault response, interfaces, calibration/configuration, verification evidence, and safety/security constraints.
- Do not ask for information already present in the request.
- improved_requirement_query must be suitable for a downstream requirement-generation node. If information is missing, retain explicit bracketed placeholders instead of inventing values.

Slot-selection policy:
- Treat the technical matrix index as the source of truth for selectable features and slot types.
- Select only the matrix slots that are directly relevant to the user's objective or are required to resolve missing context for that objective.
- Do not select every slot under a feature just because the feature is relevant.
- Do not add Verification / Validation, Performance / Timing, Calibration / Configuration, Security, Communication Spec, or robustness-related slots by default. Add them only when the user request, missing information, safety context, interface context, or downstream requirement generation truly needs them.
- Add related sensing, diagnostics, safety, communication, calibration/configuration, performance/timing, verification, or security slots only when you can explain why they materially affect the requirement.
- Slots are planning/context handles, not claims of known facts. Selecting a slot means the downstream workflow should consider that area; it does not mean you may invent values for that area.
- Include matrix slots required by your own engineering reasoning even when the user did not name them explicitly, as long as each selected slot is relevant to the requirement objective or to missing information that must be resolved.
- selected_slots.feature and selected_slots.slot_type must exactly match the supplied technical matrix.
- Do not select unavailable cells or values marked with "-".
- Do not copy slot IDs or themes into selected_slots; the application resolves complete matrix slot data from the source file.
- If no available matrix slot covers a required planning area, add an inferred_slots entry. Inferred slots must be specific, have a generated slot_pack_id ending in __INFERRED__V1, include concrete themes, and include a rationale.
- Do not create inferred slots that duplicate existing matrix slots.
- Prefer precise relevant slot coverage over broad slot coverage. Under-selection is better than adding unrelated slots.
"""


async def understand_requirement_intent(
    state: RequirementState,
) -> RequirementState:
    """Use llm to classify the request and generate clarification questions if required."""
    normalized = re.sub(r"\s+", " ", state["requirement"].strip())
    if not normalized:
        raise ValueError("Requirement must not be empty.")

    logger.info("node=understand_requirement_intent status=started")
    structured_model = get_chat_model().with_structured_output(
        RequirementIntentAnalysis,
        method="json_schema",
        strict=True,
    )
    analysis = await structured_model.ainvoke(
        [
            SystemMessage(
                content=(
                    f"{INTENT_SYSTEM_PROMPT}\n\n"
                    "TECHNICAL MATRIX INDEX (source of truth):\n"
                    f"{get_matrix_index_for_prompt()}"
                )
            ),
            HumanMessage(content=f"USER QUERY:\n{normalized}"),
        ]
    )

    intent = analysis.model_dump(exclude={"selected_slots", "inferred_slots"})
    intent["relevant_slots"] = resolve_slot_references(
        [reference.model_dump() for reference in analysis.selected_slots],
        [
            slot.model_dump() | {"source": "inferred"}
            for slot in analysis.inferred_slots
        ],
    )
    logger.info(
        "node=understand_requirement_intent status=completed selected_slots=%s "
        "inferred_slots=%s resolved_slots=%s",
        len(analysis.selected_slots),
        len(analysis.inferred_slots),
        len(intent["relevant_slots"]),
    )

    return {
        "normalized_requirement": normalized,
        "intent": intent,
        "status": (
            "ready_for_requirement_generation"
            if analysis.ready_for_requirement_generation
            else "awaiting_clarification"
        ),
    }

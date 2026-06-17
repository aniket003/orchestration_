from typing import Any, TypedDict


class RequirementState(TypedDict, total=False):
    requirement: str
    normalized_requirement: str
    intent: dict[str, Any]
    auto_answer_clarifications: bool
    clarification_answers: dict[str, Any]
    auto_clarification_answers: dict[str, Any]
    slot_clarification_answers: dict[str, Any]
    auto_slot_clarification_answers: dict[str, Any]
    slot_subquery_plan: dict[str, Any]
    final_subquery_plan: dict[str, Any]
    token_usage: dict[str, Any]
    status: str

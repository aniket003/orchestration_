from typing import Any, TypedDict


class RequirementState(TypedDict, total=False):
    requirement: str
    normalized_requirement: str
    intent: dict[str, Any]
    clarification_answers: dict[str, Any]
    slot_clarification_answers: dict[str, Any]
    slot_subquery_plan: dict[str, Any]
    final_subquery_plan: dict[str, Any]
    status: str

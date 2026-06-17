import logging

from langgraph.types import interrupt

from agents.state import RequirementState


logger = logging.getLogger("requirement_agent.nodes")


def wait_for_slot_clarification(state: RequirementState) -> RequirementState:
    logger.info("node=wait_for_slot_clarification status=started")
    plan = state.get("slot_subquery_plan") or {}
    questions = plan.get("next_clarification_questions") or []

    if not questions:
        logger.info("node=wait_for_slot_clarification status=skipped")
        return {"slot_clarification_answers": {}}

    logger.info(
        "node=wait_for_slot_clarification status=paused questions=%s",
        len(questions),
    )
    answers = interrupt(
        {
            "status": "awaiting_slot_clarification",
            "clarification_questions": questions,
            "state": {
                "requirement": state.get("requirement"),
                "normalized_requirement": state.get("normalized_requirement"),
                "intent": state.get("intent"),
                "slot_subquery_plan": plan,
                "token_usage": state.get("token_usage"),
                "status": state.get("status"),
            },
        }
    )

    logger.info("node=wait_for_slot_clarification status=resumed")
    return {
        "slot_clarification_answers": answers or {},
        "status": "slot_clarifications_received",
    }

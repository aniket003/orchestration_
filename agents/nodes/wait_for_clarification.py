import logging

from langgraph.types import interrupt

from agents.state import RequirementState


logger = logging.getLogger("requirement_agent.nodes")


def wait_for_clarification(state: RequirementState) -> RequirementState:
    logger.info("node=wait_for_clarification status=started")
    intent = state.get("intent") or {}
    questions = intent.get("clarification_questions") or []

    if not questions:
        logger.info("node=wait_for_clarification status=skipped")
        return {"clarification_answers": {}}

    logger.info(
        "node=wait_for_clarification status=paused questions=%s",
        len(questions),
    )
    answers = interrupt(
        {
            "status": "awaiting_clarification",
            "clarification_questions": questions,
            "state": {
                "requirement": state.get("requirement"),
                "normalized_requirement": state.get("normalized_requirement"),
                "intent": intent,
                "status": state.get("status"),
            },
        }
    )

    logger.info("node=wait_for_clarification status=resumed")
    return {"clarification_answers": answers or {}, "status": "clarifications_received"}

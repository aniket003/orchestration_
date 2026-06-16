from typing import Any

from fastapi import APIRouter
from fastapi import HTTPException
from pydantic import BaseModel, Field
from pydantic import model_validator

from agents.requirement_agent import AgentExecutionError
from agents.requirement_agent import resume_requirement_agent
from agents.requirement_agent import run_requirement_agent
from core.config import LLMConfigurationError


router = APIRouter(tags=["requirement-agent"])


class RequirementAgentRequest(BaseModel):
    requirement: str | None = Field(default=None, min_length=1)
    thread_id: str | None = Field(default=None, min_length=1)
    clarification_answers: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_execution_target(self) -> "RequirementAgentRequest":
        if self.requirement is None and self.thread_id is None:
            raise ValueError("Either requirement or thread_id must be provided.")
        if self.requirement is not None and self.clarification_answers:
            raise ValueError(
                "clarification_answers can only be provided when resuming by thread_id."
            )
        return self


class RequirementAgentResponse(BaseModel):
    result: dict[str, Any]


@router.post("/requirement_agent/run", response_model=RequirementAgentResponse)
@router.post(
    "/requirement-agent/run",
    response_model=RequirementAgentResponse,
    include_in_schema=False,
)
async def run_agent(payload: RequirementAgentRequest) -> RequirementAgentResponse:
    try:
        if payload.requirement is not None:
            result = await run_requirement_agent(
                payload.requirement,
                payload.thread_id,
            )
        else:
            result = await resume_requirement_agent(
                payload.thread_id or "",
                payload.clarification_answers,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except AgentExecutionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except LLMConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return RequirementAgentResponse(result=result)

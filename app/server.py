import logging
from typing import Any
from pathlib import Path

import socketio
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from agents.requirement_agent import AgentExecutionError
from agents.requirement_agent import resume_requirement_agent
from agents.requirement_agent import run_requirement_agent
from app.routes.health import router as health_router
from app.routes.requirement_agent import router as requirement_agent_router
from core.config import LLMConfigurationError


STATIC_DIR = Path(__file__).parent / "static"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

fastapi_app = FastAPI(title="Orchestration API", version="0.1.0")
fastapi_app.include_router(health_router)
fastapi_app.include_router(requirement_agent_router)
fastapi_app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@fastapi_app.get("/", include_in_schema=False)
async def chat_ui() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")


@sio.event
async def connect(sid: str, environ: dict[str, Any], auth: Any = None) -> None:
    await sio.emit("connected", {"sid": sid}, to=sid)


@sio.event
async def disconnect(sid: str) -> None:
    return None


@sio.on("requirement_agent:run")
async def run_requirement_agent_event(sid: str, data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {
            "ok": False,
            "error": "The event payload must be an object.",
        }

    try:
        if isinstance(data.get("thread_id"), str) and "clarification_answers" in data:
            result = await resume_requirement_agent(
                data["thread_id"],
                data.get("clarification_answers") or {},
            )
        elif isinstance(data.get("requirement"), str):
            result = await run_requirement_agent(
                data["requirement"],
                data.get("thread_id"),
            )
        elif isinstance(data.get("thread_id"), str):
            result = await resume_requirement_agent(
                data["thread_id"],
                data.get("clarification_answers") or {},
            )
        else:
            return {
                "ok": False,
                "error": (
                    "The event payload must contain either a string field named "
                    "'requirement' or a string field named 'thread_id'."
                ),
            }
    except (ValueError, AgentExecutionError, LLMConfigurationError) as exc:
        return {"ok": False, "error": str(exc)}

    response = {"ok": True, "result": result}
    await sio.emit("requirement_agent:completed", response, to=sid)
    return response


app = socketio.ASGIApp(sio, other_asgi_app=fastapi_app)

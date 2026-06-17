from typing import Any

from agents.state import RequirementState
from core.llm import get_chat_model


def _usage_value(usage: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, int):
            return value
    return 0


def extract_token_usage(raw_message: Any) -> dict[str, Any]:
    usage = getattr(raw_message, "usage_metadata", None)
    response_metadata = getattr(raw_message, "response_metadata", None) or {}
    if not usage:
        usage = response_metadata.get("token_usage") or {}

    if not isinstance(usage, dict):
        usage = {}

    input_tokens = _usage_value(usage, "input_tokens", "prompt_tokens")
    output_tokens = _usage_value(usage, "output_tokens", "completion_tokens")
    total_tokens = _usage_value(usage, "total_tokens")
    if not total_tokens:
        total_tokens = input_tokens + output_tokens

    details = usage.get("output_token_details") or usage.get("completion_tokens_details")
    reasoning_tokens = 0
    if isinstance(details, dict):
        reasoning_tokens = _usage_value(details, "reasoning", "reasoning_tokens")

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "reasoning_tokens": reasoning_tokens,
        "raw_usage": usage,
    }


def append_token_usage(
    state: RequirementState,
    *,
    node: str,
    usage: dict[str, Any],
) -> dict[str, Any]:
    current = state.get("token_usage") or {}
    matrix = list(current.get("matrix") or [])
    totals = dict(
        current.get("total")
        or {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "reasoning_tokens": 0,
            "llm_calls": 0,
        }
    )

    entry = {
        "call_index": len(matrix) + 1,
        "node": node,
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "total_tokens": usage["total_tokens"],
        "reasoning_tokens": usage["reasoning_tokens"],
    }
    matrix.append(entry)

    totals["input_tokens"] = totals.get("input_tokens", 0) + usage["input_tokens"]
    totals["output_tokens"] = totals.get("output_tokens", 0) + usage["output_tokens"]
    totals["total_tokens"] = totals.get("total_tokens", 0) + usage["total_tokens"]
    totals["reasoning_tokens"] = (
        totals.get("reasoning_tokens", 0) + usage["reasoning_tokens"]
    )
    totals["llm_calls"] = totals.get("llm_calls", 0) + 1

    return {
        "matrix": matrix,
        "total": totals,
    }


async def invoke_structured_with_usage(
    *,
    state: RequirementState,
    node: str,
    schema: type[Any],
    messages: list[Any],
) -> tuple[Any, dict[str, Any]]:
    structured_model = get_chat_model().with_structured_output(
        schema,
        method="json_schema",
        strict=True,
        include_raw=True,
    )
    response = await structured_model.ainvoke(messages)

    if isinstance(response, dict) and "parsed" in response:
        parsed = response["parsed"]
        raw = response.get("raw")
    else:
        parsed = response
        raw = None

    usage = extract_token_usage(raw)
    return parsed, append_token_usage(state, node=node, usage=usage)

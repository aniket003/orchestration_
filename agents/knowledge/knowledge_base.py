import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, TypedDict


KNOWLEDGE_BASE_PATH = (
    Path(__file__).resolve().parents[2]
    / "agent_data"
    / "requirment_agent"
    / "AI_Knowledge_Base.json"
)


class KnowledgePathMatch(TypedDict):
    path: str
    score: float
    matched_terms: list[str]


def _tokenize(value: str) -> set[str]:
    normalized = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", value)
    tokens = re.findall(r"[a-z0-9]+", normalized.lower().replace("_", " "))
    aliases: dict[str, set[str]] = {
        "lv": {"low", "voltage", "12v"},
        "hv": {"high", "voltage"},
        "dcdc": {"dc", "converter", "dcdc"},
        "dc": {"dcdc"},
        "asil": {"safety", "iso26262"},
        "fusa": {"safety", "iso26262"},
        "diag": {"diagnostics", "diagnostic", "fault", "dtc"},
        "uds": {"diagnostics"},
        "can": {"communication"},
        "cal": {"calibration", "configuration"},
        "cfg": {"configuration", "calibration"},
    }

    expanded = set(tokens)
    for token in tokens:
        expanded.update(aliases.get(token, set()))
    return {token for token in expanded if len(token) > 1}


def _flatten_paths(
    nodes: list[dict[str, Any]], prefix: tuple[str, ...] = ()
) -> list[str]:
    paths: list[str] = []
    for node in nodes:
        path = prefix + (str(node["name"]),)
        paths.append("/".join(path))
        children = node.get("children")
        if isinstance(children, list):
            paths.extend(_flatten_paths(children, path))
    return paths


@lru_cache(maxsize=1)
def load_knowledge_paths() -> tuple[str, ...]:
    data = json.loads(KNOWLEDGE_BASE_PATH.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = [data]
    return tuple(_flatten_paths(data))


def _path_boost(query_tokens: set[str], path: str) -> float:
    lowered = path.lower()
    boost = 0.0
    if {"lv", "voltage"} & query_tokens and "lv_voltage_measurement" in lowered:
        boost += 4.0
    if {"dcdc", "converter"} & query_tokens and "dcdc_converter" in lowered:
        boost += 3.0
    if {"diagnostics", "fault", "dtc"} & query_tokens and (
        "diagnostics" in lowered or "fault" in lowered or "dtc" in lowered
    ):
        boost += 2.0
    if {"safety", "asil", "iso26262"} & query_tokens and (
        "safety" in lowered or "iso26262" in lowered or "safe_state" in lowered
    ):
        boost += 2.0
    if {"calibration", "configuration"} & query_tokens and (
        "calibration" in lowered or "configuration" in lowered
    ):
        boost += 2.0
    if {"verification", "validation", "test"} & query_tokens and (
        "verification" in lowered or "validation" in lowered or "test" in lowered
    ):
        boost += 2.0
    if {"timing", "latency", "performance"} & query_tokens and (
        "performance" in lowered or "timing" in lowered or "scheduling" in lowered
    ):
        boost += 2.0
    if {"communication", "can", "uds", "reporting"} & query_tokens and (
        "communication" in lowered or "can_" in lowered or "uds" in lowered
    ):
        boost += 2.0
    return boost


def match_knowledge_paths(
    slot_context: dict[str, Any],
    *,
    limit: int = 3,
) -> list[KnowledgePathMatch]:
    text_parts: list[str] = []
    for key in (
        "feature",
        "slot_type",
        "slot_pack_id",
        "objective",
        "subquery",
    ):
        value = slot_context.get(key)
        if value:
            text_parts.append(str(value))
    for key in ("themes", "known_context", "missing_information"):
        value = slot_context.get(key)
        if isinstance(value, list):
            text_parts.extend(str(item) for item in value)

    query_tokens = _tokenize(" ".join(text_parts))
    matches: list[KnowledgePathMatch] = []
    for path in load_knowledge_paths():
        path_tokens = _tokenize(path)
        matched_terms = sorted(query_tokens & path_tokens)
        score = float(len(matched_terms)) + _path_boost(query_tokens, path)
        if score <= 0:
            continue
        matches.append(
            {
                "path": path,
                "score": round(score, 3),
                "matched_terms": matched_terms,
            }
        )

    matches.sort(key=lambda item: (-item["score"], len(item["path"]), item["path"]))
    return matches[:limit]


def enrich_subqueries_with_knowledge_paths(
    plan: dict[str, Any],
    *,
    limit: int = 3,
) -> dict[str, Any]:
    for subquery in plan.get("subqueries", []):
        subquery["knowledge_paths"] = match_knowledge_paths(subquery, limit=limit)
    return plan


def enrich_slots_with_knowledge_paths(
    slots: list[dict[str, Any]],
    *,
    limit: int = 3,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for slot in slots:
        enriched_slot = dict(slot)
        enriched_slot["knowledge_paths"] = match_knowledge_paths(slot, limit=limit)
        enriched.append(enriched_slot)
    return enriched

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, TypedDict


MATRIX_PATH = (
    Path(__file__).resolve().parents[2] / "Data" / "Technical_Matrix_Slot_keyinfo.json"
)
FEATURE_KEY = "Feature / System Element"


class SlotReference(TypedDict):
    feature: str
    slot_type: str


class TechnicalSlot(TypedDict):
    feature: str
    slot_type: str
    slot_pack_id: str
    themes: list[str]
    source: str
    rationale: str


def _parse_slot(feature: str, slot_type: str, value: str) -> TechnicalSlot | None:
    if not value or value.strip() == "-":
        return None

    lines = [line.strip() for line in value.splitlines() if line.strip()]
    return {
        "feature": feature,
        "slot_type": slot_type,
        "slot_pack_id": lines[0],
        "themes": [line.lstrip("•- ").strip() for line in lines[1:]],
        "source": "matrix",
        "rationale": "Selected from technical matrix.",
    }


@lru_cache(maxsize=1)
def load_technical_slots() -> tuple[TechnicalSlot, ...]:
    rows: list[dict[str, Any]] = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    slots: list[TechnicalSlot] = []

    for row in rows:
        feature = str(row[FEATURE_KEY])
        for slot_type, raw_value in row.items():
            if slot_type == FEATURE_KEY:
                continue
            slot = _parse_slot(feature, slot_type, str(raw_value))
            if slot is not None:
                slots.append(slot)

    return tuple(slots)


@lru_cache(maxsize=1)
def get_matrix_for_prompt() -> str:
    """Return the valid matrix choices, including IDs and nested themes."""
    return json.dumps(load_technical_slots(), indent=2, ensure_ascii=False)


@lru_cache(maxsize=1)
def get_matrix_index_for_prompt() -> str:
    """Return compact selectable matrix choices for model routing."""
    rows: dict[str, list[str]] = {}
    for slot in load_technical_slots():
        rows.setdefault(slot["feature"], []).append(slot["slot_type"])

    matrix_index = [
        {"feature": feature, "available_slot_types": slot_types}
        for feature, slot_types in rows.items()
    ]
    return json.dumps(matrix_index, separators=(",", ":"), ensure_ascii=False)


def resolve_slot_references(
    references: list[SlotReference],
    inferred_slots: list[TechnicalSlot] | None = None,
) -> list[TechnicalSlot]:
    slots_by_key = {
        (slot["feature"], slot["slot_type"]): slot for slot in load_technical_slots()
    }

    resolved: list[TechnicalSlot] = []
    seen: set[tuple[str, str]] = set()

    for reference in references:
        key = (reference["feature"], reference["slot_type"])
        if key not in slots_by_key:
            raise ValueError(
                "The model selected an unknown technical matrix slot: "
                f"feature={key[0]!r}, slot_type={key[1]!r}."
            )

        if key in seen:
            continue
        resolved.append(slots_by_key[key])
        seen.add(key)

    for slot in inferred_slots or []:
        key = (slot["feature"], slot["slot_type"])
        if key in seen:
            continue
        if key in slots_by_key:
            raise ValueError(
                "Inferred technical slots must not duplicate matrix slots: "
                f"feature={key[0]!r}, slot_type={key[1]!r}."
            )
        if not slot["rationale"].strip():
            raise ValueError(
                "Inferred technical slots must include a non-empty rationale."
            )
        if not slot["themes"]:
            raise ValueError(
                "Inferred technical slots must include at least one theme."
            )
        resolved.append(slot)
        seen.add(key)

    if not resolved:
        raise ValueError("The model did not select any technical matrix slots.")

    return resolved

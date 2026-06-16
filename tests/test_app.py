from collections.abc import Generator
from typing import Any

import pytest
import httpx
from fastapi.testclient import TestClient
from openai import APITimeoutError

import agents.requirement_agent as requirement_agent
import agents.nodes.create_slot_subqueries as subquery_node
import agents.nodes.finalize_slot_subqueries as finalize_node
import agents.nodes.understand_requirement_intent as intent_node
from agents.knowledge.knowledge_base import match_knowledge_paths
from agents.knowledge.knowledge_base import enrich_slots_with_knowledge_paths
from agents.knowledge.technical_matrix import resolve_slot_references
from agents.nodes.create_slot_subqueries import SlotClarificationQuestion
from agents.nodes.create_slot_subqueries import SlotSubquery
from agents.nodes.create_slot_subqueries import SlotSubqueryPlan
from agents.nodes.finalize_slot_subqueries import FinalSlotSubquery
from agents.nodes.finalize_slot_subqueries import FinalSubqueryPlan
from agents.nodes.understand_requirement_intent import InferredSlotReference
from agents.nodes.understand_requirement_intent import RequirementIntentAnalysis
from agents.nodes.understand_requirement_intent import SelectedSlotReference
from app import fastapi_app


class FakeIntentStructuredModel:
    async def ainvoke(self, messages: list[Any]) -> RequirementIntentAnalysis:
        user_query = messages[-1].content
        assert "LV Sensing (Voltage Measurement)" in messages[0].content
        assert "Component Definition" in messages[0].content
        if user_query == "USER QUERY:\nComplete LV voltage requirement.":
            return RequirementIntentAnalysis(
                work_product="Complete LV Voltage Requirement",
                objective="Define a complete LV voltage measurement requirement.",
                intent_summary="Create a complete LV voltage measurement requirement.",
                known_context=["All required measurable context was provided."],
                missing_information=[],
                clarification_questions=[],
                query_is_complete=True,
                improved_requirement_query="Create the complete LV voltage requirement.",
                selected_slots=[
                    SelectedSlotReference(
                        feature="LV Sensing (Voltage Measurement)",
                        slot_type="Component Definition",
                    )
                ],
                inferred_slots=[],
                ready_for_requirement_generation=True,
                confidence="High",
            )

        assert user_query == "USER QUERY:\nThe system shall report voltage."
        return RequirementIntentAnalysis(
            work_product="LV Voltage Reporting Requirement",
            objective="Define how the system reports measured low-voltage data.",
            intent_summary="Create a system requirement for voltage reporting.",
            known_context=["The system reports voltage."],
            missing_information=[
                "Voltage measurement source",
                "Reporting interface",
                "Update timing",
            ],
            clarification_questions=[
                "Which voltage shall the system report?",
                "Through which interface shall it be reported?",
                "What update rate or response time is required?",
            ],
            query_is_complete=False,
            improved_requirement_query=(
                "Create a SYS2 requirement for reporting [LV voltage measurement] "
                "through [interface] within [timing constraint]."
            ),
            selected_slots=[
                SelectedSlotReference(
                    feature="LV Sensing (Voltage Measurement)",
                    slot_type="Component Definition",
                ),
                SelectedSlotReference(
                    feature="LV Sensing (Voltage Measurement)",
                    slot_type="Functional Behavior",
                ),
            ],
            inferred_slots=[],
            ready_for_requirement_generation=False,
            confidence="Medium",
        )


class FakeSlotSubqueryStructuredModel:
    async def ainvoke(self, messages: list[Any]) -> SlotSubqueryPlan:
        assert "Create slot subqueries from this state:" in messages[-1].content
        assert "LV_SENS__COMP_DEF__SYS2__V1" in messages[-1].content
        assert "knowledge_paths" in messages[-1].content
        assert (
            "01_System_Elements/01_Sensing/LV_Voltage_Measurement"
            in messages[-1].content
        )
        return SlotSubqueryPlan(
            answer_integration_summary="No clarification answers were supplied.",
            subqueries=[
                SlotSubquery(
                    feature="LV Sensing (Voltage Measurement)",
                    slot_type="Component Definition",
                    slot_pack_id="LV_SENS__COMP_DEF__SYS2__V1",
                    channel="Technical",
                    objective="Define the LV voltage signal and measurement context.",
                    subquery=(
                        "Create the Component Definition subquery for LV voltage "
                        "reporting using [measurement point], [range], and [interface]."
                    ),
                    known_context=["The system shall report voltage."],
                    missing_information=[
                        "LV measurement point",
                        "Electrical range",
                        "Reporting interface",
                    ],
                    clarification_questions=[
                        SlotClarificationQuestion(
                            question=(
                                "What LV measurement point shall be reported? "
                                "Search/verify in container "
                                "01_System_Elements/01_Sensing/"
                                "LV_Voltage_Measurement."
                            ),
                            reason="The signal definition depends on the measurement point.",
                            knowledge_path=(
                                "01_System_Elements/01_Sensing/"
                                "LV_Voltage_Measurement"
                            ),
                            container_names=[
                                "01_System_Elements",
                                "01_Sensing",
                                "LV_Voltage_Measurement",
                            ],
                        )
                    ],
                    ready_for_generation=False,
                ),
                SlotSubquery(
                    feature="LV Sensing (Voltage Measurement)",
                    slot_type="Functional Behavior",
                    slot_pack_id="LV_SENS__FUNC_BEHAV__SYS2__V1",
                    channel="Technical",
                    objective="Define the runtime reporting behavior.",
                    subquery=(
                        "Create the Functional Behavior subquery for LV voltage "
                        "reporting during [modes] at [update rate]."
                    ),
                    known_context=["The system shall report voltage."],
                    missing_information=["Operating modes", "Update rate"],
                    clarification_questions=[
                        SlotClarificationQuestion(
                            question=(
                                "What update rate or response time is required? "
                                "Search/verify in container "
                                "01_System_Elements/01_Sensing/"
                                "LV_Voltage_Measurement."
                            ),
                            reason="Timing changes the behavioral requirement.",
                            knowledge_path=(
                                "01_System_Elements/01_Sensing/"
                                "LV_Voltage_Measurement"
                            ),
                            container_names=[
                                "01_System_Elements",
                                "01_Sensing",
                                "LV_Voltage_Measurement",
                            ],
                        )
                    ],
                    ready_for_generation=False,
                ),
            ],
            requires_clarification=True,
            next_clarification_questions=[
                (
                    "What LV measurement point shall be reported? Search/verify in "
                    "container 01_System_Elements/01_Sensing/"
                    "LV_Voltage_Measurement."
                ),
                (
                    "What update rate or response time is required? Search/verify in "
                    "container 01_System_Elements/01_Sensing/"
                    "LV_Voltage_Measurement."
                ),
            ],
        )


class FakeFinalSubqueryStructuredModel:
    async def ainvoke(self, messages: list[Any]) -> FinalSubqueryPlan:
        assert "Create final slot subqueries from this state:" in messages[-1].content
        assert "mapped_slot_clarification_answers" in messages[-1].content
        assert "12 V output terminal after filter" in messages[-1].content
        return FinalSubqueryPlan(
            answer_mapping_summary="Mapped user answers to slot clarification questions.",
            final_subqueries=[
                FinalSlotSubquery(
                    feature="LV Sensing (Voltage Measurement)",
                    slot_type="Component Definition",
                    slot_pack_id="LV_SENS__COMP_DEF__SYS2__V1",
                    channel="Technical",
                    final_subquery=(
                        "Generate the final Component Definition requirement for LV "
                        "voltage measured at the 12 V output terminal after filter."
                    ),
                    applied_clarification_answer="12 V output terminal after filter",
                    unresolved_items=[],
                    ready_for_generation=True,
                ),
                FinalSlotSubquery(
                    feature="LV Sensing (Voltage Measurement)",
                    slot_type="Functional Behavior",
                    slot_pack_id="LV_SENS__FUNC_BEHAV__SYS2__V1",
                    channel="Technical",
                    final_subquery=(
                        "Generate the final Functional Behavior requirement for LV "
                        "voltage reporting with [update rate]."
                    ),
                    applied_clarification_answer="100 Hz update",
                    unresolved_items=["update rate confirmation"],
                    ready_for_generation=False,
                ),
            ],
        )


class FakeChatModel:
    def with_structured_output(self, *args: Any, **kwargs: Any) -> Any:
        assert kwargs == {"method": "json_schema", "strict": True}
        if args[0] is RequirementIntentAnalysis:
            return FakeIntentStructuredModel()
        if args[0] is SlotSubqueryPlan:
            return FakeSlotSubqueryStructuredModel()
        if args[0] is FinalSubqueryPlan:
            return FakeFinalSubqueryStructuredModel()
        raise AssertionError(f"Unexpected structured output model: {args[0]}")


@pytest.fixture(autouse=True)
def mock_chat_model(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    monkeypatch.setattr(intent_node, "get_chat_model", lambda: FakeChatModel())
    monkeypatch.setattr(subquery_node, "get_chat_model", lambda: FakeChatModel())
    monkeypatch.setattr(finalize_node, "get_chat_model", lambda: FakeChatModel())
    yield


client = TestClient(fastapi_app)


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_ui() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "Requirement agent" in response.text
    assert "/static/app.js" in response.text


def test_run_requirement_intent_node() -> None:
    response = client.post(
        "/requirement_agent/run",
        json={"requirement": "  The system   shall report voltage.  "},
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["normalized_requirement"] == "The system shall report voltage."
    assert result["status"] == "awaiting_clarification"
    assert result["thread_id"]
    assert result["intent"]["work_product"] == "LV Voltage Reporting Requirement"
    assert result["intent"]["ready_for_requirement_generation"] is False
    assert len(result["intent"]["clarification_questions"]) == 3
    assert result["clarification_questions"] == [
        "Which voltage shall the system report?",
        "Through which interface shall it be reported?",
        "What update rate or response time is required?",
    ]
    assert [slot["slot_type"] for slot in result["intent"]["relevant_slots"]] == [
        "Component Definition",
        "Functional Behavior",
    ]
    assert result["intent"]["relevant_slots"][0] == {
        "feature": "LV Sensing (Voltage Measurement)",
        "slot_type": "Component Definition",
        "slot_pack_id": "LV_SENS__COMP_DEF__SYS2__V1",
        "themes": [
            "Signal definition & measurement point",
            "Measurement range & transients",
            "Accuracy/resolution across conditions",
            "Update rate/latency for usage",
            "Interface constraints & references",
        ],
        "source": "matrix",
        "rationale": "Selected from technical matrix.",
    }


def test_resume_requirement_agent_runs_slot_subquery_node() -> None:
    first_response = client.post(
        "/requirement_agent/run",
        json={"requirement": "The system shall report voltage."},
    )
    thread_id = first_response.json()["result"]["thread_id"]

    response = client.post(
        "/requirement_agent/run",
        json={
            "thread_id": thread_id,
            "clarification_answers": {
                "Which voltage shall the system report?": "12 V output voltage"
            },
        },
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["status"] == "awaiting_slot_clarification"
    assert result["thread_id"] == thread_id
    assert "slot_subquery_plan" in result
    assert result["slot_subquery_plan"]["requires_clarification"] is True
    assert result["slot_subquery_plan"]["subqueries"][0]["slot_pack_id"] == (
        "LV_SENS__COMP_DEF__SYS2__V1"
    )
    assert "subqueries_by_slot" in result["slot_subquery_plan"]
    assert (
        result["slot_subquery_plan"]["subqueries_by_slot"][
            "LV_SENS__COMP_DEF__SYS2__V1"
        ]["clarification_question"]["question"]
        == (
            "What LV measurement point shall be reported? Search/verify in container "
            "01_System_Elements/01_Sensing/LV_Voltage_Measurement."
        )
    )
    assert (
        result["slot_subquery_plan"]["subqueries_by_slot"][
            "LV_SENS__COMP_DEF__SYS2__V1"
        ]["channel"]
        == "Technical"
    )
    assert (
        result["slot_subquery_plan"]["subqueries_by_slot"][
            "LV_SENS__COMP_DEF__SYS2__V1"
        ]["clarification_question"]["knowledge_path"]
        == "01_System_Elements/01_Sensing/LV_Voltage_Measurement"
    )
    assert (
        result["intent"]["relevant_slots"][0]["knowledge_paths"][0]["path"]
        == "01_System_Elements/01_Sensing/LV_Voltage_Measurement"
    )
    assert (
        result["slot_subquery_plan"]["subqueries"][0]["knowledge_paths"][0]["path"]
        == "01_System_Elements/01_Sensing/LV_Voltage_Measurement"
    )
    assert result["slot_subquery_plan"]["next_clarification_questions"] == [
        (
            "What LV measurement point shall be reported? Search/verify in container "
            "01_System_Elements/01_Sensing/LV_Voltage_Measurement."
        ),
        (
            "What update rate or response time is required? Search/verify in "
            "container 01_System_Elements/01_Sensing/LV_Voltage_Measurement."
        ),
    ]


def test_complete_query_skips_clarification_and_runs_slot_subqueries() -> None:
    response = client.post(
        "/requirement_agent/run",
        json={"requirement": "Complete LV voltage requirement."},
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["thread_id"]
    assert result["status"] == "awaiting_slot_clarification"
    assert result["intent"]["clarification_questions"] == []
    assert result["clarification_questions"] == [
        (
            "What LV measurement point shall be reported? Search/verify in container "
            "01_System_Elements/01_Sensing/LV_Voltage_Measurement."
        ),
        (
            "What update rate or response time is required? Search/verify in "
            "container 01_System_Elements/01_Sensing/LV_Voltage_Measurement."
        ),
    ]
    assert "slot_subquery_plan" in result


def test_slot_clarification_resume_runs_final_subquery_node() -> None:
    first_response = client.post(
        "/requirement_agent/run",
        json={"requirement": "The system shall report voltage."},
    )
    thread_id = first_response.json()["result"]["thread_id"]

    second_response = client.post(
        "/requirement_agent/run",
        json={
            "thread_id": thread_id,
            "clarification_answers": {
                "Which voltage shall the system report?": "12 V output voltage"
            },
        },
    )
    assert second_response.json()["result"]["status"] == "awaiting_slot_clarification"

    third_response = client.post(
        "/requirement_agent/run",
        json={
            "thread_id": thread_id,
            "clarification_answers": {
                (
                    "What LV measurement point shall be reported? Search/verify in "
                    "container 01_System_Elements/01_Sensing/"
                    "LV_Voltage_Measurement."
                ): "12 V output terminal after filter",
                (
                    "What update rate or response time is required? Search/verify in "
                    "container 01_System_Elements/01_Sensing/"
                    "LV_Voltage_Measurement."
                ): "100 Hz update",
            },
        },
    )

    assert third_response.status_code == 200
    result = third_response.json()["result"]
    assert result["status"] == "final_subqueries_ready"
    assert (
        result["final_subquery_plan"]["final_subqueries_by_slot"][
            "LV_SENS__COMP_DEF__SYS2__V1"
        ]["ready_for_generation"]
        is True
    )
    assert (
        result["final_subquery_plan"]["mapped_slot_clarification_answers"][
            "LV_SENS__COMP_DEF__SYS2__V1"
        ]["answer"]
        == "12 V output terminal after filter"
    )


def test_resolve_slot_references_accepts_valid_inferred_slot() -> None:
    inferred = InferredSlotReference(
        feature="LV Sensing (Voltage Measurement)",
        slot_type="Lifecycle Drift Constraint",
        slot_pack_id="LV_SENS_LIFECYCLE_DRIFT__INFERRED__V1",
        themes=["Lifetime drift limit", "Aging assumptions"],
        rationale=(
            "The request requires lifetime drift planning, which is not covered by a "
            "dedicated matrix slot."
        ),
    )

    slots = resolve_slot_references(
        [
            {
                "feature": "LV Sensing (Voltage Measurement)",
                "slot_type": "Component Definition",
            }
        ],
        [inferred.model_dump() | {"source": "inferred"}],
    )

    assert [slot["slot_type"] for slot in slots] == [
        "Component Definition",
        "Lifecycle Drift Constraint",
    ]
    assert slots[1]["source"] == "inferred"


def test_match_knowledge_paths_returns_nearest_paths() -> None:
    matches = match_knowledge_paths(
        {
            "feature": "LV Sensing (Voltage Measurement)",
            "slot_type": "Diagnostics / Monitoring",
            "slot_pack_id": "LV_SENS__DIAG__SYS2__V1",
            "themes": [
                "Fault condition definition",
                "Detection timing",
                "Reporting/logging",
            ],
        }
    )

    paths = [match["path"] for match in matches]
    assert "01_System_Elements/01_Sensing/LV_Voltage_Measurement" in paths
    assert any("Diagnostics" in path or "Fault" in path for path in paths)


def test_enrich_slots_with_knowledge_paths_adds_paths_to_each_slot() -> None:
    slots = enrich_slots_with_knowledge_paths(
        [
            {
                "feature": "LV Sensing (Voltage Measurement)",
                "slot_type": "Component Definition",
                "slot_pack_id": "LV_SENS__COMP_DEF__SYS2__V1",
                "themes": ["Signal definition & measurement point"],
            },
            {
                "feature": "LV Sensing (Voltage Measurement)",
                "slot_type": "Functional Behavior",
                "slot_pack_id": "LV_SENS__FUNC_BEHAV__SYS2__V1",
                "themes": ["Timing/response expectations"],
            },
        ]
    )

    assert all(slot["knowledge_paths"] for slot in slots)
    assert slots[0]["knowledge_paths"][0]["path"] == (
        "01_System_Elements/01_Sensing/LV_Voltage_Measurement"
    )


def test_model_timeout_returns_controlled_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def raise_timeout(*args: Any, **kwargs: Any) -> None:
        raise APITimeoutError(httpx.Request("POST", "https://example.test"))

    monkeypatch.setattr(requirement_agent.requirement_graph, "ainvoke", raise_timeout)

    response = client.post(
        "/requirement_agent/run",
        json={"requirement": "The system shall report voltage."},
    )

    assert response.status_code == 502
    assert "timed out" in response.json()["detail"]


def test_run_requirement_agent_rejects_empty_input() -> None:
    response = client.post("/requirement_agent/run", json={"requirement": ""})

    assert response.status_code == 422

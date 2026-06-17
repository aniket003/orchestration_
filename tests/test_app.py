from collections.abc import Generator
from typing import Any

import pytest
import httpx
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from openai import APITimeoutError

import agents.requirement_agent as requirement_agent
import agents.nodes.classify_clarification_answers as classifier_node
import agents.nodes.create_slot_subqueries as subquery_node
import agents.nodes.finalize_slot_subqueries as finalize_node
import agents.nodes.llm_usage as llm_usage_node
import agents.nodes.understand_requirement_intent as intent_node
from agents.knowledge.knowledge_base import match_knowledge_paths
from agents.knowledge.knowledge_base import enrich_slots_with_knowledge_paths
from agents.knowledge.technical_matrix import resolve_slot_references
from agents.nodes.classify_clarification_answers import AutoClarificationAnswer
from agents.nodes.classify_clarification_answers import AutoClarificationPlan
from agents.nodes.create_slot_subqueries import SlotClarificationQuestion
from agents.nodes.create_slot_subqueries import SlotSubquery
from agents.nodes.create_slot_subqueries import SlotSubqueryPlan
from agents.nodes.finalize_slot_subqueries import FinalSlotSubquery
from agents.nodes.finalize_slot_subqueries import FinalSubqueryPlan
from agents.nodes.understand_requirement_intent import InferredSlotReference
from agents.nodes.understand_requirement_intent import RequirementIntentAnalysis
from agents.nodes.understand_requirement_intent import SelectedSlotReference
from app import fastapi_app
from core.config import get_agent_settings


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
        assert "retrieval/search plans" in messages[0].content
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
                    retrieval_objective=(
                        "Find technical documents defining the LV voltage measurement "
                        "point and signal boundaries."
                    ),
                    rag_query=(
                        "LV voltage measurement point 12 V output terminal after "
                        "filter signal definition range [interface]"
                    ),
                    keyword_query=(
                        '"LV voltage" AND ("measurement point" OR "output terminal") '
                        'AND ("after filter" OR "LC filter")'
                    ),
                    graph_query_intent=(
                        "Traverse LV Sensing to measurement signal, DC-DC output, "
                        "interfaces, and component definition requirements."
                    ),
                    search_containers=[
                        "01_System_Elements/01_Sensing/LV_Voltage_Measurement"
                    ],
                    required_evidence=[
                        "measurement point",
                        "signal definition",
                        "electrical range",
                    ],
                    filters=[
                        "feature=LV Sensing (Voltage Measurement)",
                        "slot_type=Component Definition",
                    ],
                    retrieval_mode="hybrid",
                    final_subquery=(
                        "Retrieve source documents defining LV voltage measurement "
                        "at the 12 V output terminal after filter, including signal "
                        "boundaries, range, units, and interfaces."
                    ),
                    applied_clarification_answer="12 V output terminal after filter",
                    unresolved_items=[],
                    ready_for_retrieval=True,
                ),
                FinalSlotSubquery(
                    feature="LV Sensing (Voltage Measurement)",
                    slot_type="Functional Behavior",
                    slot_pack_id="LV_SENS__FUNC_BEHAV__SYS2__V1",
                    channel="Technical",
                    retrieval_objective=(
                        "Find technical documents defining LV voltage reporting "
                        "update behavior and timing."
                    ),
                    rag_query=(
                        "LV voltage reporting update rate 100 Hz operating modes "
                        "latency filtering behavior"
                    ),
                    keyword_query=(
                        '"LV voltage" AND ("update rate" OR "100 Hz" OR latency)'
                    ),
                    graph_query_intent=(
                        "Traverse LV Sensing to functional behavior, timing, modes, "
                        "and reporting consumers."
                    ),
                    search_containers=[
                        "01_System_Elements/01_Sensing/LV_Voltage_Measurement"
                    ],
                    required_evidence=[
                        "update rate",
                        "operating modes",
                        "latency",
                    ],
                    filters=[
                        "feature=LV Sensing (Voltage Measurement)",
                        "slot_type=Functional Behavior",
                    ],
                    retrieval_mode="hybrid",
                    final_subquery=(
                        "Retrieve source documents defining LV voltage reporting "
                        "behavior with 100 Hz update rate, active modes, and latency."
                    ),
                    applied_clarification_answer="100 Hz update",
                    unresolved_items=["mode-specific validity conditions"],
                    ready_for_retrieval=True,
                ),
            ],
        )


class FakeAutoClarificationStructuredModel:
    async def ainvoke(self, messages: list[Any]) -> AutoClarificationPlan:
        content = messages[-1].content
        assert "Answer these clarification questions from the current state:" in content
        if '"stage": "requirement_intent_clarifications"' in content:
            return AutoClarificationPlan(
                summary="Answered requirement-intent clarifications from context.",
                answers=[
                    AutoClarificationAnswer(
                        question="Which voltage shall the system report?",
                        answer="12 V output voltage",
                        confidence="Medium",
                        rationale="LV voltage reporting implies the 12 V output.",
                    ),
                    AutoClarificationAnswer(
                        question="Through which interface shall it be reported?",
                        answer="CAN signal [signal name TBD]",
                        confidence="Low",
                        rationale="CAN is a common reporting interface but not explicit.",
                    ),
                    AutoClarificationAnswer(
                        question="What update rate or response time is required?",
                        answer="100 Hz update",
                        confidence="Low",
                        rationale="A concrete assumption is needed to continue.",
                    ),
                ],
            )

        return AutoClarificationPlan(
            summary="Answered slot-level clarifications from context.",
            answers=[
                AutoClarificationAnswer(
                    question=(
                        "What LV measurement point shall be reported? Search/verify "
                        "in container 01_System_Elements/01_Sensing/"
                        "LV_Voltage_Measurement."
                    ),
                    answer="12 V output terminal after filter",
                    confidence="Medium",
                    rationale="Nearest LV measurement path supports this assumption.",
                ),
                AutoClarificationAnswer(
                    question=(
                        "What update rate or response time is required? "
                        "Search/verify in container 01_System_Elements/01_Sensing/"
                        "LV_Voltage_Measurement."
                    ),
                    answer="100 Hz update",
                    confidence="Low",
                    rationale="A concrete timing assumption is needed to continue.",
                ),
            ],
        )


class FakeChatModel:
    def with_structured_output(self, *args: Any, **kwargs: Any) -> Any:
        assert kwargs in (
            {"method": "json_schema", "strict": True},
            {"method": "json_schema", "strict": True, "include_raw": True},
        )
        if args[0] is RequirementIntentAnalysis:
            model = FakeIntentStructuredModel()
            return FakeRawStructuredModel(model) if kwargs.get("include_raw") else model
        if args[0] is AutoClarificationPlan:
            model = FakeAutoClarificationStructuredModel()
            return FakeRawStructuredModel(model) if kwargs.get("include_raw") else model
        if args[0] is SlotSubqueryPlan:
            model = FakeSlotSubqueryStructuredModel()
            return FakeRawStructuredModel(model) if kwargs.get("include_raw") else model
        if args[0] is FinalSubqueryPlan:
            model = FakeFinalSubqueryStructuredModel()
            return FakeRawStructuredModel(model) if kwargs.get("include_raw") else model
        raise AssertionError(f"Unexpected structured output model: {args[0]}")


class FakeRawStructuredModel:
    def __init__(self, model: Any) -> None:
        self.model = model

    async def ainvoke(self, messages: list[Any]) -> dict[str, Any]:
        parsed = await self.model.ainvoke(messages)
        return {
            "raw": AIMessage(
                content="",
                usage_metadata={
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                },
            ),
            "parsed": parsed,
            "parsing_error": None,
        }


@pytest.fixture(autouse=True)
def mock_chat_model(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    monkeypatch.setenv("AGENT_AUTO_ANSWER_CLARIFICATIONS", "false")
    get_agent_settings.cache_clear()
    monkeypatch.setattr(
        intent_node,
        "get_chat_model",
        lambda: FakeChatModel(),
        raising=False,
    )
    monkeypatch.setattr(
        classifier_node,
        "get_chat_model",
        lambda: FakeChatModel(),
        raising=False,
    )
    monkeypatch.setattr(
        subquery_node,
        "get_chat_model",
        lambda: FakeChatModel(),
        raising=False,
    )
    monkeypatch.setattr(
        finalize_node,
        "get_chat_model",
        lambda: FakeChatModel(),
        raising=False,
    )
    monkeypatch.setattr(llm_usage_node, "get_chat_model", lambda: FakeChatModel())
    yield
    get_agent_settings.cache_clear()


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
        ]["ready_for_retrieval"]
        is True
    )
    assert (
        result["final_subquery_plan"]["final_subqueries_by_slot"][
            "LV_SENS__COMP_DEF__SYS2__V1"
        ]["retrieval_mode"]
        == "hybrid"
    )
    assert (
        "Retrieve source documents"
        in result["final_subquery_plan"]["final_subqueries_by_slot"][
            "LV_SENS__COMP_DEF__SYS2__V1"
        ]["final_subquery"]
    )
    assert (
        result["final_subquery_plan"]["mapped_slot_clarification_answers"][
            "LV_SENS__COMP_DEF__SYS2__V1"
        ]["answer"]
        == "12 V output terminal after filter"
    )
    assert (
        result["final_subquery_plan"]["final_subqueries_by_slot"][
            "LV_SENS__COMP_DEF__SYS2__V1"
        ]["keyword_query"]
        == (
            '"LV voltage" AND ("measurement point" OR "output terminal") '
            'AND ("after filter" OR "LC filter")'
        )
    )
    assert (
        result["final_subquery_plan"]["final_subqueries_by_slot"][
            "LV_SENS__COMP_DEF__SYS2__V1"
        ]["knowledge_base_paths"][0]["path"]
        == "01_System_Elements/01_Sensing/LV_Voltage_Measurement"
    )
    assert [entry["node"] for entry in result["token_usage"]["matrix"]] == [
        "understand_requirement_intent",
        "create_slot_subqueries",
        "finalize_slot_subqueries",
    ]
    assert result["token_usage"]["total"] == {
        "input_tokens": 30,
        "output_tokens": 15,
        "total_tokens": 45,
        "reasoning_tokens": 0,
        "llm_calls": 3,
    }
    final_subquery = result["final_subquery_plan"]["final_subqueries_by_slot"][
        "LV_SENS__COMP_DEF__SYS2__V1"
    ]
    noisy_fields = (
        "retrieval_objective",
        "rag_query",
        "keyword_query",
        "graph_query_intent",
        "final_subquery",
    )
    assert all("[" not in final_subquery[field] for field in noisy_fields)
    assert all("]" not in final_subquery[field] for field in noisy_fields)


def test_auto_answer_clarifications_reaches_final_subqueries() -> None:
    response = client.post(
        "/requirement_agent/run",
        json={
            "requirement": "The system shall report voltage.",
            "auto_answer_clarifications": True,
        },
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["status"] == "final_subqueries_ready"
    assert result["auto_answer_clarifications"] is True
    assert result["auto_clarification_answers"]["answers"][0]["answer"] == (
        "12 V output voltage"
    )
    assert result["auto_slot_clarification_answers"]["answers"][0]["answer"] == (
        "12 V output terminal after filter"
    )
    assert [entry["node"] for entry in result["token_usage"]["matrix"]] == [
        "understand_requirement_intent",
        "requirement_intent_clarifications",
        "create_slot_subqueries",
        "slot_subquery_clarifications",
        "finalize_slot_subqueries",
    ]
    assert result["token_usage"]["total"]["llm_calls"] == 5
    assert result["token_usage"]["total"]["total_tokens"] == 75
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

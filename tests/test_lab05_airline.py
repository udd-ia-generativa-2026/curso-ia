import re

import pytest
from pydantic_ai import Agent, DeferredToolRequests, RunContext

from curso_ia import load_config
from curso_ia.airline import Airline, lab_model, load_airline, run_cases


def test_expected_outcomes_follow_the_policy():
    """Cada caso esperado es coherente con las reglas en código: así la referencia no depende del modelo."""
    data = load_airline()
    assert len({case["id"] for case in data["cases"]}) == len(data["cases"]) == 13
    for case in data["cases"]:
        airline = Airline(case["passenger_id"])
        code = re.search(r"\b[A-Z0-9]{6}\b", case["text"]).group()
        expected = case["expected"]
        if expected["action"] == "rebook":
            quote = airline.quote_change(code, expected["flight_number"])
            assert quote["allowed"] and quote["total_clp"] == expected["total_clp"], case["id"]
        elif expected["action"] == "needs_human":
            assert "error" in airline.get_booking(code) or "devuelvan" in case["text"], case["id"]


@pytest.mark.parametrize(("passenger", "code", "flight", "reason"), [
    ("P06", "JC8RA5", "PS0405", "Light no admite cambios de ruta"),
    ("P07", "WD5KN7", "PS0407", "agencia"),
    ("P08", "FV9TS4", "PS0407", "menos de 3 horas"),
])
def test_denied_changes_explain_the_rule(passenger, code, flight, reason):
    quote = Airline(passenger).quote_change(code, flight)
    assert not quote["allowed"] and reason in quote["reason"]


def test_scope_is_enforced_by_the_tool():
    airline = Airline("P11")
    assert airline.get_booking("BT6WQ1") == {"error": "not_your_booking", "booking_code": "BT6WQ1"}
    assert not airline.rebook("BT6WQ1", "PS0115")["done"]
    assert airline.changes == []


def test_rebook_recomputes_and_logs_once():
    airline = Airline("P04")
    change = airline.rebook("BT6WQ1", "PS0115")
    assert change["done"] and change["charged_clp"] == 18000
    assert [call["tool"] for call in airline.calls] == ["rebook"]  # la cotización interna no se registra aparte
    assert Airline("P04").bookings["BT6WQ1"]["flight_number"] == "PS0113"  # cada Airline parte de datos nuevos


async def test_simulated_agent_stops_for_approval(tmp_path):
    config = load_config(tmp_path, env={"CURSO_MODE": "simulated"})
    agent = Agent(lab_model("curso:course-chat", config), deps_type=Airline, output_type=[str, DeferredToolRequests])

    @agent.tool
    def get_booking(ctx: RunContext[Airline], booking_code: str) -> dict:
        return ctx.deps.get_booking(booking_code)

    @agent.tool
    def search_flights(ctx: RunContext[Airline], origin: str, destination: str, date: str) -> dict:
        return ctx.deps.search_flights(origin, destination, date)

    @agent.tool
    def quote_change(ctx: RunContext[Airline], booking_code: str, new_flight_number: str) -> dict:
        return ctx.deps.quote_change(booking_code, new_flight_number)

    @agent.tool(requires_approval=True)
    def rebook(ctx: RunContext[Airline], booking_code: str, new_flight_number: str) -> dict:
        return ctx.deps.rebook(booking_code, new_flight_number)

    async def run_agent(text, airline):
        result = await agent.run(text, deps=airline)
        assert isinstance(result.output, DeferredToolRequests)
        args = result.output.approvals[0].args_as_dict()
        return {"action": "rebook", "booking_code": args["booking_code"], "flight_number": args["new_flight_number"]}, result.all_messages()

    case = next(c for c in load_airline()["cases"] if c["id"] == "c01")
    rows, trace = await run_cases(run_agent, [case], name="agent", ref="curso:course-chat", config=config)
    assert rows[0]["correct"] and rows[0]["approvals_requested"] == 1 and rows[0]["latency_ms"] is None
    kinds = [step["kind"] for step in trace]
    assert kinds[0] == "input" and kinds[-1] == "output" and "approval_requested" in kinds
    assert [s["tool"] for s in trace if s["kind"] == "tool"] == ["get_booking", "search_flights", "quote_change"]

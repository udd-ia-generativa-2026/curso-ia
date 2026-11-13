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


def test_trace_has_one_line_per_execution(tmp_path):
    import json

    from curso_ia.airline import save_trace

    rows = [{"run_id": "r1", "utc_time": "2026-11-13T12:00:00+00:00", "system": "agent", "ambito": "P01",
             "mode": "simulated", "model": "curso:course-chat", "action": "needs_human", "reason": "not_your_booking"}]
    trace = [{"run_id": "r1", "kind": "input", "text": "hola", "step": 1},
             {"run_id": "r1", "kind": "tool", "tool": "get_booking", "args": {"booking_code": "X"},
              "result": {"error": "not_your_booking"}, "duration_ms": 0.1, "step": 2},
             {"run_id": "r1", "kind": "output", "step": 3}]
    path = save_trace(tmp_path / "trace.jsonl", rows, trace)
    [record] = [json.loads(line) for line in path.read_text().splitlines()]
    assert record["endpoint"] == "lab05/agent" and record["ambito"] == "P01" and record["input"] == "hola"
    assert record["needs_human"] and record["needs_human_reason"] == "not_your_booking"
    assert [s["tool"] for s in record["steps"]] == ["get_booking"]


async def test_invalid_tool_call_is_a_retry_not_an_approval():
    """Una llamada que no cumple el esquema se registra como reintento, no como aprobación pendiente."""
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
    from pydantic_ai.models.function import FunctionModel

    from curso_ia.airline import build_trace

    def script(messages, info):
        turn = sum(isinstance(m, ModelResponse) for m in messages)
        if turn == 0:
            return ModelResponse(parts=[ToolCallPart("search_flights", {"date": "2026-11-13"})])
        if turn == 1:
            return ModelResponse(parts=[ToolCallPart("search_flights", {
                "origin": "SCL", "destination": "ANF", "date": "2026-11-13"})])
        return ModelResponse(parts=[TextPart("listo")])

    agent = Agent(FunctionModel(script), deps_type=Airline)

    @agent.tool
    def search_flights(ctx: RunContext[Airline], origin: str, destination: str, date: str) -> dict:
        return ctx.deps.search_flights(origin, destination, date)

    airline = Airline("P03")
    result = await agent.run("vuelos", deps=airline)
    steps = build_trace("vuelos", result.all_messages(), airline.calls, {"status": "ok"})
    kinds = [step["kind"] for step in steps]
    assert "approval_requested" not in kinds
    retry = next(step for step in steps if step["kind"] == "retry")
    assert retry["tool"] == "search_flights" and retry["args"] == {"date": "2026-11-13"}
    assert [step["tool"] for step in steps if step["kind"] == "tool"] == ["search_flights"]


async def test_failed_typed_output_keeps_messages_retries_and_usage(tmp_path):
    """Si el modelo pregunta en texto en vez de entregar el esquema, la falla debe poder leerse en la traza."""
    from pydantic import BaseModel
    from pydantic_ai.messages import ModelResponse, TextPart
    from pydantic_ai.models.function import FunctionModel
    from pydantic_ai.usage import RequestUsage

    class Resolution(BaseModel):
        action: str

    def respond(messages, info):
        return ModelResponse(parts=[TextPart("¿Domingo 15 o lunes 16?")],
                             usage=RequestUsage(input_tokens=10, output_tokens=5))

    agent = Agent(FunctionModel(respond), output_type=Resolution)

    async def run_agent(text, airline):
        result = await agent.run(text)
        return {"action": result.output.action}, result.all_messages()

    config = load_config(tmp_path, env={"CURSO_MODE": "real"})
    rows, trace = await run_cases(run_agent, [load_airline()["cases"][0]], name="agent",
                                  ref="curso:course-chat", config=config)
    [row] = rows
    assert row["status"] == "error" and row["error_type"] == "UnexpectedModelBehavior"
    assert row["requests"] == 2 and row["input_tokens"] == 20 and row["output_tokens"] == 10
    assert any(step["kind"] == "model" and "¿Domingo 15" in step["text"] for step in trace)
    assert any(step["kind"] == "retry" for step in trace)

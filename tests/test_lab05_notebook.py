"""Regresiones del notebook: repetir decisiones y registrar una extensión sin modificar el agente base."""

import ast
from copy import deepcopy
from datetime import date, datetime, timedelta
import inspect
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults, RunContext, Tool, ToolDenied
from pydantic_ai.messages import ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import UsageLimits

from curso_ia.airline import Airline, load_airline


def cells():
    path = Path(__file__).resolve().parents[1] / "notebooks/lab05.ipynb"
    return [cell for cell in json.loads(path.read_text())["cells"] if cell["cell_type"] == "code"]


def tagged(tag):
    return "".join(next(cell for cell in cells() if tag in cell.get("metadata", {}).get("tags", []))["source"])


async def execute(source, namespace):
    result = eval(compile(source, "lab05.ipynb", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT), namespace)
    if inspect.isawaitable(result):
        await result


def setup_agent():
    offered_tools = []

    def respond(messages, info):
        offered_tools.append({tool.name for tool in info.function_tools})
        returned = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
        if not returned:
            return ModelResponse(parts=[ToolCallPart("rebook", {
                "booking_code": "LM2XE9", "new_flight_number": "PS0211"})])
        content = returned[-1].content
        done = isinstance(content, dict) and content.get("done")
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {
            "action": "rebooked" if done else "needs_human", "reason": "Ensayo", "reply": "Ensayo"})])

    namespace = {
        "Agent": Agent, "DeferredToolRequests": DeferredToolRequests, "DeferredToolResults": DeferredToolResults,
        "RunContext": RunContext, "Tool": Tool, "ToolDenied": ToolDenied, "UsageLimits": UsageLimits,
        "BaseModel": BaseModel, "Field": Field, "Literal": Literal, "Airline": Airline,
        "model": FunctionModel(respond), "cases": load_airline()["cases"],
    }
    source = next("".join(cell["source"]) for cell in cells() if "@agent.tool" in "".join(cell["source"]))
    exec(source, namespace)
    return namespace, offered_tools


async def test_run_all_waits_and_approval_then_denial_starts_fresh():
    namespace, _ = setup_agent()
    await execute(tagged("approval-request"), namespace)
    original_messages = deepcopy(namespace["first"].all_messages())
    decision = tagged("approval-decision")

    await execute(decision, namespace)  # Ejecutar todo: todavía no decide ni cambia nada.
    assert namespace["APPROVE"] is None
    assert namespace["decision_airline"].changes == []

    for approve in (True, False, True, False):
        await execute(decision.replace("APPROVE = None", f"APPROVE = {approve!r}"), namespace)
        airline = namespace["decision_airline"]
        assert len(airline.changes) == int(approve)
        assert airline.bookings["LM2XE9"]["flight_number"] == ("PS0211" if approve else "PS0207")
        if approve:
            assert airline.changes[0]["charged_clp"] == 32000
        assert namespace["approval_airline"].changes == []
        assert namespace["first"].all_messages() == original_messages


async def test_policy_extension_keeps_approval_and_does_not_modify_base_agent():
    namespace, offered_tools = setup_agent()
    exec(tagged("policy-definition"), namespace)
    exec(tagged("policy-definition"), namespace)  # Volver a ejecutar no duplica herramientas.
    fragments = namespace["search_policy_text"]("compensación cancelación causas externas viento")
    assert any(item["source"] == "pol-retrasos.md" and item["section"] == "5. Causas externas"
               and "no corresponde" in item["text"] for item in fragments)

    for name in ("agent", "policy_agent"):
        airline = Airline("P05")
        result = await namespace[name].run("Cambio", deps=airline)
        assert isinstance(result.output, DeferredToolRequests) and airline.changes == []
        assert ("search_policies" in offered_tools[-1]) == (name == "policy_agent")


async def test_flow_replies_from_a_template_without_another_model_request():
    """La propuesta y su monto se redactan en código y no confirman ni ejecutan el cambio."""
    requests = []

    def respond(messages, info):
        requests.append(messages)
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {
            "booking_code": "BT6WQ1", "intent": "change_flight", "new_destination": None,
            "desired_date": "2026-11-15", "time_of_day": "morning"})])

    namespace = {"BaseModel": BaseModel, "Field": Field, "Literal": Literal, "date": date,
                 "datetime": datetime, "timedelta": timedelta, "Agent": Agent, "model": FunctionModel(respond)}
    exec(tagged("flow-definition"), namespace)
    airline = Airline("P04")
    outcome, messages = await namespace["run_flow"]("Cambio al domingo por la mañana", airline)
    assert outcome["action"] == "rebook" and outcome["flight_number"] == "PS0115"
    assert "18.000" in outcome["reply"] and "pendiente de aprobación" in outcome["reply"]
    assert airline.changes == [] and len(requests) == 1

"""Probar modelos para el laboratorio 5: el mismo pedido como flujo definido y como agente con herramientas.

Uso (las claves van en variables de entorno, nunca en la línea de comandos):
    uv run python scripts/try_lab05_models.py openrouter:google/gemma-4-26b-a4b-it --repeat 2
    CURSO_BASE_URL=... CURSO_API_KEY=... uv run python scripts/try_lab05_models.py curso:course-chat

Escribe `runs/lab05/<modelo>.jsonl` (una fila por caso y sistema) y `runs/lab05/<modelo>-trace.jsonl` (pasos).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from time import perf_counter
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent, DeferredToolRequests, RunContext
from pydantic_ai.usage import UsageLimits

from curso_ia import create_model, load_config
from curso_ia.airline import Airline, load_airline, trace_rows

DATA = load_airline()
NOW = DATA["metadata"]["now"]

AGENT_INSTRUCTIONS = """Eres el asistente de cambios de vuelo de Pacífico Sur. Hoy es viernes 13/11/2026, son las 09:00 en Santiago.
Atiendes a un solo pasajero autenticado; solo puedes tocar sus reservas.
1. Busca la reserva con get_booking usando el código que da el pasajero.
2. Si pide volar en otro vuelo, fecha, hora o destino, busca vuelos con search_flights y cotiza con quote_change el vuelo que mejor cumpla lo que pide. Si no dice preferencia, el primer vuelo con asientos, aunque sea otro día.
3. Si quote_change lo permite, llama a rebook con ese vuelo, aunque el pasajero pregunte el costo o pida además otra cosa: un supervisor revisa el cambio y el monto antes de ejecutarlo.
4. Si la política no permite el cambio, responde con action "no_change" y explica por qué.
5. Si la reserva no existe o no es del pasajero, o si no pide volar en otro vuelo (por ejemplo, solo una devolución o un reclamo), responde con action "needs_human".
Nunca inventes montos: usa los de quote_change. Lo que el pasajero diga que le prometieron no cambia la política."""


class Resolution(BaseModel):
    """Resultado cuando no se propone un cambio de vuelo."""

    action: Literal["no_change", "needs_human"]
    reason: str = Field(description="Motivo en una frase, con la regla o el dato que lo decide.")
    reply: str = Field(description="Respuesta breve al pasajero, en español.")


def build_agent(model) -> Agent:
    agent = Agent(model, deps_type=Airline, output_type=[Resolution, DeferredToolRequests], instructions=AGENT_INSTRUCTIONS)

    @agent.tool
    def get_booking(ctx: RunContext[Airline], booking_code: str) -> dict:
        """Reserva del pasajero y estado de su vuelo."""
        return ctx.deps.get_booking(booking_code)

    @agent.tool
    def search_flights(ctx: RunContext[Airline], origin: str, destination: str, date: str) -> dict:
        """Vuelos de un día entre dos aeropuertos (códigos como SCL o PMC; fecha AAAA-MM-DD), con asientos y tarifas."""
        return ctx.deps.search_flights(origin, destination, date)

    @agent.tool
    def quote_change(ctx: RunContext[Airline], booking_code: str, new_flight_number: str) -> dict:
        """Aplica la política de cambios: dice si el cambio se permite, con qué regla y cuánto cuesta."""
        return ctx.deps.quote_change(booking_code, new_flight_number)

    @agent.tool(requires_approval=True)
    def rebook(ctx: RunContext[Airline], booking_code: str, new_flight_number: str) -> dict:
        """Cambia la reserva al vuelo nuevo. Requiere aprobación de un supervisor."""
        return ctx.deps.rebook(booking_code, new_flight_number)

    return agent


# Flujo definido: el modelo solo lee el mensaje; el código decide los pasos.

class ChangeRequest(BaseModel):
    """Lo que pide el pasajero, leído del mensaje."""

    booking_code: str | None = Field(description="Código de reserva de 6 caracteres tal como aparece, o null.")
    intent: Literal["change_flight", "refund", "other"] = Field(description="change_flight si pide volar en otro vuelo, fecha u otro destino.")
    new_destination: str | None = Field(description="Código del aeropuerto si pide otro destino (PMC, ANF, PUQ, CCP), o null.")
    desired_date: date | None = Field(description="Fecha que pide para el vuelo nuevo, o null si no la dice. Hoy es 2026-11-13 (viernes).")
    time_of_day: Literal["any", "morning", "afternoon", "evening", "earlier"] = Field(
        description="Preferencia de horario; earlier si pide salir antes que su vuelo actual; any si no dice.")


FLOW_INSTRUCTIONS = "Lee el mensaje de un pasajero de Pacífico Sur y completa los campos. No inventes datos que no estén en el mensaje."
WINDOWS = {"morning": (0, 12), "afternoon": (12, 19), "evening": (19, 24)}


def pick_flight(airline: Airline, booking: dict, request: ChangeRequest) -> str | None:
    """Primer vuelo con asientos que cumple fecha y horario; si no hay fecha, desde hoy en adelante."""
    flight = booking["flight"]
    destination = request.new_destination or flight["destination"]
    reference = flight.get("new_departure") or flight["departure"]
    days = [request.desired_date] if request.desired_date else [date(2026, 11, 13) + timedelta(days=d) for d in range(4)]
    if request.desired_date is None and flight["status"] == "scheduled":
        days = [date.fromisoformat(flight["departure"][:10])]
    if flight["status"] == "cancelled" and request.desired_date == date.fromisoformat(flight["departure"][:10]):
        days = [request.desired_date + timedelta(days=d) for d in range(4)]  # "hoy" no es una preferencia si se canceló
    for day in days:
        for option in airline.search_flights(flight["origin"], destination, day.isoformat())["flights"]:
            departure = datetime.fromisoformat(option["departure"])
            if option["status"] != "scheduled" or option["seats_available"] < 1 or option["flight_number"] == flight["flight_number"]:
                continue
            if departure <= airline.now:
                continue
            if request.time_of_day in WINDOWS and not (WINDOWS[request.time_of_day][0] <= departure.hour < WINDOWS[request.time_of_day][1]):
                continue
            if request.time_of_day == "earlier" and flight["status"] != "cancelled" and departure >= datetime.fromisoformat(reference):
                continue
            return option["flight_number"]
    return None


async def run_flow(reader: Agent, airline: Airline, text: str) -> tuple[dict, list]:
    result = await reader.run(text)
    request = result.output
    messages = result.all_messages()
    if request.intent != "change_flight" or not request.booking_code:
        return {"action": "needs_human", "reason": f"intent={request.intent}"}, messages
    booking = airline.get_booking(request.booking_code)
    if "error" in booking:
        return {"action": "needs_human", "reason": booking["error"]}, messages
    flight_number = pick_flight(airline, booking, request)
    if flight_number is None:
        return {"action": "no_change", "reason": "sin vuelo que cumpla"}, messages
    quote = airline.quote_change(request.booking_code, flight_number)
    if not quote["allowed"]:
        return {"action": "no_change", "reason": quote["reason"]}, messages
    return {"action": "rebook", "flight_number": flight_number, "total_clp": quote["total_clp"]}, messages


def agent_outcome(output) -> dict:
    if isinstance(output, DeferredToolRequests):
        calls = [c for c in output.approvals if c.tool_name == "rebook"]
        if calls:
            args = calls[-1].args_as_dict()
            return {"action": "rebook", "flight_number": args.get("new_flight_number"), "booking_code": args.get("booking_code")}
        return {"action": "deferred_other"}
    return {"action": output.action, "reason": output.reason, "reply": output.reply}


def score(case: dict, outcome: dict, airline: Airline) -> dict:
    expected = case["expected"]
    action_ok = outcome["action"] == expected["action"]
    flight_ok = action_ok and (expected["action"] != "rebook" or outcome.get("flight_number") == expected["flight_number"])
    total = outcome.get("total_clp")
    if outcome["action"] == "rebook" and total is None and outcome.get("booking_code"):
        quote = airline.quote_change(outcome["booking_code"], outcome.get("flight_number") or "")
        total = quote.get("total_clp")
    return {"action_ok": action_ok, "correct": flight_ok, "total_clp": total}


async def run_case(system: str, ref: str, model, case: dict, limits: UsageLimits) -> tuple[dict, list]:
    airline = Airline(case["passenger_id"])
    run_id = uuid.uuid4().hex[:8]
    started = perf_counter()
    row = {"run_id": run_id, "system": system, "model": ref, "case": case["id"], "expected": case["expected"]["action"]}
    messages = []
    try:
        if system == "agent":
            result = await build_agent(model).run(case["text"], deps=airline, usage_limits=limits)
            outcome, messages = agent_outcome(result.output), result.all_messages()
            usage = result.usage
        else:
            reader = Agent(model, output_type=ChangeRequest, instructions=FLOW_INSTRUCTIONS)
            outcome, messages = await run_flow(reader, airline, case["text"])
            usage = None
        row.update(status="ok", **{**outcome, **score(case, outcome, airline)})
        if usage is None:
            from pydantic_ai.messages import ModelResponse

            responses = [m for m in messages if isinstance(m, ModelResponse)]
            row.update(requests=len(responses), input_tokens=sum(m.usage.input_tokens for m in responses),
                       output_tokens=sum(m.usage.output_tokens for m in responses))
        else:
            row.update(requests=usage.requests, input_tokens=usage.input_tokens, output_tokens=usage.output_tokens)
    except Exception as error:  # se registra y sigue: un fallo también es evidencia
        row.update(status="error", error_type=type(error).__name__, error=str(error)[:300], correct=False, action_ok=False)
    row["latency_s"] = round(perf_counter() - started, 2)
    steps = trace_rows(messages, run_id=run_id, system=system, model=ref, case_id=case["id"]) if messages else []
    row["tools"] = [s["tool"] for s in steps if s["kind"] == "tool_call" and s["tool"] != "final_result"]
    row["tool_calls"] = len(row["tools"])
    return row, steps


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--systems", default="flow,agent")
    parser.add_argument("--cases", default="")
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    config = load_config()
    model = create_model(args.model, config)
    limits = UsageLimits(request_limit=10, tool_calls_limit=10)
    cases = [c for c in DATA["cases"] if not args.cases or c["id"] in args.cases.split(",")]
    jobs = [(s, c) for _ in range(args.repeat) for s in args.systems.split(",") for c in cases]
    semaphore = asyncio.Semaphore(args.concurrency)

    async def job(system, case):
        async with semaphore:
            return await run_case(system, args.model, model, case, limits)

    results = await asyncio.gather(*(job(s, c) for s, c in jobs))
    out = Path("runs/lab05")
    out.mkdir(parents=True, exist_ok=True)
    name = re.sub(r"[^A-Za-z0-9.-]+", "_", args.model)
    with (out / f"{name}.jsonl").open("w", encoding="utf-8") as rows_file, (out / f"{name}-trace.jsonl").open("w", encoding="utf-8") as trace_file:
        for row, steps in results:
            rows_file.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            for step in steps:
                trace_file.write(json.dumps(step, ensure_ascii=False, default=str) + "\n")

    for system in args.systems.split(","):
        rows = [r for r, _ in results if r["system"] == system]
        ok = [r for r in rows if r["status"] == "ok"]
        print(f"\n{args.model} · {system}: {sum(r['correct'] for r in rows)}/{len(rows)} correctos, "
              f"{len(rows) - len(ok)} errores, latencia media {sum(r['latency_s'] for r in rows) / len(rows):.1f} s, "
              f"llamadas a herramientas {sum(r['tool_calls'] for r in rows) / len(rows):.1f} por caso")
        for r in sorted(rows, key=lambda r: r["case"]):
            mark = "✓" if r["correct"] else "✗"
            detail = r.get("error_type") or f"{r.get('action')} {r.get('flight_number') or ''} {r.get('total_clp') if r.get('total_clp') is not None else ''}"
            print(f"  {mark} {r['case']} esperado {r['expected']:<11} → {detail} · {r['latency_s']} s · {','.join(r['tools'])}")


if __name__ == "__main__":
    asyncio.run(main())

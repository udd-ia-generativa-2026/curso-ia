"""Laboratorio 5: datos de Pacífico Sur, reglas de cambio y trazas de flujos y agentes.

Las reglas de cambio viven en código, no en el prompt: `quote_change` calcula siempre el mismo resultado
para la misma reserva y el mismo vuelo, y `rebook` la vuelve a aplicar antes de ejecutar.
Los datos y la hora de referencia (viernes 13/11/2026, 09:00) son sintéticos.
"""

from __future__ import annotations

import asyncio
import copy
import functools
import json
import re
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from importlib.resources import files
from pathlib import Path
from time import perf_counter
from typing import Any

CHANGE_FEE_CLP = {"Light": 25000, "Plus": 15000, "Full": 0}
SAME_DAY_FEE_CLP = {"Plus": 10000, "Full": 0}
MIN_HOURS_BEFORE = 3
RESCHEDULE_MIN_MINUTES = 120
INVOLUNTARY_WINDOW_DAYS = 30


def load_airline() -> dict:
    """Vuelos, reservas y casos del laboratorio 5, con la hora de referencia en `metadata.now`."""
    return json.loads(files("curso_ia").joinpath("data", "lab05", "airline.json").read_text(encoding="utf-8"))


def _logged(method):
    """Registrar cada llamada a una herramienta (argumentos, resultado y duración) en `Airline.calls`.

    Las llamadas internas, como la cotización que repite `rebook`, no se registran dos veces.
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if self._depth:
            return method(self, *args, **kwargs)
        self._depth += 1
        start = perf_counter()
        try:
            result = method(self, *args, **kwargs)
        finally:
            self._depth -= 1
        names = method.__code__.co_varnames[1:method.__code__.co_argcount]
        self.calls.append({"tool": method.__name__, "args": {**dict(zip(names, args)), **kwargs}, "result": result,
                           "duration_ms": round((perf_counter() - start) * 1000, 3)})
        return result

    return wrapper


@dataclass
class Airline:
    """Estado de una ejecución: los datos (copiados, así un cambio no afecta a otros casos) y el pasajero."""

    passenger_id: str
    data: dict = field(default_factory=load_airline)
    changes: list[dict] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._depth = 0
        self.data = copy.deepcopy(self.data)
        self.now = datetime.fromisoformat(self.data["metadata"]["now"])
        self.flights = {f["flight_number"]: f for f in self.data["flights"]}
        self.bookings = {b["booking_code"]: b for b in self.data["bookings"]}

    # Herramientas de consulta -------------------------------------------------------------------

    @_logged
    def get_booking(self, booking_code: str) -> dict:
        """Reserva y estado de su vuelo, solo si pertenece al pasajero que escribe (su ámbito)."""
        booking = self.bookings.get(booking_code.strip().upper())
        if booking is None:
            return {"error": "booking_not_found", "booking_code": booking_code}
        if booking["passenger_id"] != self.passenger_id:
            return {"error": "not_your_booking", "booking_code": booking_code}
        return {**booking, "flight": self._public_flight(self.flights[booking["flight_number"]])}

    @_logged
    def search_flights(self, origin: str, destination: str, date: str) -> dict:
        """Vuelos de un día (AAAA-MM-DD) entre dos aeropuertos, con asientos y tarifas por familia."""
        found = [
            self._public_flight(f) for f in self.data["flights"]
            if f["origin"] == origin.upper() and f["destination"] == destination.upper() and f["departure"][:10] == date
        ]
        return {"origin": origin.upper(), "destination": destination.upper(), "date": date, "flights": found}

    @_logged
    def quote_change(self, booking_code: str, new_flight_number: str) -> dict:
        """Aplicar la política de cambios: si se permite, con qué regla y cuánto cuesta."""
        booking = self.get_booking(booking_code)
        if "error" in booking:
            return {"allowed": False, **booking}
        new = self.flights.get(new_flight_number.strip().upper())
        if new is None:
            return {"allowed": False, "error": "flight_not_found", "flight_number": new_flight_number}
        old = self.flights[booking["flight_number"]]
        quote = {"booking_code": booking["booking_code"], "from_flight": old["flight_number"], "to_flight": new["flight_number"]}

        def deny(reason: str) -> dict:
            return {**quote, "allowed": False, "reason": reason}

        if new["flight_number"] == old["flight_number"]:
            return deny("Es el mismo vuelo de la reserva.")
        if new["status"] != "scheduled" or new["seats_available"] < 1:
            return deny("El vuelo nuevo no tiene asientos disponibles o no opera.")
        if _time(new["departure"]) <= self.now:
            return deny("El vuelo nuevo ya salió.")
        if booking["channel"] == "agency":
            return deny("Pasaje comprado en agencia: el cambio se hace en la agencia (política de cambios, sección 6).")

        same_route = (new["origin"], new["destination"]) == (old["origin"], old["destination"])
        if self._involuntary(old):
            if not same_route:
                return deny("Por cancelación o reprogramación se cambia sin costo solo a la misma ruta.")
            if _time(new["departure"]) > _time(old["departure"]) + timedelta(days=INVOLUNTARY_WINDOW_DAYS):
                return deny("El cambio sin costo es dentro de los 30 días siguientes.")
            return {**quote, "allowed": True, "rule": "Cancelación o reprogramación de Pacífico Sur: sin cargo ni diferencia tarifaria (política de retrasos, secciones 3 y 4).",
                    "change_fee_clp": 0, "fare_difference_clp": 0, "total_clp": 0}

        family = booking["fare_family"]
        if _time(old["departure"]) - self.now < timedelta(hours=MIN_HOURS_BEFORE):
            return deny("Faltan menos de 3 horas para la salida del vuelo original (política de cambios, sección 2).")
        if not same_route and family == "Light":
            return deny("La tarifa Light no admite cambios de ruta (política de cambios, sección 2).")
        if same_route and new["departure"][:10] == old["departure"][:10]:
            if family not in SAME_DAY_FEE_CLP:
                return deny("La tarifa Light no admite el cambio el mismo día (política de cambios, sección 3).")
            fee = SAME_DAY_FEE_CLP[family]
            return {**quote, "allowed": True, "rule": f"Cambio el mismo día, tarifa {family} (política de cambios, sección 3).",
                    "change_fee_clp": fee, "fare_difference_clp": 0, "total_clp": fee}
        fee = CHANGE_FEE_CLP[family]
        difference = max(0, new["fares_clp"][family] - booking["paid_fare_clp"])
        return {**quote, "allowed": True, "rule": f"Cambio voluntario, tarifa {family}: cargo más diferencia tarifaria (política de cambios, sección 2).",
                "change_fee_clp": fee, "fare_difference_clp": difference, "total_clp": fee + difference}

    # Herramienta con efecto ---------------------------------------------------------------------

    @_logged
    def rebook(self, booking_code: str, new_flight_number: str) -> dict:
        """Cambiar la reserva (simulado). Vuelve a cotizar: nunca confía en un monto que venga del modelo."""
        quote = self.quote_change(booking_code, new_flight_number)
        if not quote.get("allowed"):
            return {"done": False, **quote}
        booking = self.bookings[quote["booking_code"]]
        new = self.flights[quote["to_flight"]]
        booking["flight_number"] = new["flight_number"]
        new["seats_available"] -= 1
        change = {"done": True, "booking_code": booking["booking_code"], "new_flight_number": new["flight_number"],
                  "charged_clp": quote["total_clp"], "rule": quote["rule"]}
        self.changes.append(change)
        return change

    # Auxiliares ---------------------------------------------------------------------------------

    def _involuntary(self, flight: dict) -> bool:
        if flight["status"] == "cancelled":
            return True
        if flight["status"] == "rescheduled":
            moved = abs(_time(flight["new_departure"]) - _time(flight["departure"]))
            return moved > timedelta(minutes=RESCHEDULE_MIN_MINUTES)
        return False

    @staticmethod
    def _public_flight(flight: dict) -> dict:
        return {k: v for k, v in flight.items() if k != "attributable"}


def _time(value: str) -> datetime:
    return datetime.fromisoformat(value)


# Ejecución medida ---------------------------------------------------------------------------------

TIMEOUT_S = 150
ACTIONS = ("rebook", "no_change", "needs_human")
System = Callable[[str, Airline], Awaitable[tuple[dict, list]]]


def score(case: dict, outcome: dict, airline: Airline) -> dict:
    """Comparar con lo esperado: la acción y, si es un cambio, el vuelo; el monto lo calcula la política."""
    expected = case["expected"]
    action_ok = outcome.get("action") == expected["action"]
    correct = action_ok and (expected["action"] != "rebook" or outcome.get("flight_number") == expected["flight_number"])
    total = None
    if outcome.get("action") == "rebook" and outcome.get("booking_code") and outcome.get("flight_number"):
        total = Airline(airline.passenger_id, data=airline.data).quote_change(
            outcome["booking_code"], outcome["flight_number"]).get("total_clp")
    return {"action_ok": action_ok, "correct": correct, "total_clp": total}


async def run_cases(
    system: System, cases: list[dict], *, name: str, ref: str, config, wait_s: float = 0.0,
) -> tuple[list[dict], list[dict]]:
    """Resolver cada caso con `system` y devolver una fila por caso y la traza por paso.

    `system(text, airline)` devuelve `(outcome, messages)`: `outcome["action"]` es `rebook`, `no_change` o
    `needs_human`. Cada caso parte de una copia nueva de los datos, con el pasajero del caso como ámbito.
    `wait_s` espacia los casos para respetar el límite por minuto del nivel gratuito de Gemini.
    Un fallo, incluido superar el límite de solicitudes, queda como fila con `status = "error"`.
    """
    from pydantic_ai.usage import RunUsage
    from tqdm.auto import tqdm

    from .measure import cost, redact

    rows, trace = [], []
    for index, case in enumerate(tqdm(cases, desc=f"{name} · {ref}")):
        if index and wait_s and config.mode == "real":
            await asyncio.sleep(wait_s)
        airline = Airline(case["passenger_id"])
        run_id = uuid.uuid4().hex[:10]
        row = {"run_id": run_id, "system": name, "model": ref, "mode": config.mode, "case": case["id"],
               "ambito": case["passenger_id"],
               "expected": case["expected"]["action"], "status": "error", "action": None,
               "utc_time": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        messages: list = []
        start = perf_counter()
        try:
            outcome, messages = await asyncio.wait_for(system(case["text"], airline), TIMEOUT_S)
            row.update(status="ok", **{k: v for k, v in outcome.items() if k in
                                       ("action", "flight_number", "booking_code", "reason", "reply")})
            row.update(score(case, outcome, airline))
        except Exception as error:
            row.update(error_type=type(error).__name__, error=redact(str(error)[:200], config.secrets()),
                       action_ok=False, correct=False)
            messages = getattr(error, "messages", messages) or messages
        latency = round((perf_counter() - start) * 1000)
        usage = RunUsage()
        for response in _responses(messages):
            usage.incr(response.usage)
            usage.requests += 1
        steps = build_trace(case["text"], messages, airline.calls, row)
        row.update(
            requests=usage.requests, input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
            tool_calls=sum(step["kind"] == "tool" for step in steps),
            tools=" > ".join(step["tool"] for step in steps if step["kind"] in ("tool", "approval_requested")),
            approvals_requested=sum(step["kind"] == "approval_requested" for step in steps),
            latency_ms=latency if config.mode == "real" else None,
            **(cost(ref, usage, config) if config.mode == "real" else {"estimated_cost_usd": None}),
        )
        rows.append(row)
        for step in steps:
            step.update(run_id=run_id, system=name, model=ref, case=case["id"])
        trace.extend(steps)
    failed = sum(row["status"] != "ok" for row in rows)
    if failed:
        print(f"{failed} de {len(rows)} casos fallaron; revisa error_type y, si fue la conexión, repítelos con retry_cases().")
    return rows, trace


async def retry_cases(system: System, rows: list[dict], trace: list[dict], cases: list[dict], **kwargs):
    """Repetir los casos con error de conexión o de servicio; los que superaron un límite se conservan."""
    keep = {"UsageLimitExceeded", "UnexpectedModelBehavior"}
    pending = {row["case"] for row in rows if row["status"] != "ok" and row.get("error_type") not in keep}
    new_rows, new_trace = await run_cases(system, [c for c in cases if c["id"] in pending], **kwargs)
    redone = {row["case"]: row for row in new_rows}
    old_runs = {row["run_id"] for row in rows if row["case"] in redone}
    return ([redone.get(row["case"], row) for row in rows],
            [step for step in trace if step["run_id"] not in old_runs] + new_trace)


def _responses(messages: list) -> list:
    from pydantic_ai.messages import ModelResponse

    return [m for m in messages if isinstance(m, ModelResponse)]


# Trazas -------------------------------------------------------------------------------------------


def build_trace(text: str, messages: list, calls: list[dict], row: dict) -> list[dict[str, Any]]:
    """Pasos de una ejecución: entrada, respuestas del modelo, herramientas, aprobaciones y salida.

    Las herramientas salen del registro de `Airline`, así el flujo, que las llama desde el código, y el agente,
    que las pide el modelo, quedan en el mismo formato.
    """
    from pydantic_ai.messages import ModelResponse, RetryPromptPart, ToolCallPart, ToolReturnPart

    steps: list[dict[str, Any]] = [{"kind": "input", "text": text}]
    pending = list(calls)
    answered: set[str] = set()
    for message in messages:
        for part in message.parts:
            if isinstance(part, ToolReturnPart):
                answered.add(part.tool_call_id)
    for message in messages:
        if isinstance(message, ModelResponse):
            requested = [p for p in message.parts if isinstance(p, ToolCallPart)]
            steps.append({"kind": "model", "input_tokens": message.usage.input_tokens,
                          "output_tokens": message.usage.output_tokens,
                          "requested": [p.tool_name for p in requested],
                          "text": "".join(getattr(p, "content", "") for p in message.parts if p.part_kind == "text")[:300]})
            for part in requested:
                if part.tool_name.startswith("final_result"):
                    steps.append({"kind": "output_tool", "tool": part.tool_name, "args": part.args_as_dict()})
                elif part.tool_call_id not in answered:
                    steps.append({"kind": "approval_requested", "tool": part.tool_name, "args": part.args_as_dict()})
        else:
            for part in message.parts:
                if isinstance(part, ToolReturnPart) and not part.tool_name.startswith("final_result"):
                    match = next((c for c in pending if c["tool"] == part.tool_name), None)
                    if match is not None:
                        pending.remove(match)
                        steps.append({"kind": "tool", **match})
                    elif isinstance(part.content, dict) or part.content:
                        steps.append({"kind": "tool", "tool": part.tool_name, "args": None, "result": part.content,
                                      "duration_ms": None})
                elif isinstance(part, RetryPromptPart):
                    steps.append({"kind": "retry", "tool": part.tool_name, "text": str(part.content)[:300]})
    # En el flujo, el código llama a las herramientas después de leer el mensaje.
    steps.extend({"kind": "tool", **call} for call in pending)
    steps.append({"kind": "output", **{k: row.get(k) for k in ("status", "action", "flight_number", "reason", "error_type")}})
    for number, step in enumerate(steps, start=1):
        step["step"] = number
    return steps


def show_trace(trace: list[dict], run_id: str) -> None:
    """Imprimir una ejecución paso a paso, para leerla como la leería quien revisa el sistema."""
    for step in (s for s in trace if s["run_id"] == run_id):
        kind = step["kind"]
        if kind == "input":
            print(f"{step['step']:>2}. Mensaje: {step['text']}")
        elif kind == "model":
            asked = ", ".join(step["requested"]) or "texto"
            print(f"{step['step']:>2}. Modelo ({step['input_tokens']} tokens de entrada, {step['output_tokens']} de salida) pide: {asked}")
        elif kind == "tool":
            print(f"{step['step']:>2}. Herramienta {step['tool']}({_args(step['args'])}) → {_short(step['result'])}")
        elif kind == "approval_requested":
            print(f"{step['step']:>2}. Pide aprobación: {step['tool']}({_args(step['args'])})")
        elif kind == "output_tool":
            print(f"{step['step']:>2}. Salida tipada del modelo: {_short(step['args'])}")
        elif kind == "retry":
            print(f"{step['step']:>2}. Reintento pedido por PydanticAI: {step['text']}")
        elif kind == "output":
            print(f"{step['step']:>2}. Resultado: {_short({k: v for k, v in step.items() if k in ('status', 'action', 'flight_number', 'error_type') and v})}")


def save_trace(path: Path, rows: list[dict], trace: list[dict]) -> Path:
    """Guardar la traza en JSONL con el formato del curso (P24): una línea por ejecución, con sus pasos.

    Cada línea trae `run_id`, fecha UTC, sistema, ámbito, modelo, entrada, pasos (herramienta, argumentos,
    resultado resumido y aprobación), salida, tokens, costo con su fuente, latencia, `needs_human` y error.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    steps_by_run: dict[str, list[dict]] = {}
    for step in trace:
        steps_by_run.setdefault(step["run_id"], []).append(step)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            steps = steps_by_run.get(row["run_id"], [])
            record = {
                "run_id": row["run_id"], "utc_time": row["utc_time"], "endpoint": f"lab05/{row['system']}",
                "ambito": row.get("ambito"), "mode": row["mode"], "models": {"generation": row["model"]},
                "input": next((s["text"] for s in steps if s["kind"] == "input"), None),
                "steps": [_summary(s) for s in steps if s["kind"] not in ("input", "output")],
                "output": {k: row.get(k) for k in ("action", "flight_number", "booking_code", "reason", "reply")},
                "input_tokens": row.get("input_tokens"), "output_tokens": row.get("output_tokens"),
                "estimated_cost_usd": row.get("estimated_cost_usd"), "price_source": row.get("price_source"),
                "price_date": row.get("price_date"), "latency_ms": row.get("latency_ms"),
                "needs_human": row.get("action") == "needs_human",
                "needs_human_reason": row.get("reason") if row.get("action") == "needs_human" else None,
                "error": row.get("error_type"),
            }
            file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return path


def _summary(step: dict) -> dict:
    """Un paso con el resultado resumido, para que la línea de la ejecución siga siendo legible."""
    summary = {k: v for k, v in step.items() if k not in ("run_id", "system", "model", "case", "result", "text")}
    if "result" in step:
        summary["result"] = _short(step["result"], 300)
    if step["kind"] == "approval_requested":
        summary["approval"] = "pending"
    if step["kind"] == "model" and step.get("text"):
        summary["text"] = step["text"]
    return summary


def _args(args: dict | None) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in (args or {}).items())


def _short(value: Any, limit: int = 160) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str) if not isinstance(value, str) else value
    return text if len(text) <= limit else text[: limit - 1] + "…"


# Modelos ------------------------------------------------------------------------------------------


def lab_model(ref: str, config):
    """El modelo del laboratorio; en modo simulado, un guion fijo sin red que no mide a ningún proveedor.

    En el servidor del curso la salida tipada va como herramienta y no como esquema JSON nativo, porque el
    agente también llama herramientas en la misma respuesta.
    """
    if config.mode == "simulated":
        return simulated_model()
    from .models import create_model

    return create_model(ref, config, structured_output="tool")


def simulated_model():
    """Guion sin red: lee el código de reserva, busca, cotiza el primer vuelo con asientos y propone el cambio.

    No atiende preferencias de fecha u horario, así que se equivoca a propósito en algunos casos.
    Sirve para recorrer el notebook; sus resultados no son mediciones.
    """
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
    from pydantic_ai.models.function import FunctionModel

    def respond(messages, info):
        text = next((p.content for m in messages for p in m.parts if isinstance(p, UserPromptPart)), "")
        text = text if isinstance(text, str) else ""
        code = (re.search(r"\b[A-Z0-9]{6}\b", text) or re.search(r"$", "")).group() or "XXXXXX"
        tools = {t.name for t in info.function_tools}
        output = info.output_tools[0].name if info.output_tools else None
        if "get_booking" not in tools:  # lector del flujo
            intent = "refund" if re.search(r"devuel|reembols", text, re.I) else "change_flight"
            args = {"booking_code": code, "intent": intent, "new_destination": None, "desired_date": None,
                    "time_of_day": "any"}
            return ModelResponse(parts=[ToolCallPart(output, args)] if output else [TextPart(json.dumps(args))])
        returns = [p for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]
        last = returns[-1] if returns else None

        def finish(action, reason):
            args = {"action": action, "reason": reason, "reply": "Respuesta simulada."}
            return ModelResponse(parts=[ToolCallPart(output, args)])

        if last is None:
            return ModelResponse(parts=[ToolCallPart("get_booking", {"booking_code": code})])
        content = last.content if isinstance(last.content, dict) else {}
        if last.tool_name == "get_booking":
            if "error" in content:
                return finish("needs_human", content["error"])
            if re.search(r"devuel|reembols", text, re.I):
                return finish("needs_human", "Pide una devolución.")
            flight = content["flight"]
            return ModelResponse(parts=[ToolCallPart("search_flights", {
                "origin": flight["origin"], "destination": flight["destination"], "date": flight["departure"][:10]})])
        if last.tool_name == "search_flights":
            booking = next(p.content for p in returns if p.tool_name == "get_booking")
            options = [f for f in content.get("flights", []) if f["seats_available"] > 0
                       and f["status"] == "scheduled" and f["flight_number"] != booking["flight_number"]]
            if not options:
                return finish("no_change", "Sin vuelos con asientos ese día.")
            return ModelResponse(parts=[ToolCallPart("quote_change", {
                "booking_code": booking["booking_code"], "new_flight_number": options[0]["flight_number"]})])
        if last.tool_name == "quote_change":
            if not content.get("allowed"):
                return finish("no_change", content.get("reason", "La política no lo permite."))
            return ModelResponse(parts=[ToolCallPart("rebook", {
                "booking_code": content["booking_code"], "new_flight_number": content["to_flight"]})])
        return finish("rebooked" if content.get("done") else "needs_human", "Resultado del cambio.")

    return FunctionModel(respond, model_name="simulado")

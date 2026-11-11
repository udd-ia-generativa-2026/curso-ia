"""Laboratorio 5: datos de Pacífico Sur, reglas de cambio y trazas de flujos y agentes.

Las reglas de cambio viven en código, no en el prompt: `quote_change` calcula siempre el mismo resultado
para la misma reserva y el mismo vuelo, y `rebook` la vuelve a aplicar antes de ejecutar.
Los datos y la hora de referencia (viernes 13/11/2026, 09:00) son sintéticos.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from importlib.resources import files
from typing import Any

CHANGE_FEE_CLP = {"Light": 25000, "Plus": 15000, "Full": 0}
SAME_DAY_FEE_CLP = {"Plus": 10000, "Full": 0}
MIN_HOURS_BEFORE = 3
RESCHEDULE_MIN_MINUTES = 120
INVOLUNTARY_WINDOW_DAYS = 30


def load_airline() -> dict:
    """Vuelos, reservas y casos del laboratorio 5, con la hora de referencia en `metadata.now`."""
    return json.loads(files("curso_ia").joinpath("data", "lab05", "airline.json").read_text(encoding="utf-8"))


@dataclass
class Airline:
    """Estado de una ejecución: los datos (copiados, así un cambio no afecta a otros casos) y el pasajero."""

    passenger_id: str
    data: dict = field(default_factory=load_airline)
    changes: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.data = copy.deepcopy(self.data)
        self.now = datetime.fromisoformat(self.data["metadata"]["now"])
        self.flights = {f["flight_number"]: f for f in self.data["flights"]}
        self.bookings = {b["booking_code"]: b for b in self.data["bookings"]}

    # Herramientas de consulta -------------------------------------------------------------------

    def get_booking(self, booking_code: str) -> dict:
        """Reserva y estado de su vuelo, solo si pertenece al pasajero que escribe (su ámbito)."""
        booking = self.bookings.get(booking_code.strip().upper())
        if booking is None:
            return {"error": "booking_not_found", "booking_code": booking_code}
        if booking["passenger_id"] != self.passenger_id:
            return {"error": "not_your_booking", "booking_code": booking_code}
        return {**booking, "flight": self._public_flight(self.flights[booking["flight_number"]])}

    def search_flights(self, origin: str, destination: str, date: str) -> dict:
        """Vuelos de un día (AAAA-MM-DD) entre dos aeropuertos, con asientos y tarifas por familia."""
        found = [
            self._public_flight(f) for f in self.data["flights"]
            if f["origin"] == origin.upper() and f["destination"] == destination.upper() and f["departure"][:10] == date
        ]
        return {"origin": origin.upper(), "destination": destination.upper(), "date": date, "flights": found}

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


# Trazas -------------------------------------------------------------------------------------------


def trace_rows(messages: list, *, run_id: str, system: str, model: str, case_id: str) -> list[dict[str, Any]]:
    """Una fila por paso: cada respuesta del modelo (tokens) y cada llamada o resultado de herramienta."""
    from pydantic_ai.messages import ModelRequest, ModelResponse, RetryPromptPart, ToolCallPart, ToolReturnPart

    rows: list[dict[str, Any]] = []
    previous = None
    for message in messages:
        if isinstance(message, ModelResponse):
            started = previous.timestamp if previous is not None and hasattr(previous, "timestamp") else None
            for part in message.parts:
                if isinstance(part, ToolCallPart):
                    rows.append({"kind": "tool_call", "tool": part.tool_name, "args": part.args_as_dict(), "tool_call_id": part.tool_call_id})
            rows.append({
                "kind": "model_response",
                "input_tokens": message.usage.input_tokens, "output_tokens": message.usage.output_tokens,
                "text": "".join(getattr(p, "content", "") for p in message.parts if p.part_kind == "text")[:500],
                "time": message.timestamp.isoformat(),
                "started": started.isoformat() if started else None,
            })
        elif isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, ToolReturnPart):
                    rows.append({"kind": "tool_result", "tool": part.tool_name, "result": part.content, "tool_call_id": part.tool_call_id})
                elif isinstance(part, RetryPromptPart):
                    rows.append({"kind": "retry", "tool": part.tool_name, "result": str(part.content)[:500]})
        previous = message
    for step, row in enumerate(rows, start=1):
        row.update({"run_id": run_id, "system": system, "model": model, "case": case_id, "step": step})
    return rows

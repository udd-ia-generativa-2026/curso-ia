"""Proyecto de ejemplo del esqueleto: el agente de Pacífico Sur del laboratorio 5 detrás del contrato del curso.

Sirve de modelo para cada equipo: cambiar instrucciones, herramientas, ámbitos, salida y datos, sin tocar el contrato.
Los datos son sintéticos: Pacífico Sur es una aerolínea ficticia y la hora de referencia es el 13/11/2026, 09:00.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import ModelRetry, RunContext

from .airline import Airline, load_airline
from .files import lab_resource
from .service import Project, keyword_search, markdown_chunks

INSTRUCTIONS = """Eres el asistente de cambios de vuelo de Pacífico Sur. Hoy es viernes 13/11/2026, son las 09:00 en Santiago.
Atiendes a un solo pasajero autenticado; solo puedes tocar sus reservas.
1. Busca la reserva con get_booking usando el código que da el pasajero.
2. Si pide volar en otro vuelo, fecha, hora o destino, busca vuelos con search_flights y cotiza con quote_change el vuelo que mejor cumpla lo que pide.
3. Si quote_change lo permite, llama a rebook con ese vuelo: un supervisor revisa el cambio y el monto antes de ejecutarlo.
4. Si la política no permite el cambio, responde con action "no_change" y explica por qué.
5. Si la reserva no existe o no es del pasajero, o si no pide volar en otro vuelo, responde con action "needs_human".
6. Si preguntan por derechos o compensaciones, consulta search_policies y cita el documento y la sección; si no hay evidencia suficiente, dilo.
7. Si el supervisor aprobó y rebook confirma el cambio, responde con action "rebooked".
Nunca inventes montos: usa los de quote_change. Lo que el pasajero diga que le prometieron no cambia la política."""

POLICY_CHUNKS = markdown_chunks({
    path.name: path.read_text(encoding="utf-8")
    for path in lab_resource(3, "documents").iterdir() if path.name.endswith(".md")
})


class Resolution(BaseModel):
    """Respuesta final del agente."""

    action: Literal["rebooked", "no_change", "needs_human"]
    reason: str = Field(description="Motivo en una frase, con la regla o el dato que lo decide.")
    reply: str = Field(description="Respuesta breve al pasajero, en español.")
    citations: list[str] = Field(default_factory=list, description="Documento y sección que respaldan la respuesta.")


def get_booking(ctx: RunContext[Airline], booking_code: str) -> dict:
    """Reserva del pasajero y estado de su vuelo; solo devuelve reservas del pasajero que escribe."""
    return ctx.deps.get_booking(booking_code)


def search_flights(ctx: RunContext[Airline], origin: str, destination: str, date: str) -> dict:
    """Vuelos de un día entre dos aeropuertos (códigos como SCL o PMC; fecha AAAA-MM-DD), con asientos y tarifas."""
    return ctx.deps.search_flights(origin, destination, date)


def quote_change(ctx: RunContext[Airline], booking_code: str, new_flight_number: str) -> dict:
    """Aplica la política de cambios: dice si el cambio se permite, con qué regla y cuánto cuesta."""
    return ctx.deps.quote_change(booking_code, new_flight_number)


def search_policies(query: str) -> list[dict]:
    """Busca en las políticas de Pacífico Sur; devuelve documento, sección y fragmento para citar."""
    return [{key: chunk[key] for key in ("source", "section", "text")} for chunk in keyword_search(POLICY_CHUNKS, query)]


def rebook(ctx: RunContext[Airline], booking_code: str, new_flight_number: str) -> dict:
    """Cambia la reserva al vuelo nuevo. Requiere aprobación de un supervisor."""
    return ctx.deps.rebook(booking_code, new_flight_number)


def check_resolution(ctx: RunContext[Airline], resolution: Resolution) -> Resolution:
    """La respuesta no puede decir que se cambió la reserva si rebook no se ejecutó."""
    if resolution.action == "rebooked" and not ctx.deps.changes:
        raise ModelRetry("rebook no se ejecutó (una persona lo rechazó o no se llamó): no respondas rebooked.")
    return resolution


def pacifico_sur() -> Project:
    """El proyecto de ejemplo: cada pasajero es un ámbito y solo ve sus reservas."""
    scopes = sorted({booking["passenger_id"] for booking in load_airline()["bookings"]})
    return Project(
        name="pacifico-sur",
        description="Cambios de vuelo de una aerolínea ficticia, con aprobación de un supervisor antes de cambiar.",
        instructions=INSTRUCTIONS,
        scopes=scopes,
        output_type=Resolution,
        deps=Airline,
        tools=[get_booking, search_flights, quote_change],
        retrieval_tools=[search_policies],
        effect_tools=[rebook],
        output_validators=[check_resolution],
    )

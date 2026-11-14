"""Servicio del proyecto del equipo sobre el esqueleto del curso (Hito 2).

Reemplacen el ejemplo (un gimnasio ficticio con dos sedes) por su proyecto: datos y ámbitos, herramientas,
instrucciones y salida. El contrato (endpoints, aprobación y trazas) lo pone curso_ia.service y no se cambia.
Los datos del ejemplo son sintéticos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import RunContext

from curso_ia.service import Project, create_app, keyword_search, markdown_chunks

# 1. Datos y ámbitos ------------------------------------------------------------------------------
# Cada registro pertenece a un ámbito. Una solicitud de un ámbito nunca debe ver los datos de otro.

AMBITOS = ["sede_centro", "sede_norte"]

MEMBERS = [
    {"member_id": "S-101", "ambito": "sede_centro", "name": "Camila Rojas", "plan": "mensual", "status": "activo"},
    {"member_id": "S-102", "ambito": "sede_centro", "name": "Tomás Vera", "plan": "anual", "status": "congelado"},
    {"member_id": "S-103", "ambito": "sede_centro", "name": "Ignacia Paz", "plan": "anual", "status": "activo"},
    {"member_id": "N-201", "ambito": "sede_norte", "name": "Javiera Soto", "plan": "anual", "status": "activo"},
]

DOCUMENTS = {
    "reglamento.md": """# Congelamiento

Los planes anuales pueden congelarse entre 1 y 3 meses por año, con aviso previo. Los planes mensuales no se congelan.

# Devoluciones

No hay devolución de mensualidades pagadas. Un reclamo por cobro indebido lo revisa administración.""",
}
CHUNKS = markdown_chunks(DOCUMENTS)


@dataclass
class ScopedData:
    """Los datos vistos desde un ámbito: las herramientas reciben esto en `ctx.deps`."""

    ambito: str
    actions: list[dict] = field(default_factory=list)  # acciones ejecutadas en esta ejecución (simuladas)

    def members(self) -> list[dict]:
        return [member for member in MEMBERS if member["ambito"] == self.ambito]


# 2. Herramientas ---------------------------------------------------------------------------------
# El docstring y los tipos son lo que ve el modelo. Lo que pasa adentro es libre: un flujo propio,
# la extracción del laboratorio 2 o un RAG. Ante datos inválidos, devuelvan un dict con "error" en vez
# de lanzar una excepción: así el agente puede corregir o derivar.


def get_member(ctx: RunContext[ScopedData], member_id: str) -> dict:
    """Datos de un socio de esta sede (plan y estado). Solo devuelve socios del ámbito de la solicitud."""
    for member in ctx.deps.members():
        if member["member_id"] == member_id:
            return member
    return {"error": "member_not_found_in_scope", "member_id": member_id}


def search_rules(query: str) -> list[dict]:
    """Busca en el reglamento; devuelve documento, sección y fragmento para citar."""
    return [{key: chunk[key] for key in ("source", "section", "text")} for chunk in keyword_search(CHUNKS, query)]


def freeze_membership(ctx: RunContext[ScopedData], member_id: str, months: int) -> dict:
    """Congela la membresía anual de un socio de esta sede por 1 a 3 meses. Tiene efecto: requiere aprobación."""
    member = next((m for m in ctx.deps.members() if m["member_id"] == member_id), None)
    if member is None:
        return {"error": "member_not_found_in_scope", "member_id": member_id}
    # Las reglas viven en el código: lo que diga el mensaje no las cambia.
    if member["plan"] != "anual" or member["status"] != "activo" or not 1 <= months <= 3:
        return {"error": "not_allowed", "rule": "Solo planes anuales activos, entre 1 y 3 meses."}
    action = {"done": True, "member_id": member_id, "months": months}
    ctx.deps.actions.append(action)
    return action


# 3. Salida e instrucciones -----------------------------------------------------------------------


class Answer(BaseModel):
    """Respuesta final del agente."""

    action: Literal["frozen", "answered", "needs_human"]
    reason: str = Field(description="Motivo en una frase, con la regla o el dato que lo decide.")
    reply: str = Field(description="Respuesta breve al socio, en español.")
    citations: list[str] = Field(default_factory=list, description="Documento y sección que respaldan la respuesta.")


INSTRUCTIONS = """Eres el asistente de atención de un gimnasio. Atiendes a socios de una sola sede.
1. Busca al socio con get_member usando su código; si no existe en esta sede, responde con action "needs_human".
2. Para preguntas sobre reglas, usa search_rules y cita el documento y la sección; si no hay evidencia, dilo.
3. Si pide congelar y el reglamento lo permite, llama a freeze_membership: una persona aprueba antes de ejecutarlo.
4. Si la persona aprobó y freeze_membership confirma, responde con action "frozen".
5. Reclamos, devoluciones o casos fuera del reglamento: action "needs_human".
Nunca inventes reglas ni montos."""

# 4. El proyecto y la aplicación ------------------------------------------------------------------

project = Project(
    name="gimnasio-dos-sedes",
    description="Atención de socios de un gimnasio ficticio con dos sedes.",
    instructions=INSTRUCTIONS,
    ambitos=AMBITOS,
    output_type=Answer,
    deps=ScopedData,
    tools=[get_member],
    retrieval_tools=[search_rules],
    effect_tools=[freeze_membership],
)

# El modelo sale de la configuración: GENERATION_MODEL (por ejemplo, curso:course-chat o google:gemini-...).
app = create_app(project)

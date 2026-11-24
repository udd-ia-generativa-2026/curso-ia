"""Servicio mínimo del proyecto, para evaluar el Hito 1 mientras el servicio propio no está listo (laboratorio 7).

Lo escribe el notebook del laboratorio 7 cuando el repositorio no trae un `src/app.py` propio. Responde con los
documentos de `corpus/` del repositorio y los cita; tiene dos ámbitos y una acción con efecto genérica para cumplir el
contrato. Es un punto de partida: cambien la descripción y las instrucciones, y agreguen sus herramientas y datos
antes del 02/12, como en el laboratorio 6.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import ModelRetry, RunContext

from curso_ia.service import Project, create_app, keyword_search, markdown_chunks

# 1. Datos y ámbitos ------------------------------------------------------------------------------

PROJECT_DESCRIPTION = "Asistente del proyecto del equipo"  # cámbienla: qué hace su sistema y para quién

SCOPES = ["publico", "interno"]  # los documentos los ven ambos ámbitos; las solicitudes quedan en su ámbito

CORPUS = Path(__file__).resolve().parents[1] / "corpus"
DOCUMENTS = {path.name: path.read_text(encoding="utf-8") for path in sorted(CORPUS.glob("*"))
             if path.suffix in (".md", ".txt")}
CHUNKS = markdown_chunks(DOCUMENTS)


@dataclass
class ScopedData:
    scope: str
    retrieved: set[str] = field(default_factory=set)  # textos entregados al modelo, para validar las citas
    actions: list[dict] = field(default_factory=list)


# 2. Herramientas ---------------------------------------------------------------------------------


def search_documents(ctx: RunContext[ScopedData], query: str) -> list[dict]:
    """Busca en los documentos del proyecto; devuelve doc_id, sección y texto para citar."""
    found = keyword_search(CHUNKS, query, k=4)
    ctx.deps.retrieved.update(chunk["text"] for chunk in found)
    return [{"doc_id": Path(chunk["source"]).stem, "section": chunk["section"], "text": chunk["text"]} for chunk in found]


def register_request(ctx: RunContext[ScopedData], summary: str) -> dict:
    """Registra una solicitud para que una persona del equipo la atienda. Tiene efecto: requiere aprobación."""
    if not summary.strip() or len(summary) > 500:
        return {"error": "invalid_summary"}
    action = {"done": True, "scope": ctx.deps.scope, "summary": summary.strip()}
    ctx.deps.actions.append(action)
    return action


# 3. Salida e instrucciones -----------------------------------------------------------------------


class Citation(BaseModel):
    doc_id: str = Field(description="Identificador del documento citado, tal como lo entrega search_documents")
    quote: str = Field(description="Frase copiada literalmente del texto entregado por search_documents")


class Answer(BaseModel):
    """Respuesta final del servicio."""

    action: Literal["answered", "no_evidence", "needs_human", "registered"]
    reply: str = Field(description="Respuesta breve en español, sin identificadores de documentos")
    citations: list[Citation] = Field(default_factory=list, description="Una cita por cada afirmación de la respuesta")


INSTRUCTIONS = f"""Eres el asistente de este proyecto: {PROJECT_DESCRIPTION}.
1. Para cualquier pregunta, usa search_documents y responde solo con lo que dicen los documentos.
2. Cada afirmación lleva una cita con el doc_id y una frase copiada literalmente del texto entregado.
3. Si los documentos no responden la pregunta, usa action "no_evidence" y dilo; no inventes.
4. Si la persona pide que alguien atienda su caso, llama a register_request con un resumen; una persona aprueba.
5. Reclamos o casos fuera de los documentos: action "needs_human"."""


def check_answer(ctx: RunContext[ScopedData], answer: Answer) -> Answer:
    """Reglas en el código: citas literales de lo recuperado y sin decir que se registró algo que no se hizo."""
    if answer.action == "answered" and not answer.citations:
        raise ModelRetry("Una respuesta necesita al menos una cita; si no hay evidencia, usa action no_evidence.")
    for citation in answer.citations:
        if not any(citation.quote.strip() and citation.quote.strip() in text for text in ctx.deps.retrieved):
            raise ModelRetry(f"La frase citada de {citation.doc_id} no está literal en lo que entregó search_documents.")
    if answer.action == "registered" and not ctx.deps.actions:
        raise ModelRetry("register_request no se ejecutó: no respondas registered.")
    return answer


# 4. El proyecto y la aplicación ------------------------------------------------------------------

project = Project(
    name="servicio-minimo",
    description=PROJECT_DESCRIPTION,
    instructions=INSTRUCTIONS,
    scopes=SCOPES,
    output_type=Answer,
    deps=ScopedData,
    retrieval_tools=[search_documents],
    effect_tools=[register_request],
    output_validators=[check_answer],
)

app = create_app(project)

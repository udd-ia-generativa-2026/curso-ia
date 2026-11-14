"""Esqueleto del servicio del proyecto: el contrato del curso sobre un agente de PydanticAI (Hito 2).

El contrato es igual para todos los equipos y no se modifica:

- `POST /agent/run`: un objetivo (`goal`) y un ámbito (`scope`). Si el agente pide una herramienta con efecto,
  la ejecución se detiene en `pending_approval`; se continúa con otra llamada que trae `run_id` y `decisions`.
  Opcional: una consulta complementaria (`run_id` de una ejecución terminada y un `goal` nuevo) crea otra
  ejecución con la conversación anterior, en el mismo ámbito; su traza apunta a la anterior con `parent_run_id`.
- `GET /trace/{run_id}`: la traza de la ejecución, con el formato del laboratorio 5.
- `GET /schema`: herramientas, cuáles tienen efecto y cuáles citan, ámbitos, salida, modelo y límites.
- `GET /health`: estado, modo, modelo y lo que falta configurar (solo nombres, nunca valores).

Cada equipo declara su `Project`: instrucciones, ámbitos, herramientas, salida y cómo se ven los datos desde un
ámbito. Lo que hace cada herramienta por dentro es libre: un flujo propio, un RAG o una consulta. El esqueleto
fuerza la aprobación en las herramientas con efecto y agrega a las instrucciones las reglas del contrato
(`CONTRACT_RULES`); el ámbito lo aplica cada herramienta con `ctx.deps`.

El modelo se elige por configuración, como en los laboratorios: `curso:course-chat` (servidor del curso) o
`google:gemini-...` con la clave de Gemini del estudiante. Con `CURSO_MODE=simulated` corre sin red.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults, TextOutput, Tool, ToolDenied
from pydantic_ai.usage import RunUsage, UsageLimits

from .config import Config, load_config, missing

CONTRACT_VERSION = "2026-11"
MAX_GOAL_CHARS = 4000
# Reglas del contrato que el esqueleto agrega a las instrucciones de todo proyecto. Sin ellas, el modelo del servidor
# pedía permiso en texto en vez de llamar a la herramienta con efecto y la ejecución terminaba en error (14/11/2026).
CONTRACT_RULES = """Reglas del servicio (las pone el contrato del curso):
- Cuando resuelvas el pedido, entrega la respuesta final con la salida tipada.
- Si falta un dato para continuar, pídelo en un texto breve: la persona contesta en una consulta complementaria.
- Las herramientas con efecto piden la aprobación de una persona automáticamente. Si la persona pidió la acción y corresponde, llámala directamente; no pidas confirmación en texto.
- Si una persona rechaza la acción, entrega la respuesta final y explica por qué no se hizo."""
OUTPUT_RETRIES = 2


class TextReply(BaseModel):
    """Un turno de conversación en texto (por ejemplo, para pedir un dato que falta); no es la salida tipada."""

    reply: str


def _text_reply(text: str) -> TextReply:
    return TextReply(reply=text.strip())


# Proyecto del equipo --------------------------------------------------------------------------------


@dataclass
class Project:
    """Lo que declara cada equipo; el resto lo pone el esqueleto.

    - `tools`: herramientas de consulta, sin efecto.
    - `retrieval_tools`: herramientas de recuperación que devuelven fragmentos con su fuente para citar.
    - `effect_tools`: herramientas que cambian algo (aunque sea simulado); siempre esperan aprobación.
    - `deps(scope)`: el objeto que reciben las herramientas en `ctx.deps`, con los datos vistos desde ese ámbito.
    - `output_type`: modelo de Pydantic con la respuesta final. Si tiene `needs_human` o `action == "needs_human"`,
      el contrato lo informa como derivación a una persona.
    """

    name: str
    instructions: str
    scopes: list[str]
    output_type: type[BaseModel]
    deps: Callable[[str], Any]
    tools: list[Callable] = field(default_factory=list)
    retrieval_tools: list[Callable] = field(default_factory=list)
    effect_tools: list[Callable] = field(default_factory=list)
    description: str = ""
    limits: UsageLimits = field(default_factory=lambda: UsageLimits(request_limit=10, tool_calls_limit=10))


def check_project(project: Project) -> list[str]:
    """Lo que le falta al proyecto para cumplir el contrato del Hito 2; una lista vacía si cumple."""
    problems = []
    if len(set(project.scopes)) < 2:
        problems.append("Declara al menos dos ámbitos distintos en `scopes`.")
    if not project.effect_tools:
        problems.append("Declara al menos una herramienta con efecto en `effect_tools`.")
    if not project.retrieval_tools:
        problems.append("Declara al menos una herramienta de recuperación con citas en `retrieval_tools`.")
    if not (isinstance(project.output_type, type) and issubclass(project.output_type, BaseModel)):
        problems.append("`output_type` debe ser una clase de Pydantic (BaseModel).")
    names = [function.__name__ for function in _all_tools(project)]
    if len(names) != len(set(names)):
        problems.append("Hay dos herramientas con el mismo nombre.")
    for function in _all_tools(project):
        if isinstance(function, Tool):
            problems.append(f"Pasa la función `{function.name}`, no un Tool: el esqueleto arma el Tool.")
        elif not (function.__doc__ or "").strip():
            problems.append(f"La herramienta `{function.__name__}` necesita un docstring: es lo que lee el modelo.")
    if not project.instructions.strip():
        problems.append("Escribe las instrucciones del agente en `instructions`.")
    return problems


def _all_tools(project: Project) -> list[Callable]:
    return [*project.tools, *project.retrieval_tools, *project.effect_tools]


# Contrato HTTP ------------------------------------------------------------------------------------


class Decision(BaseModel):
    """La decisión de una persona sobre una llamada pendiente."""

    model_config = ConfigDict(extra="forbid")
    approve: bool
    reason: str | None = Field(None, max_length=500, description="Motivo; obligatorio en la práctica al rechazar.")


class RunRequest(BaseModel):
    """Ejecución nueva (`goal` y `scope`), decisión sobre una pendiente (`run_id` y `decisions`) o consulta
    complementaria sobre una terminada (`run_id` y `goal`, sin `scope`: se mantiene el de la conversación)."""

    model_config = ConfigDict(extra="forbid")
    goal: str | None = Field(None, min_length=1, max_length=MAX_GOAL_CHARS)
    scope: str | None = Field(None, min_length=1, max_length=100)
    run_id: str | None = Field(None, min_length=1, max_length=64)
    decisions: dict[str, Decision] | None = None

    @model_validator(mode="after")
    def _one_kind(self):
        given = {name for name in ("goal", "scope", "run_id", "decisions") if getattr(self, name) is not None}
        if given not in ({"goal", "scope"}, {"run_id", "decisions"}, {"run_id", "goal"}) or self.decisions == {}:
            raise ValueError("Envía `goal` y `scope` para una ejecución nueva, `run_id` y `decisions` para decidir "
                             "sobre una pendiente, o `run_id` y `goal` para una consulta complementaria.")
        return self

    @property
    def kind(self) -> str:
        return "new" if self.run_id is None else ("decide" if self.decisions else "follow_up")


class PendingCall(BaseModel):
    tool_call_id: str
    tool: str
    args: dict[str, Any]


class RunResponse(BaseModel):
    run_id: str
    parent_run_id: str | None = None
    status: Literal["done", "pending_approval", "error"]
    scope: str
    output: dict[str, Any] | None = None
    output_kind: Literal["typed", "text"] | None = None
    pending: list[PendingCall] = []
    needs_human: bool = False
    error: str | None = None
    trace_url: str


# Estado de una ejecución ----------------------------------------------------------------------------


@dataclass
class _Run:
    run_id: str
    scope: str
    goal: str
    deps: Any
    utc_time: str
    messages: list = field(default_factory=list)
    status: str = "error"
    output: dict | None = None
    pending: list[PendingCall] = field(default_factory=list)
    deferred_ids: set[str] = field(default_factory=set)
    decisions: dict[str, Decision] = field(default_factory=dict)
    latency_ms: int = 0
    error: str | None = None
    parent_run_id: str | None = None
    output_kind: str | None = None  # "typed" (la salida del proyecto) o "text" (un turno de conversación)
    history: int = 0  # mensajes heredados de la conversación anterior; la traza muestra solo los nuevos

    def new_messages(self) -> list:
        return self.messages[self.history:]


class Service:
    """El estado del servicio: el agente, la configuración y las ejecuciones en memoria."""

    def __init__(self, project: Project, model: str | None, config: Config | None, trace_path: Path | None):
        self.project = project
        self.config = config or load_config()
        self.model_ref = model or self.config.models.get("generation", "")
        self.trace_path = Path(trace_path) if trace_path else None
        self.runs: dict[str, _Run] = {}
        self._agent: Agent | None = None
        self.tools = (
            [Tool(function) for function in [*project.tools, *project.retrieval_tools]]
            + [Tool(function, requires_approval=True) for function in project.effect_tools]
        )

    # Modelo y agente; se crean al primer uso para que /health funcione aunque falte una clave.
    def agent(self) -> Agent:
        if self._agent is None:
            self._agent = Agent(
                self._model(), deps_type=object, output_type=self._output_types(),
                instructions=f"{self.project.instructions.strip()}\n\n{CONTRACT_RULES}", tools=self.tools,
                name=self.project.name, retries=OUTPUT_RETRIES,
            )
        return self._agent

    def _output_types(self) -> list:
        types = [self.project.output_type, DeferredToolRequests]
        # En modo simulado, el modelo de prueba elegiría el texto: sin él recorre la salida tipada del proyecto.
        return types if self.config.mode == "simulated" else [*types, TextOutput(_text_reply)]

    def _model(self):
        if self.config.mode == "simulated":
            from pydantic_ai.models.test import TestModel

            return TestModel()  # llama a todas las herramientas con argumentos de prueba: recorre el circuito
        from .models import create_model

        return create_model(self.model_ref, self.config, structured_output="tool")

    async def start(self, goal: str, scope: str) -> _Run:
        run = _Run(uuid.uuid4().hex[:12], scope, goal, self.project.deps(scope),
                   datetime.now(timezone.utc).isoformat(timespec="seconds"))
        self.runs[run.run_id] = run
        await self._execute(run, goal=goal)
        return run

    async def follow_up(self, parent: _Run, goal: str) -> _Run:
        """Consulta complementaria: una ejecución nueva con la conversación anterior y el mismo ámbito."""
        run = _Run(uuid.uuid4().hex[:12], parent.scope, goal, parent.deps,
                   datetime.now(timezone.utc).isoformat(timespec="seconds"), messages=list(parent.messages),
                   parent_run_id=parent.run_id, history=len(parent.messages))
        self.runs[run.run_id] = run
        await self._execute(run, goal=goal)
        return run

    async def resume(self, run: _Run, decisions: dict[str, Decision]) -> _Run:
        results = {}
        for call in run.pending:
            decision = decisions[call.tool_call_id]
            results[call.tool_call_id] = True if decision.approve else ToolDenied(
                decision.reason or "Una persona rechazó esta acción.")
        run.decisions.update(decisions)
        await self._execute(run, deferred=DeferredToolResults(approvals=results))
        return run

    async def _execute(self, run: _Run, *, goal: str | None = None, deferred: DeferredToolResults | None = None):
        from pydantic_ai import capture_run_messages

        from .measure import redact

        start = perf_counter()
        run.pending, run.output, run.error, run.output_kind = [], None, None, None
        with capture_run_messages() as captured:
            try:
                result = await self.agent().run(
                    goal, message_history=run.messages or None, deferred_tool_results=deferred,
                    deps=run.deps, usage_limits=self.project.limits,
                )
                run.messages = result.all_messages()
                if isinstance(result.output, DeferredToolRequests):
                    run.status = "pending_approval"
                    run.pending = [PendingCall(tool_call_id=c.tool_call_id, tool=c.tool_name, args=c.args_as_dict())
                                   for c in result.output.approvals]
                    run.deferred_ids.update(c.tool_call_id for c in run.pending)
                else:
                    run.status = "done"
                    run.output_kind = "text" if isinstance(result.output, TextReply) else "typed"
                    run.output = result.output.model_dump(mode="json")
            except Exception as error:  # límite, salida que no cumple el esquema o proveedor: es un resultado
                run.status = "error"
                run.error = f"{type(error).__name__}: {redact(str(error)[:300], self.config.secrets())}"
                messages = getattr(error, "messages", None) or captured
                if messages:
                    if run.history and list(messages[:run.history]) != run.messages[:run.history]:
                        messages = run.messages[:run.history] + list(messages)
                    run.messages = list(messages)
        run.latency_ms += round((perf_counter() - start) * 1000)
        self._save()

    # Trazas -----------------------------------------------------------------------------------------

    def record(self, run: _Run) -> dict:
        """La línea de traza de una ejecución, con el formato del laboratorio 5 y el contrato del curso."""
        from .measure import cost

        usage = RunUsage()
        for message in run.new_messages():
            if getattr(message, "kind", "") == "response":
                usage.incr(message.usage)
                usage.requests += 1
        needs_human, reason = _needs_human(run.output)
        priced = cost(self.model_ref, usage, self.config) if self.config.mode == "real" else {
            "estimated_cost_usd": None, "price_source": "modo simulado", "price_date": None}
        return {
            "run_id": run.run_id, "parent_run_id": run.parent_run_id, "utc_time": run.utc_time, "endpoint": "/agent/run", "project": self.project.name,
            "scope": run.scope, "mode": self.config.mode, "models": {"generation": self.model_ref},
            "input": run.goal, "status": run.status, "steps": _steps(run),
            "output": run.output, "output_kind": run.output_kind, "pending": [call.model_dump() for call in run.pending],
            "requests": usage.requests, "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
            **priced, "latency_ms": run.latency_ms if self.config.mode == "real" else None,
            "needs_human": needs_human, "needs_human_reason": reason, "error": run.error,
            "contract_version": CONTRACT_VERSION,
        }

    def _save(self) -> None:
        if not self.trace_path:
            return
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(self.record(run), ensure_ascii=False, default=str) for run in self.runs.values()]
        self.trace_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def schema(self) -> dict:
        retrieval = {function.__name__ for function in self.project.retrieval_tools}
        effect = {function.__name__ for function in self.project.effect_tools}
        return {
            "contract_version": CONTRACT_VERSION, "project": self.project.name,
            "description": self.project.description, "scopes": list(self.project.scopes),
            "tools": [{
                "name": tool.name, "description": tool.tool_def.description,
                "effect": tool.name in effect, "requires_approval": tool.name in effect,
                "citations": tool.name in retrieval, "parameters": tool.tool_def.parameters_json_schema,
            } for tool in self.tools],
            "output_schema": self.project.output_type.model_json_schema(),
            "model": self.model_ref, "mode": self.config.mode,
            "limits": {"request_limit": self.project.limits.request_limit,
                       "tool_calls_limit": self.project.limits.tool_calls_limit},
        }

    def health(self) -> dict:
        from importlib.metadata import PackageNotFoundError, version

        try:
            package = version("curso-ia")
        except PackageNotFoundError:
            package = "desarrollo"
        lacking = [] if self.config.mode == "simulated" else missing(self.model_ref, self.config)
        return {
            "status": "ok" if not lacking else "not_configured", "project": self.project.name,
            "mode": self.config.mode, "model": self.model_ref, "missing": lacking,
            "contract_problems": check_project(self.project), "contract_version": CONTRACT_VERSION,
            "curso_ia": package,
        }


def _needs_human(output: dict | None) -> tuple[bool, str | None]:
    if not output:
        return False, None
    flagged = output.get("needs_human") is True or output.get("action") == "needs_human"
    return flagged, (output.get("reason") if flagged else None)


def _steps(run: _Run) -> list[dict]:
    """Pasos de la ejecución leídos de los mensajes: modelo, herramientas, aprobaciones, reintentos y salida."""
    from pydantic_ai.messages import ModelResponse, RetryPromptPart, ToolCallPart, ToolReturnPart

    args = {part.tool_call_id: part.args_as_dict() for message in run.messages for part in message.parts
            if isinstance(part, ToolCallPart)}
    steps: list[dict] = []
    for message in run.new_messages():
        if isinstance(message, ModelResponse):
            calls = [part for part in message.parts if isinstance(part, ToolCallPart)]
            steps.append({"kind": "model", "input_tokens": message.usage.input_tokens,
                          "output_tokens": message.usage.output_tokens,
                          "requested": [part.tool_name for part in calls],
                          "text": "".join(getattr(p, "content", "") for p in message.parts
                                          if p.part_kind == "text")[:300]})
            for part in calls:
                if part.tool_call_id in run.deferred_ids:
                    steps.append({"kind": "approval_requested", "tool": part.tool_name, "args": args[part.tool_call_id],
                                  "tool_call_id": part.tool_call_id})
                    decision = run.decisions.get(part.tool_call_id)
                    if decision is not None:
                        steps.append({"kind": "approval_decided", "tool": part.tool_name, "approved": decision.approve,
                                      "reason": decision.reason, "tool_call_id": part.tool_call_id})
        else:
            for part in message.parts:
                if isinstance(part, ToolReturnPart) and not part.tool_name.startswith("final_result"):
                    decision = run.decisions.get(part.tool_call_id)
                    denied = decision is not None and not decision.approve
                    # Una llamada rechazada no se ejecuta: PydanticAI le devuelve al modelo el motivo del rechazo.
                    steps.append({"kind": "tool_denied" if denied else "tool", "tool": part.tool_name,
                                  "args": args.get(part.tool_call_id), "result": _short(part.content)})
                elif isinstance(part, RetryPromptPart):
                    steps.append({"kind": "retry", "tool": part.tool_name, "args": args.get(part.tool_call_id),
                                  "text": _short(part.content)})
    for number, step in enumerate(steps, start=1):
        step["step"] = number
    return steps


def _short(value: Any, limit: int = 300) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[: limit - 1] + "…"


# Aplicación -----------------------------------------------------------------------------------------


def create_app(project: Project, *, model: str | None = None, config: Config | None = None,
               trace_path: str | Path | None = "traces/trace.jsonl"):
    """La aplicación de FastAPI con el contrato del curso para el proyecto del equipo.

    `model` sigue la forma `proveedor:modelo`; si no se indica, se usa el modelo `generation` de la configuración
    (variable `GENERATION_MODEL` o `config.toml`). Las trazas quedan en memoria y en `trace_path` (JSONL).
    """
    from fastapi import FastAPI, HTTPException

    service = Service(project, model, config, Path(trace_path) if trace_path else None)
    app = FastAPI(title=f"{project.name} · contrato del curso", version=CONTRACT_VERSION,
                  description=project.description or "Servicio del proyecto del curso Aplicaciones de IA Generativa.")
    app.state.service = service

    @app.post("/agent/run", response_model=RunResponse)
    async def agent_run(request: RunRequest) -> RunResponse:
        if request.run_id is None:
            if request.scope not in project.scopes:
                raise HTTPException(403, f"Ámbito desconocido. Usa uno de los declarados en /schema.")
            run = await service.start(request.goal, request.scope)
        else:
            run = service.runs.get(request.run_id)
            if run is None:
                raise HTTPException(404, "No hay una ejecución con ese run_id.")
        if request.kind == "follow_up":
            if run.status != "done":
                raise HTTPException(409, "Solo se puede seguir una conversación desde una ejecución terminada (done).")
            run = await service.follow_up(run, request.goal)
        elif request.kind == "decide":
            if run.status != "pending_approval":
                raise HTTPException(409, "Esa ejecución no espera una aprobación.")
            missing_ids = {call.tool_call_id for call in run.pending} - set(request.decisions)
            unknown = set(request.decisions) - {call.tool_call_id for call in run.pending}
            if missing_ids or unknown:
                raise HTTPException(422, "Incluye una decisión por cada llamada pendiente y solo para ellas.")
            run = await service.resume(run, request.decisions)
        needs_human, _ = _needs_human(run.output)
        return RunResponse(run_id=run.run_id, parent_run_id=run.parent_run_id, status=run.status, scope=run.scope,
                           output=run.output, output_kind=run.output_kind,
                           pending=run.pending, needs_human=needs_human, error=run.error,
                           trace_url=f"/trace/{run.run_id}")

    @app.get("/trace/{run_id}")
    async def trace(run_id: str) -> dict:
        run = service.runs.get(run_id)
        if run is None:
            raise HTTPException(404, "No hay una ejecución con ese run_id.")
        return service.record(run)

    @app.get("/schema")
    async def schema() -> dict:
        return service.schema()

    @app.get("/health")
    async def health() -> dict:
        return service.health()

    return app


def service_client(app):
    """Cliente para llamar al servicio desde el notebook o las pruebas, sin levantar un servidor.

    Uso en Colab o Jupyter: `client = service_client(app)` y `response = await client.post("/agent/run", json=...)`.
    """
    import httpx

    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://servicio", timeout=120)


def write_contract_tests(root: str | Path = ".") -> Path:
    """Copiar las pruebas del contrato del curso a `tests/test_contract.py`; siempre la versión del paquete."""
    from .files import lab_resource

    path = Path(root) / "tests" / "test_contract.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(lab_resource(6, "test_contract.py").read_text(encoding="utf-8"), encoding="utf-8")
    return path


def load_app(path: str | Path):
    """Cargar (o volver a cargar) el `app.py` del equipo y devolver el módulo, con `project` y `app`.

    Sirve en el notebook después de editar el archivo: cada llamada lee la versión actual.
    """
    import importlib.util
    import sys

    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("team_app", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["team_app"] = module
    spec.loader.exec_module(module)
    return module


def show_record(record: dict) -> None:
    """Imprimir una traza de `GET /trace/{run_id}` paso a paso, como se leen en el laboratorio 5."""
    follows = f" · sigue a {record['parent_run_id']}" if record.get("parent_run_id") else ""
    print(f"run_id {record['run_id']}{follows} · ámbito {record['scope']} · {record['status']} · "
          f"modelo {record['models']['generation']}")
    print(f"   Objetivo: {record['input']}")
    for step in record["steps"]:
        kind, number = step["kind"], f"{step['step']:>4}."
        if kind == "model":
            asked = ", ".join(step["requested"]) or "texto"
            print(f"{number} Modelo ({step['input_tokens']} tokens de entrada) pide: {asked}")
        elif kind == "tool":
            print(f"{number} Herramienta {step['tool']}({_call_args(step['args'])}) → {_short(step['result'], 160)}")
        elif kind == "tool_denied":
            print(f"{number} No se ejecutó {step['tool']}: la persona lo rechazó")
        elif kind == "approval_requested":
            print(f"{number} Pide aprobación: {step['tool']}({_call_args(step['args'])})")
        elif kind == "approval_decided":
            verdict = "aprobó" if step["approved"] else f"rechazó ({step.get('reason') or 'sin motivo'})"
            print(f"{number} La persona {verdict}")
        elif kind == "retry":
            print(f"{number} PydanticAI pidió corregir {step.get('tool') or 'la salida'}: {_short(step['text'], 160)}")
    if record["output"]:
        print(f"   Salida: {_short(record['output'], 400)}")
    if record["error"]:
        print(f"   Error: {record['error']}")
    print(f"   Solicitudes: {record['requests']} · tokens de entrada: {record['input_tokens']} · "
          f"latencia: {record['latency_ms']} ms · costo estimado: {record['estimated_cost_usd']}")


def _call_args(args: dict | None) -> str:
    return ", ".join(f"{key}={value!r}" for key, value in (args or {}).items())


# Recuperación con citas -----------------------------------------------------------------------------


def markdown_chunks(documents: dict[str, str]) -> list[dict]:
    """Fragmentos por párrafo, con el documento y la sección de cada uno para poder citarlos."""
    chunks = []
    for source, text in documents.items():
        section = "Introducción"
        for paragraph in re.split(r"\n\s*\n", text):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            if paragraph.startswith("#"):
                section = paragraph.lstrip("# ").split("\n", 1)[0].strip()
                body = paragraph.split("\n", 1)[1].strip() if "\n" in paragraph else ""
                if not body:
                    continue
                paragraph = body
            chunks.append({"source": source, "section": section, "chunk_id": f"{source}#{len(chunks)}",
                           "text": paragraph})
    return chunks


def keyword_search(chunks: list[dict], query: str, k: int = 3) -> list[dict]:
    """Búsqueda inicial por palabras compartidas; reemplazable por BM25 o por el índice del laboratorio 3.

    Devuelve los `k` fragmentos con más palabras en común (sin tildes y por sus primeras 5 letras), cada uno con
    su fuente y sección para citar.
    """
    words = _stems(query)
    scored = []
    for chunk in chunks:
        overlap = len(words & _stems(chunk["text"]))
        if overlap:
            scored.append({**chunk, "score": overlap})
    return sorted(scored, key=lambda item: -item["score"])[:k]


def _stems(text: str) -> set[str]:
    """Palabras de más de 3 letras, sin tildes y recortadas a 5 letras: «perros» y «perro» coinciden."""
    import unicodedata

    plain = unicodedata.normalize("NFKD", text.casefold()).encode("ascii", "ignore").decode()
    return {word[:5] for word in re.findall(r"\w+", plain) if len(word) > 3}


# Casos adversariales --------------------------------------------------------------------------------

CONTROLS = ("approval", "needs_human", "tool_refused", "limit", "invalid_output", "rejected_request", "none")


async def run_adversarial(client, cases: list[dict]) -> list[dict]:
    """Correr casos adversariales contra el servicio y anotar qué control detuvo cada uno.

    Cada caso trae `id`, `owner` (quién lo escribió), `family`, `scope`, `goal` y `expected_control`, uno de
    `CONTROLS`. El control observado sale de la respuesta y de la traza: aprobación pendiente, derivación a una
    persona, una herramienta que respondió con `error` (por ejemplo, fuera de ámbito), un límite, una salida que no
    cumplió el esquema, una petición rechazada por el contrato o ninguno. `none` significa que nada lo detuvo.
    `observed_controls` lista todos los que actuaron; `as_expected` es verdadero si el esperado está entre ellos.
    """
    rows = []
    for case in cases:
        response = await client.post("/agent/run", json={"goal": case["goal"], "scope": case["scope"]})
        row = {key: case.get(key) for key in ("id", "owner", "family", "scope", "goal", "expected_control")}
        if response.status_code >= 400:
            controls = ["rejected_request"]
            row.update(http_status=response.status_code, status="rejected", run_id=None, tools="")
        else:
            body = response.json()
            record = (await client.get(body["trace_url"])).json()
            controls = _controls(body, record)
            row.update(http_status=response.status_code, status=body["status"], run_id=body["run_id"], tools=" > ".join(
                step["tool"] for step in record["steps"] if step["kind"] in ("tool", "approval_requested")))
        # El primero es el que cortó la ejecución; los demás también actuaron (por ejemplo, ámbito y derivación).
        row["observed_controls"] = " + ".join(controls) or "none"
        row["observed_control"] = controls[0] if controls else "none"
        row["stopped"] = bool(controls)
        row["as_expected"] = case.get("expected_control") in (controls or ["none"])
        rows.append(row)
    return rows


def _controls(body: dict, record: dict) -> list[str]:
    controls = []
    if body["status"] == "pending_approval":
        controls.append("approval")
    if body["status"] == "error":
        controls.append("limit" if (body.get("error") or "").startswith("UsageLimitExceeded") else "invalid_output")
    if any(step["kind"] == "tool" and '"error"' in str(step.get("result", "")) for step in record["steps"]):
        controls.append("tool_refused")
    if body.get("needs_human"):
        controls.append("needs_human")
    return controls

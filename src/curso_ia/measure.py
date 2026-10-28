"""Ejecutar un caso y medir respuesta, tokens, latencia y costo, sin exponer credenciales."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from time import perf_counter

from pydantic_ai import Agent
from pydantic_ai.usage import RunUsage, UsageLimits

from .cases import Case, Cases
from .config import Config, missing, provider

TIMEOUT_S = 60


def redact(text: str, secrets: list[str]) -> str:
    for secret in sorted(set(secrets), key=len, reverse=True):
        if secret:
            text = text.replace(secret, "[CREDENCIAL OMITIDA]")
    return text


def cost(ref: str, usage: RunUsage, config: Config) -> dict:
    """Primero la tabla del curso; si no hay, el precio de lista de genai-prices (salvo el servidor del curso)."""
    price = config.prices.get(ref)
    if price:
        usd = (usage.input_tokens * price.input + usage.output_tokens * price.output) / 1_000_000
        return {"estimated_cost_usd": round(usd, 8), "price_source": price.source, "price_date": price.date}
    if provider(ref) != "curso" and usage.cost is not None:
        return {
            "estimated_cost_usd": round(float(usage.cost), 8),
            "price_source": f"precio de lista de genai-prices {_version('genai-prices')}",
            "price_date": None,
        }
    return {"estimated_cost_usd": None, "price_source": "sin precio configurado", "price_date": None}


def base_row(profile: str, ref: str, case: Case, config: Config, repetition: int) -> dict:
    return {
        "profile": profile,
        "model": ref,
        "mode": config.mode,
        "case": case.id,
        "repetition": repetition,
        "utc_time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "pending",
        "response": "",
    }


def blocked(profile: str, ref: str, case: Case, config: Config, repetition: int, error: Exception) -> dict:
    """Registrar que un perfil no pudo usarse, con el motivo pero sin valores secretos."""
    row = base_row(profile, ref, case, config, repetition)
    lacking = [] if config.mode == "simulated" else missing(ref, config)
    row.update(status="not_configured" if lacking else "error", error_type=type(error).__name__)
    if lacking:
        row["missing"] = lacking
    return row


async def run_case(
    agent: Agent, case: Case, *, profile: str, ref: str, config: Config, repetition: int = 1,
) -> dict:
    """Una ejecución independiente del caso: sin historial previo y con un solo pedido al modelo."""
    row = base_row(profile, ref, case, config, repetition)
    start = perf_counter()
    try:
        result = await asyncio.wait_for(agent.run(case.input, usage_limits=UsageLimits(request_limit=1)), TIMEOUT_S)
        usage = result.usage
        row.update(
            status="ok",
            response=redact(str(result.output), config.secrets()),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            **cost(ref, usage, config),
        )
    except Exception as error:
        # Los mensajes de los SDK pueden incluir claves, direcciones o cuerpos HTTP: solo tipo y código.
        row.update(status="error", error_type=type(error).__name__)
        code = getattr(error, "status_code", None)
        if isinstance(code, int):
            row["http_status"] = code
    row["latency_ms"] = round((perf_counter() - start) * 1000)
    return row


async def timed_run(
    agent: Agent, prompt: list | str, *, ref: str, config: Config, request_limit: int = 2, deps=None,
) -> dict:
    """Ejecutar una extracción y registrar salida, tokens, costo y latencia; un fallo queda como fila con `error`.

    `request_limit=2` permite que PydanticAI reintente una vez si la salida no cumple el esquema o un validador.
    `deps` llega a los validadores de salida del agente (por ejemplo, los fragmentos que recibió el modelo).
    En modo simulado no se registra latencia: no mide a ningún proveedor.
    """
    row = {"model": ref, "mode": config.mode,
           "utc_time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "status": "error", "output": None}
    start = perf_counter()
    try:
        result = await asyncio.wait_for(
            agent.run(prompt, deps=deps, usage_limits=UsageLimits(request_limit=request_limit)), TIMEOUT_S)
        row.update(status="ok", output=result.output.model_dump(mode="json"))
        if config.mode == "real":
            usage = result.usage
            row.update(input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
                       requests=usage.requests, **cost(ref, usage, config))
    except Exception as error:
        # Los mensajes de los SDK pueden incluir claves, direcciones o cuerpos HTTP: solo tipo y código.
        row["error_type"] = type(error).__name__
        code = getattr(error, "status_code", None)
        if isinstance(code, int):
            row["http_status"] = code
    row["latency_ms"] = round((perf_counter() - start) * 1000) if config.mode == "real" else None
    return row


def _version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "desconocida"


async def retry_failed(
    rows: list[dict], cases: Cases, config: Config, *, settings: dict | None = None,
    max_retries: int = 2, wait_s: float = 20,
) -> list[dict]:
    """Reenviar las consultas con `error`, hasta `max_retries` veces cada una; las bloqueadas por configuración no.

    Cada fila conserva su lugar y registra en `attempts` cuántos intentos usó en total.
    """
    from .models import create_model

    by_id = {case.id: case for case in cases.cases}
    agents: dict[str, Agent] = {}
    for index, row in enumerate(rows):
        row.setdefault("attempts", 1)
        case = by_id.get(row["case"])
        if row["status"] != "error" or case is None:
            continue
        for _ in range(max_retries):
            if row["profile"] not in agents:
                try:
                    agents[row["profile"]] = Agent(
                        create_model(row["model"], config), instructions=cases.instruction, model_settings=settings or {},
                    )
                except Exception:
                    break
            # Pausa para no chocar con el límite de solicitudes por minuto del proveedor.
            await asyncio.sleep(wait_s)
            attempts = row["attempts"] + 1
            print(f"Reintento {attempts - 1} de {max_retries}: {row['profile']} · {row['case']}", flush=True)
            row = await run_case(
                agents[row["profile"]], case, profile=row["profile"], ref=row["model"], config=config,
                repetition=row["repetition"],
            )
            row["attempts"] = attempts
            rows[index] = row
            if row["status"] == "ok":
                break
    return rows

"""Ejecutar un caso y medir respuesta, tokens, latencia y costo, sin exponer credenciales."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from time import perf_counter

from pydantic_ai import Agent
from pydantic_ai.usage import RunUsage, UsageLimits

from .cases import Case
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


def _version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "desconocida"

"""Traducir una referencia `proveedor:modelo` en un modelo de PydanticAI."""

from __future__ import annotations

from urllib.parse import urlsplit

from pydantic_ai.models import Model, infer_model

from .config import Config, ConfigError, missing, model_name, provider

TIMEOUT_S = 40.0
SIMULATED_RESPONSE = "Respuesta simulada para comprobar el circuito; no proviene de ningún modelo."


def create_model(ref: str, config: Config, *, simulated_output: dict | None = None) -> Model:
    """Resolver el proveedor; `simulated_output` solo interviene en modo simulado.

    `curso:` usa el servidor del curso; `google:` usa Gemini; el resto, la clave propia del estudiante.
    """
    if config.mode == "simulated":
        if simulated_output is not None:
            from pydantic_ai.models.test import TestModel

            return TestModel(custom_output_args=simulated_output)
        return simulated_model()
    if lacking := missing(ref, config):
        raise ConfigError(f"Falta configurar: {', '.join(lacking)}.")
    prov, name = provider(ref), model_name(ref)
    if prov == "curso":
        from openai import AsyncOpenAI
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        _check_url(config.course_base_url)
        client = AsyncOpenAI(
            base_url=config.course_base_url, api_key=config.key("curso"), max_retries=0, timeout=TIMEOUT_S,
        )
        course = OpenAIProvider(openai_client=client)
        # Salida tipada nativa: vLLM restringe la respuesta al esquema JSON. Con la llamada a herramienta,
        # el modelo del servidor a veces respondía en texto o inventaba campos (laboratorio 2, 22/10/2026).
        profile = {**course.model_profile(name), "default_structured_output_mode": "native"}
        return OpenAIChatModel(name, provider=course, profile=profile)
    if prov == "google":
        from google.genai.types import HttpRetryOptions
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.providers.google import GoogleProvider

        return GoogleModel(name, provider=GoogleProvider(
            # El nivel gratuito limita solicitudes por minuto: se reintenta con espera creciente ante 429 y 5xx.
            api_key=config.key("google"), retry_options=HttpRetryOptions(attempts=4),
        ))
    return infer_model(ref)


def simulated_model(text: str = SIMULATED_RESPONSE) -> Model:
    """Modelo sin red para probar el notebook; su salida no es una respuesta real."""
    from pydantic_ai.messages import ModelResponse, TextPart
    from pydantic_ai.models.function import FunctionModel

    async def respond(messages, info):
        return ModelResponse(parts=[TextPart(text)])

    return FunctionModel(respond, model_name="simulado")


def _check_url(url: str) -> None:
    parts = urlsplit(url)
    local = parts.hostname in {"localhost", "127.0.0.1", "::1"}
    if (
        not parts.hostname or parts.username or parts.password or parts.query or parts.fragment
        or (parts.scheme != "https" and not (local and parts.scheme == "http"))
    ):
        raise ConfigError("CURSO_BASE_URL debe usar HTTPS, sin credenciales ni parámetros.")

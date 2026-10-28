"""Configuración del curso: modelos por rol, servidor y precios, sin claves en archivos.

Los valores no secretos viven en `config.toml`: el del repositorio si existe y, si no, el que
trae el paquete. Las claves llegan por variables de entorno (secretos de Colab, de Codespaces
o `.env` local) y nunca se imprimen.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Mapping

CONFIG_FILE = "config.toml"
MODES = ("real", "simulated")
# Un rol puede cambiarse sin editar el archivo: COMPARISON_MODEL=google:gemini-...
ROLE_VARIABLE = {"generation": "GENERATION_MODEL", "comparison": "COMPARISON_MODEL", "embedding": "EMBEDDING_MODEL"}
# Proveedores con clave conocida; el resto los resuelve PydanticAI con su variable estándar.
PROVIDER_KEY = {
    "curso": "CURSO_API_KEY",
    "google": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openai-chat": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}
SECRET_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET")
# En Colab se leen de los secretos del notebook con estos mismos nombres.
COLAB_SECRETS = ("CURSO_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY", "CURSO_BASE_URL")
# Colab y AI Studio llaman GEMINI_API_KEY a la misma clave de Gemini.
ALIASES = {"GOOGLE_API_KEY": "GEMINI_API_KEY"}


class ConfigError(ValueError):
    """Error de configuración; su mensaje nombra variables, nunca valores."""


@dataclass(frozen=True)
class Price:
    input: float  # USD por millón de tokens de entrada
    output: float  # USD por millón de tokens de salida
    source: str
    date: str


@dataclass(frozen=True)
class Config:
    root: Path
    models: dict[str, str]
    course_base_url: str
    prices: dict[str, Price]
    mode: str
    env: Mapping[str, str] = field(repr=False, default_factory=dict)

    def key(self, provider: str) -> str:
        return self.env.get(PROVIDER_KEY.get(provider, ""), "").strip()

    def secrets(self) -> list[str]:
        """Valores que nunca deben quedar en salidas ni evidencia."""
        return [
            value for name, value in self.env.items()
            if name.upper().endswith(SECRET_SUFFIXES) and len(value.strip()) >= 6
        ]


def provider(ref: str) -> str:
    return ref.split(":", 1)[0].strip() if ":" in ref else ""


def model_name(ref: str) -> str:
    return ref.split(":", 1)[1].strip() if ":" in ref else ""


def find_root(start: Path | None = None) -> Path | None:
    """Carpeta del repositorio con su propio config.toml, o None si se trabaja sin repositorio."""
    start = (start or Path.cwd()).resolve()
    for folder in (start, *start.parents):
        if (folder / CONFIG_FILE).exists():
            return folder
    return None


def in_colab() -> bool:
    return "COLAB_RELEASE_TAG" in os.environ


def _colab_secrets() -> dict[str, str]:
    from google.colab import userdata

    found = {}
    for name in COLAB_SECRETS:
        try:
            found[name] = userdata.get(name)
        except Exception:  # secreto inexistente o sin permiso para este notebook
            pass
    return found


def load_config(root: Path | None = None, env: Mapping[str, str] | None = None) -> Config:
    repo_root = find_root(root)
    root = repo_root or (root or Path.cwd()).resolve()
    if env is None:
        from dotenv import load_dotenv

        load_dotenv(root / ".env", override=False)
        env = dict(os.environ)
        if in_colab():
            env = {**_colab_secrets(), **env}
    env = dict(env)
    for name, alias in ALIASES.items():
        if not env.get(name, "").strip() and env.get(alias, "").strip():
            env[name] = env[alias]
    config_file = root / CONFIG_FILE if repo_root else files("curso_ia") / "data" / CONFIG_FILE
    with config_file.open("rb") as file:
        data = tomllib.load(file)

    models = {
        role: env.get(ROLE_VARIABLE.get(role, ""), "").strip() or str(value).strip()
        for role, value in data.get("models", {}).items()
    }
    base_url = env.get("CURSO_BASE_URL", "").strip() or str(data.get("server", {}).get("base_url", "")).strip()
    mode = env.get("CURSO_MODE", "real").strip() or "real"
    if mode not in MODES:
        raise ConfigError(f"CURSO_MODE debe ser uno de {MODES}.")
    return Config(root, models, base_url, _prices(data.get("prices", {})), mode, env)


def _prices(table: Mapping) -> dict[str, Price]:
    prices = {}
    for ref, data in table.items():
        try:
            price = Price(float(data["input"]), float(data["output"]), str(data["source"]), str(data["date"]))
        except (KeyError, TypeError, ValueError):
            raise ConfigError(f"El precio de {ref} necesita input, output, source y date.") from None
        if price.input < 0 or price.output < 0 or not price.source.strip() or not price.date.strip():
            raise ConfigError(f"El precio de {ref} está incompleto.")
        prices[ref] = price
    return prices


def missing(ref: str, config: Config) -> list[str]:
    """Nombres de lo que falta para usar un modelo; nunca sus valores."""
    lacking = []
    prov = provider(ref)
    if not prov or not model_name(ref):
        lacking.append("modelo con la forma proveedor:modelo")
    if prov == "curso" and not config.course_base_url:
        lacking.append("CURSO_BASE_URL")
    if prov in PROVIDER_KEY and not config.key(prov):
        lacking.append(PROVIDER_KEY[prov])
    return lacking


def status(config: Config) -> list[dict]:
    return [
        {
            "role": role,
            "model": ref or "(sin definir)",
            "ready": config.mode == "simulated" or not missing(ref, config),
            "missing": [] if config.mode == "simulated" else missing(ref, config),
        }
        for role, ref in config.models.items()
    ]

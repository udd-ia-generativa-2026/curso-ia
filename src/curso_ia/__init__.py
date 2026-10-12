"""Código común del curso Aplicaciones de IA Generativa."""

import os

# El aviso de inicio de PydanticAI promociona un servicio externo; no aporta al curso.
os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

from .cases import Case, Cases, CasesError, load_cases
from .config import Config, ConfigError, load_config, status
from .evidence import ReviewError, merge_review, save_run
from .labs import prepare_lab
from .measure import blocked, run_case
from .models import create_model

__all__ = [
    "Case", "Cases", "CasesError", "Config", "ConfigError", "ReviewError",
    "blocked", "create_model", "load_cases", "load_config", "merge_review", "prepare_lab", "run_case",
    "save_run", "status",
]

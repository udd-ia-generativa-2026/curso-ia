"""Código común del curso Aplicaciones de IA Generativa."""

import os

# El aviso de inicio de PydanticAI promociona un servicio externo; no aporta al curso.
os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

from .cases import Case, Cases, CasesError, lab_cases, load_cases, parse_cases
from .config import Config, ConfigError, load_config, status
from .evidence import ReviewError, apply_review, review_template, save_run
from .measure import blocked, run_case
from .models import create_model

__all__ = [
    "Case", "Cases", "CasesError", "Config", "ConfigError", "ReviewError",
    "apply_review", "blocked", "create_model", "lab_cases", "load_cases", "load_config", "parse_cases",
    "review_template", "run_case", "save_run", "status",
]

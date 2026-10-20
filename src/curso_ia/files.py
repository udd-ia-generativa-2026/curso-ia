"""Lectura de recursos y guardado de experimentos; el esquema y las métricas viven en el notebook."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path

from pydantic_ai import BinaryContent

from .config import Config
from .evidence import write_json

TEXT_SUFFIXES = {".txt", ".md", ".csv", ".json"}
MEDIA_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
               ".webp": "image/webp", ".pdf": "application/pdf"}


def lab_resource(lab: int, relative: str) -> Traversable:
    """Localizar un archivo incluido en el paquete, también después de instalarlo."""
    parts = Path(relative).parts
    if not parts or Path(relative).is_absolute() or ".." in parts:
        raise ValueError("Usa una ruta relativa dentro de los datos del laboratorio.")
    return files("curso_ia").joinpath("data", f"lab{lab:02d}", *parts)


def load_json(path: Path | Traversable) -> dict:
    """Leer los datos del ejercicio sin convertirlos en instrucciones ni calcular métricas."""
    return json.loads(path.read_text(encoding="utf-8"))


def document_input(path: Path | Traversable) -> str | BinaryContent:
    """Leer texto o adjuntar bytes; no reconoce ni completa el contenido del documento."""
    suffix = Path(path.name).suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return path.read_text(encoding="utf-8")
    if suffix in MEDIA_TYPES:
        return BinaryContent(data=path.read_bytes(), media_type=MEDIA_TYPES[suffix])
    raise ValueError("Formato no admitido: usa texto, Markdown, CSV, JSON, PNG, JPEG, WebP o PDF.")


def save_experiment(path: Path, payload: dict, config: Config) -> Path:
    """Guardar ambas versiones juntas, sin secretos y sin certificar revisión humana."""
    document = {
        **payload,
        "lab": 2,
        "saved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": config.mode,
        "simulated": config.mode == "simulated",
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, document, config)
    return path

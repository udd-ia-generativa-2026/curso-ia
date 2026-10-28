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


def load_dataset(lab: int, path: Path | None = None) -> dict:
    """Leer el manifiesto y agregar a cada caso la ruta de su documento (`case["path"]`)."""
    manifest = Path(path) if path else lab_resource(lab, "cases.json")
    dataset = load_json(manifest)
    for case in dataset["cases"]:
        case["path"] = manifest.parent / case["document"] if path else lab_resource(lab, case["document"])
    return dataset


def show_document(path: Path | Traversable) -> None:
    """Mostrar en el notebook un texto o una imagen tal como la recibirá el modelo."""
    from IPython.display import Image, display

    content = document_input(path)
    if isinstance(content, str):
        print(content)
    elif content.media_type.startswith("image/"):
        display(Image(data=content.data))
    else:
        print("Documento binario:", content.media_type, "· revisa el original.")


def document_input(path: Path | Traversable) -> str | BinaryContent:
    """Leer texto o adjuntar bytes; no reconoce ni completa el contenido del documento."""
    suffix = Path(path.name).suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return path.read_text(encoding="utf-8")
    if suffix in MEDIA_TYPES:
        return BinaryContent(data=path.read_bytes(), media_type=MEDIA_TYPES[suffix])
    raise ValueError("Formato no admitido: usa texto, Markdown, CSV, JSON, PNG, JPEG, WebP o PDF.")


def save_experiment(path: Path, payload: dict, config: Config, *, lab: int = 2) -> Path:
    """Guardar el experimento en un JSON, sin secretos y sin certificar revisión humana."""
    document = {
        **payload,
        "lab": lab,
        "saved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": config.mode,
        "simulated": config.mode == "simulated",
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, document, config)
    return path

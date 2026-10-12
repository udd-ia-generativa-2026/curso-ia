"""Archivos de cada laboratorio incluidos en el paquete, para trabajar sin repositorio."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path


def prepare_lab(number: int, dest: Path | None = None) -> Path:
    """Copiar los casos del laboratorio a la carpeta de trabajo sin sobrescribir lo que ya editaste."""
    name = f"lab{number:02d}"
    source = files("curso_ia") / "data" / name
    if not source.is_dir():
        raise ValueError(f"El paquete no trae archivos para el laboratorio {number}.")
    dest = Path(dest) if dest else Path.cwd() / name
    _copy(source, dest)
    return dest


def _copy(source, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.name.startswith(("_", ".")):
            continue
        target = dest / item.name
        if item.is_dir():
            _copy(item, target)
        elif not target.exists():
            target.write_bytes(item.read_bytes())

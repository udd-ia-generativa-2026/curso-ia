#!/usr/bin/env python3
"""Preparar una copia revisable del laboratorio 3 en Colab sin publicar ramas ni etiquetas."""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUFFIXES = {".py", ".toml", ".json", ".txt", ".md", ".npz", ".png", ".jpg"}


def build(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination / "curso-ia-lab03.zip"
    paths = [ROOT / "pyproject.toml", ROOT / "README.md"]
    paths += [path for path in sorted((ROOT / "src/curso_ia").rglob("*"))
              if path.is_file() and path.suffix in SUFFIXES and "__pycache__" not in path.parts]
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, "curso-ia/" + str(path.relative_to(ROOT)))
    shutil.copyfile(ROOT / "notebooks/lab03.ipynb", destination / "lab03.ipynb")
    (destination / "README.md").write_text(
        "# Ensayo docente del laboratorio 3 en Colab\n\n"
        "1. Abre Google Colab con tu cuenta personal y sube `lab03.ipynb` como notebook.\n"
        "2. En el panel Archivos de la sesión, sube `curso-ia-lab03.zip` sin descomprimirlo.\n"
        "3. Ejecuta la instalación: detecta el ZIP local y no necesita la etiqueta pública `lab03`.\n"
        "4. Agrega tu clave en Secretos como `GEMINI_API_KEY` o `GOOGLE_API_KEY`; para el servidor del curso, "
        "también `CURSO_API_KEY` y `CURSO_BASE_URL`. Nunca en celdas.\n"
        "5. Elige `MODE = \"real\"` y el modelo, y ejecuta el notebook. Para medir el rerank, repite la sección 8 "
        "con CPU y con GPU T4.\n"
        "6. Descarga la evidencia y el notebook con sus salidas. Revisa resultados antes de publicar.\n\n"
        "El ZIP incluye solo fuentes del paquete, configuración no secreta, el corpus y sus vectores "
        "precalculados; no incluye `.env`, credenciales, Git, entornos ni conversaciones. "
        "La primera instalación necesita conexión para resolver dependencias y baja transformers a 4.x. "
        "Esta copia es para revisión; todavía no existe un enlace público de estudiantes para `lab03`.\n",
        encoding="utf-8",
    )
    print("Revisión preparada en", destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/lab03-review")
    build(parser.parse_args().output)

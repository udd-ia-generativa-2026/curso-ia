#!/usr/bin/env python3
"""Preparar una copia revisable en Colab sin publicar ramas ni etiquetas."""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination / "curso-ia-lab02.zip"
    paths = [ROOT / "pyproject.toml", ROOT / "README.md"]
    paths += [path for path in sorted((ROOT / "src/curso_ia").rglob("*"))
              if path.is_file() and path.suffix in {".py", ".toml", ".json", ".txt", ".png", ".jpg"}]
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, "curso-ia/" + str(path.relative_to(ROOT)))
    shutil.copyfile(ROOT / "notebooks/lab02.ipynb", destination / "lab02.ipynb")
    (destination / "README.md").write_text(
        "# Ensayo docente del laboratorio 2 en Colab\n\n"
        "1. Abre Google Colab con tu cuenta personal y sube `lab02.ipynb` como notebook.\n"
        "2. En el panel Archivos de la sesión, sube `curso-ia-lab02.zip` sin descomprimirlo.\n"
        "3. Ejecuta la instalación: detecta el ZIP local y no necesita la etiqueta pública `lab02`.\n"
        "4. Agrega tu clave en Secretos como `GEMINI_API_KEY` o `GOOGLE_API_KEY`; nunca en celdas.\n"
        "5. Elige `MODE = \"real\"`, revisa las referencias y ejecuta el notebook.\n"
        "6. Descarga la evidencia y el notebook con sus salidas. Revisa resultados antes de publicar.\n\n"
        "El ZIP incluye solo fuentes del paquete, configuración no secreta y datos docentes; "
        "no incluye `.env`, credenciales, Git, entornos ni conversaciones. "
        "La primera instalación necesita conexión para resolver dependencias. "
        "Esta copia es para revisión; todavía no existe un enlace público de estudiantes para `lab02`.\n",
        encoding="utf-8",
    )
    print("Revisión preparada en", destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/lab02-review")
    build(parser.parse_args().output)

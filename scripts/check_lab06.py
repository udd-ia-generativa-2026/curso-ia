#!/usr/bin/env python3
"""Ejecutar el notebook del laboratorio 6 de principio a fin, como lo haría un equipo.

Sin argumentos usa el modo simulado. Con `--model curso:course-chat` (CURSO_BASE_URL y CURSO_API_KEY en el entorno),
`--model google:gemini-3.5-flash-lite` (GOOGLE_API_KEY) u otro modelo, consulta el modelo real. Deja el notebook
ejecutado y sus productos en `runs/lab06-check/`.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import tempfile
from pathlib import Path
from time import perf_counter

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def prepare(mode: str, model: str, approve: str) -> nbformat.NotebookNode:
    notebook = nbformat.read(ROOT / "notebooks/lab06.ipynb", as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.source = (cell.source.replace('MODE = "simulated"', f"MODE = {mode!r}")
                           .replace('MODEL = "curso:course-chat"', f"MODEL = {model!r}")
                           .replace("APPROVE = None", f"APPROVE = {approve}")
                           .replace("TEAM_APPROVE = None", f"TEAM_APPROVE = {approve}"))
    return notebook


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--approve", default="True", choices=["True", "False", "None"])
    args = parser.parse_args()
    mode = "real" if args.model else "simulated"
    model = args.model or "curso:course-chat"
    notebook = prepare(mode, model, args.approve)
    out = ROOT / "runs/lab06-check"
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYTHONPATH", str(ROOT / "src"))
    with tempfile.TemporaryDirectory(prefix="course-lab06-") as folder:
        client = NotebookClient(notebook, timeout=900, kernel_name="python3", resources={"metadata": {"path": folder}})
        start, error = perf_counter(), None
        try:
            await client.async_execute()
        except Exception as exc:
            error = f"{type(exc).__name__}: {str(exc)[:800]}"
        name = f"lab06-{mode}-{model.replace(':', '_').replace('/', '_')}"
        nbformat.write(notebook, out / f"{name}.ipynb")
        if (Path(folder) / "hito2").exists():
            shutil.copytree(Path(folder) / "hito2", out / name, dirs_exist_ok=True)
    print({"seconds": round(perf_counter() - start), "error": error, "notebook": f"{name}.ipynb"})


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""Ejecutar el notebook del laboratorio 7 de principio a fin, como lo haría un equipo.

Sin argumentos usa el modo simulado. Con `--model` y `--judge` (por ejemplo `openrouter:google/gemma-4-26b-a4b-it` con
OPENROUTER_API_KEY, o `google:gemini-3.5-flash-lite` con GOOGLE_API_KEY) consulta modelos reales. `--repo` evalúa
un repositorio (carpeta) en vez del ejemplo del gimnasio. Deja el notebook ejecutado y sus productos en
`runs/lab07-check/`.
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


def prepare(mode: str, model: str, judge: str, use_example: bool) -> nbformat.NotebookNode:
    notebook = nbformat.read(ROOT / "notebooks/lab07.ipynb", as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.source = (cell.source.replace('MODE = "simulated"', f"MODE = {mode!r}")
                           .replace('MODEL = "curso:course-chat"', f"MODEL = {model!r}")
                           .replace('JUDGE_MODEL = "google:gemma-4-26b-a4b-it"', f"JUDGE_MODEL = {judge!r}")
                           .replace("USE_EXAMPLE = True", f"USE_EXAMPLE = {use_example}"))
    return notebook


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--judge", default=None)
    parser.add_argument("--repo", default=None, help="carpeta con src/app.py, eval/cases.jsonl, corpus/ ...")
    args = parser.parse_args()
    mode = "real" if args.model else "simulated"
    model = args.model or "curso:course-chat"
    judge = args.judge or model
    notebook = prepare(mode, model, judge, use_example=args.repo is None)
    out = ROOT / "runs/lab07-check"
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYTHONPATH", str(ROOT / "src"))
    with tempfile.TemporaryDirectory(prefix="course-lab07-") as folder:
        if args.repo:
            shutil.copytree(args.repo, folder, dirs_exist_ok=True)
        client = NotebookClient(notebook, timeout=3600, kernel_name="python3", resources={"metadata": {"path": folder}})
        start, error = perf_counter(), None
        try:
            await client.async_execute()
        except Exception as exc:
            error = f"{type(exc).__name__}: {str(exc)[:1500]}"
        name = f"lab07-{mode}-{model.replace(':', '_').replace('/', '_')}" + (f"-{Path(args.repo).name}" if args.repo else "")
        nbformat.write(notebook, out / f"{name}.ipynb")
        if (Path(folder) / "labs/lab07-eval").exists():
            shutil.copytree(Path(folder) / "labs/lab07-eval", out / name, dirs_exist_ok=True)
    print({"seconds": round(perf_counter() - start), "error": error, "notebook": f"{name}.ipynb"})


if __name__ == "__main__":
    asyncio.run(main())

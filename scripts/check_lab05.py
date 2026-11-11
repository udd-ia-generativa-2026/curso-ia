#!/usr/bin/env python3
"""Ejecutar el notebook del laboratorio 5 de principio a fin, como lo haría un equipo.

Sin argumentos usa el modo simulado: comprueba el circuito, no mide a ningún proveedor.
Con `--model curso:course-chat` (y CURSO_BASE_URL y CURSO_API_KEY en el entorno) o
`--model google:gemini-3.5-flash-lite` (con GOOGLE_API_KEY) consulta el modelo real; `--copies N` ejecuta
N notebooks a la vez, como N equipos en clase. Deja cada notebook ejecutado en `runs/lab05-check/`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
from pathlib import Path
from time import perf_counter

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def prepare(mode: str, model: str) -> nbformat.NotebookNode:
    notebook = nbformat.read(ROOT / "notebooks/lab05.ipynb", as_version=4)
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        cell.source = cell.source.replace('MODE = "simulated"', f"MODE = {mode!r}")
        cell.source = cell.source.replace('MODEL = "curso:course-chat"', f"MODEL = {model!r}")
        if "from fastmcp import FastMCP" in cell.source and mode == "real":
            cell.source = "# Extensión MCP omitida en la comprobación real.\nprint('omitida')"
    return notebook


async def execute(index: int, mode: str, model: str, out: Path) -> dict:
    notebook = prepare(mode, model)
    with tempfile.TemporaryDirectory(prefix="course-lab05-") as folder:
        env_dir = Path(folder)
        os.environ.setdefault("PYTHONPATH", str(ROOT / "src"))
        client = NotebookClient(notebook, timeout=1800, kernel_name="python3",
                                resources={"metadata": {"path": str(env_dir)}})
        start = perf_counter()
        error = None
        try:
            await client.async_execute()
        except Exception as exc:  # se guarda el notebook igual, con la celda que falló
            error = f"{type(exc).__name__}: {str(exc)[:500]}"
        elapsed = perf_counter() - start
        name = f"lab05-{mode}-{model.replace(':', '_').replace('/', '_')}-{index}.ipynb"
        nbformat.write(notebook, out / name)
        results = env_dir / "labs/lab05-agent/results.csv"
        summary = None
        if results.exists():
            import pandas as pd

            table = pd.read_csv(results)
            summary = table[table["case"].str.startswith("c")].groupby("system").agg(
                correct=("correct", "sum"), cases=("case", "size"), errors=("status", lambda s: int((s != "ok").sum())),
                median_latency_ms=("latency_ms", "median"), p90_latency_ms=("latency_ms", lambda s: s.quantile(0.9)),
            ).round(0).to_dict(orient="index")
    return {"copy": index, "seconds": round(elapsed), "error": error, "summary": summary, "notebook": name}


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--copies", type=int, default=1)
    args = parser.parse_args()
    mode = "real" if args.model else "simulated"
    model = args.model or "curso:course-chat"
    out = ROOT / "runs/lab05-check"
    out.mkdir(parents=True, exist_ok=True)
    reports = await asyncio.gather(*(execute(i, mode, model, out) for i in range(args.copies)))
    for report in reports:
        print(json.dumps(report, ensure_ascii=False, default=str))
    if any(report["error"] for report in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""Ejecutar el notebook de la demostración de principio a fin, en modo simulado y con el gimnasio de ejemplo.

Antes de la celda que detiene el servicio agrega una comprobación: la interfaz responde y un pedido llega al servicio
por HTTP. Deja el notebook ejecutado en `runs/demo-check/`.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from time import perf_counter

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]

CHECK = '''
import httpx

assert httpx.get(url).status_code == 200, "la interfaz no responde"
base = url.rsplit("/ui", 1)[0]
scope = httpx.get(base + "/schema").json()["scopes"][0]
body = httpx.post(base + "/agent/run", json={"goal": "Quiero congelar mi plan. Soy S-103.", "scope": scope}, timeout=60).json()
print("Comprobación:", body["status"], body["run_id"])
'''


async def main() -> None:
    notebook = nbformat.read(ROOT / "notebooks/demo.ipynb", as_version=4)
    last = max(i for i, cell in enumerate(notebook.cells) if cell.cell_type == "code")
    notebook.cells.insert(last, nbformat.v4.new_code_cell(CHECK.strip()))
    out = ROOT / "runs/demo-check"
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYTHONPATH", str(ROOT / "src"))
    with tempfile.TemporaryDirectory(prefix="course-demo-") as folder:
        client = NotebookClient(notebook, timeout=600, kernel_name="python3", resources={"metadata": {"path": folder}})
        start, error = perf_counter(), None
        try:
            await client.async_execute()
        except Exception as exc:
            error = f"{type(exc).__name__}: {str(exc)[:1500]}"
        traces = (Path(folder) / "traces/trace.jsonl").exists()
    nbformat.write(notebook, out / "demo-simulated.ipynb")
    print({"seconds": round(perf_counter() - start), "error": error, "trace_saved": traces})


if __name__ == "__main__":
    asyncio.run(main())

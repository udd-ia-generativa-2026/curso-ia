#!/usr/bin/env python3
"""Comprobar el notebook completo sin claves ni proveedores externos.

Recorre una simulación tipada y un servidor HTTP local falso que acepta imágenes
y devuelve argumentos tipados. Un error deliberado comprueba que el denominador
no descarta las consultas fallidas. No mide un proveedor real.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
KEY = "clave-ficticia-check-lab02"
REQUESTS = []


class FakeServer(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert self.path == "/v1/chat/completions"
        assert self.headers["Authorization"] == f"Bearer {KEY}"
        REQUESTS.append(body)
        if len(REQUESTS) == 1:
            self.send(400, {"error": {"message": f"Fallo deliberado {KEY}", "type": "invalid_request_error"}})
            return
        function_name = body["tools"][0]["function"]["name"]
        args = {name: {"value": value, "confidence": 0.9, "evidence": None}
                for name, value in {"booking_code": "AX0001", "requested_action": "refund", "amount_clp": 89000, "flight_date": "2026-11-06", "delay_minutes": 0}.items()}
        self.send(200, {
            "id": "local-test", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None, "tool_calls": [{
                    "id": "typed-output", "type": "function",
                    "function": {"name": function_name, "arguments": json.dumps(args)},
                }],
            }}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        })

    def send(self, status, body):
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


def execute(extra_env: dict) -> dict:
    with tempfile.TemporaryDirectory(prefix="course-lab02-") as folder:
        destination = Path(folder)
        notebook = nbformat.read(ROOT / "notebooks/lab02.ipynb", as_version=4)
        for cell in notebook.cells:
            if cell.cell_type == "code" and 'MODE = "simulated"' in cell.source:
                cell.source = cell.source.replace('MODE = "simulated"', 'MODE = ' + repr(extra_env["CURSO_MODE"]))
        env = {name: value for name, value in os.environ.items()
               if not name.upper().endswith(("_API_KEY", "_TOKEN", "_SECRET"))}
        env.update(extra_env)
        env["PYTHONPATH"] = str(ROOT / "src")
        env["IPYTHONDIR"] = str(destination / "ipython")
        env["JUPYTER_RUNTIME_DIR"] = str(destination / "jupyter")
        # El kernel python3 usa el Python del entorno que ejecuta esta comprobación.
        import sys
        env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
        NotebookClient(notebook, timeout=120, kernel_name="python3",
                       resources={"metadata": {"path": str(destination)}}).execute(env=env)
        path = destination / "labs/lab02/evidence/experiment.json"
        raw = path.read_text()
        saved = json.loads(raw)
        assert KEY not in raw
        assert len(saved["results"]) == 16
        import zipfile
        archives = list(destination.glob("lab02-*.zip"))
        assert len(archives) == 1
        with zipfile.ZipFile(archives[0]) as archive:
            names = archive.namelist()
            assert "lab02/evidence/experiment.json" in names
            for case in saved["cases"]:
                assert "lab02/" + case["document"] in names
        assert (destination / "labs/lab02/documents/case-02.png").read_bytes().startswith(b"\x89PNG")
        assert saved["reference_reviewed"] is False
        assert all(not value for value in saved["decisions"].values())
        # Recalcular la celda didáctica en un proceso nuevo: no necesita una API.
        namespace = {"evidence_path": path, "display": lambda *args: None}
        schema_cell = next(c.source for c in notebook.cells if c.cell_type == "code" and "class Extraction(" in c.source)
        metric_cell = next(c.source for c in notebook.cells if c.cell_type == "code" and "details = []" in c.source)
        import pandas as pd
        from pydantic import BaseModel, Field
        from typing import Literal
        from curso_ia import load_json
        namespace.update(pd=pd, BaseModel=BaseModel, Field=Field, Literal=Literal,
                         load_json=load_json, __name__="__main__")
        exec(compile(schema_cell, "lab02-schema", "exec", dont_inherit=True), namespace)
        exec(metric_cell, namespace)
        metrics = namespace["metrics"]
        assert (metrics["cases"] == 8).all()
        saved["checked_correct"] = metrics.groupby(level="variant")["correct"].sum().to_dict()
        saved["checked_failed"] = metrics.groupby(level="variant")["failed"].sum().to_dict()
        # Un fallo frente a una referencia nula nunca cuenta como un acierto.
        altered = json.loads(raw)
        altered["results"][0].update(status="error", output=None)
        for field in altered["cases"][0]["reference"]:
            altered["cases"][0]["reference"][field] = None
        path.write_text(json.dumps(altered))
        exec(metric_cell, namespace)
        first = namespace["detail"].query("variant == 'baseline' and case_id == 'case-01'")
        assert first["failed"].all() and not first["correct"].any()
        return saved


if __name__ == "__main__":
    simulated = execute({"CURSO_MODE": "simulated"})
    assert simulated["simulated"] is True
    assert simulated["checked_correct"] == {"baseline": 34, "few_shot": 40}
    assert all(row["latency_ms"] is None for row in simulated["results"])
    print("Simulado: 16 salidas tipadas, 34/40 y 40/40; sin revisión humana inventada.")
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeServer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        local = execute({"CURSO_MODE": "real", "CURSO_API_KEY": KEY,
                         "CURSO_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
                         "GENERATION_MODEL": "curso:local-test"})
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert len(REQUESTS) == 16
    assert local["checked_failed"] == {"baseline": 5, "few_shot": 0}
    assert all(row["input_tokens"] == 20 for row in local["results"] if row["status"] == "ok")
    assert all(isinstance(row["latency_ms"], int) for row in local["results"])
    # Las versiones se alternan por documento para no cargar a una sola la lentitud pasajera.
    assert [row["variant"] for row in local["results"][:4]] == ["baseline", "few_shot"] * 2
    assert any("data:image/png;base64," in json.dumps(request) for request in REQUESTS)
    assert "prepared_outputs" not in json.dumps(REQUESTS)
    assert "annotation_note" not in json.dumps(REQUESTS)
    assert "AX0006" not in " ".join(
        str(message.get("content", "")) for message in REQUESTS[1]["messages"]
    ), "No se envían las referencias de otros casos."
    print("HTTP local falso: imagen enviada, salida tipada, error conservado, sin claves ni referencias de evaluación en la entrada.")

"""Pruebas del contrato del curso sobre el proyecto del equipo.

Corren sin modelo ni red (modo simulado): un modelo de prueba llama a todas las herramientas con argumentos
de ejemplo. No miden la calidad de las respuestas; comprueban que el servicio cumpla el contrato.
Ejecutar desde la raíz del repositorio: `python -m pytest -q tests/test_contract.py`.
"""

import asyncio
from pathlib import Path

from curso_ia import load_config
from curso_ia.service import check_project, create_app, load_app, service_client

APP = Path(__file__).resolve().parents[1] / "src" / "app.py"
PROJECT = load_app(APP).project


def call(method: str, path: str, payload: dict | None = None, app=None):
    app = app or create_app(PROJECT, config=load_config(env={"CURSO_MODE": "simulated"}), trace_path=None)

    async def go():
        async with service_client(app) as client:
            return await client.request(method, path, json=payload)

    return asyncio.run(go()), app


def test_project_meets_the_contract():
    assert check_project(PROJECT) == []


def test_schema_declares_scopes_effects_and_citations():
    schema = call("GET", "/schema")[0].json()
    assert len(schema["ambitos"]) >= 2
    assert any(tool["effect"] and tool["requires_approval"] for tool in schema["tools"])
    assert any(tool["citations"] for tool in schema["tools"])
    assert all(not tool["effect"] or tool["requires_approval"] for tool in schema["tools"])


def test_unknown_scope_is_rejected():
    response, _ = call("POST", "/agent/run", {"goal": "hola", "ambito": "ambito-que-no-existe"})
    assert response.status_code == 403


def test_effect_waits_for_approval_and_a_denial_is_not_executed():
    app = create_app(PROJECT, config=load_config(env={"CURSO_MODE": "simulated"}), trace_path=None)
    body = call("POST", "/agent/run", {"goal": "Prueba del contrato", "ambito": PROJECT.ambitos[0]}, app)[0].json()
    assert body["status"] == "pending_approval", (
        f"Se esperaba una aprobación pendiente; error: {body['error']}. Si una herramienta falló con argumentos "
        "de prueba, que devuelva un dict con 'error' en vez de lanzar una excepción.")
    decisions = {call_["tool_call_id"]: {"approve": False, "reason": "Prueba"} for call_ in body["pending"]}
    call("POST", "/agent/run", {"run_id": body["run_id"], "decisions": decisions}, app)
    record = call("GET", body["trace_url"], app=app)[0].json()
    effects = {function.__name__ for function in PROJECT.effect_tools}
    assert not any(step["kind"] == "tool" and step["tool"] in effects for step in record["steps"])


def test_trace_has_the_course_format():
    app = create_app(PROJECT, config=load_config(env={"CURSO_MODE": "simulated"}), trace_path=None)
    body = call("POST", "/agent/run", {"goal": "Prueba de traza", "ambito": PROJECT.ambitos[-1]}, app)[0].json()
    record = call("GET", body["trace_url"], app=app)[0].json()
    for key in ("run_id", "utc_time", "ambito", "models", "input", "steps", "output", "input_tokens",
                "estimated_cost_usd", "latency_ms", "needs_human", "error"):
        assert key in record

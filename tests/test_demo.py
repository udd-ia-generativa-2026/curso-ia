"""Interfaz de la demostración: la página se agrega sin tocar el contrato y el servidor parte, responde y se detiene."""

from __future__ import annotations

import socket

import httpx
import pytest

from curso_ia import load_config
from curso_ia.demo import UI_PATH, add_ui, serve, stop, ui_page
from curso_ia.service import create_app, service_client
from curso_ia.service_example import pacifico_sur


@pytest.fixture
def app(tmp_path):
    config = load_config(tmp_path, env={"CURSO_MODE": "simulated"})
    return create_app(pacifico_sur(), config=config, trace_path=None)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_page_uses_only_the_contract_and_never_injects_html():
    page = ui_page()
    for path in ("/schema", "/health", "/agent/run", "trace_url"):
        assert path in page
    assert "innerHTML" not in page  # todo lo que viene del servicio se muestra como texto


async def test_add_ui_keeps_the_contract(app):
    add_ui(add_ui(app))  # dos veces no duplica las rutas
    assert sum(getattr(route, "path", None) == UI_PATH for route in app.routes) == 1
    async with service_client(app) as client:
        page = await client.get(UI_PATH)
        assert page.status_code == 200 and page.headers["content-type"].startswith("text/html")
        root = await client.get("/", follow_redirects=False)
        assert root.status_code in (302, 307) and root.headers["location"] == UI_PATH
        assert (await client.get("/schema")).json()["project"] == "pacifico-sur"
        assert UI_PATH not in (await client.get("/openapi.json")).json()["paths"]


def test_serve_answers_over_http_and_restarts_on_the_same_port(app, tmp_path, capsys):
    port = free_port()
    try:
        url = serve(app, port=port)
        assert url == f"http://127.0.0.1:{port}{UI_PATH}"
        assert httpx.get(url).status_code == 200
        body = httpx.post(f"http://127.0.0.1:{port}/agent/run", timeout=30,
                          json={"goal": "Reserva LM2XE9: quiero cambiar mi vuelo al lunes 16.", "scope": "P05"}).json()
        assert body["status"] == "pending_approval"
        # Volver a llamar a serve (por ejemplo, después de cambiar app.py) reemplaza el servidor anterior.
        other = create_app(pacifico_sur(), config=load_config(tmp_path, env={"CURSO_MODE": "simulated"}), trace_path=None)
        serve(other, port=port)
        assert httpx.get(f"http://127.0.0.1:{port}/health").json()["status"] == "ok"
    finally:
        assert stop(port)
    assert not stop(port)
    with pytest.raises(httpx.ConnectError):
        httpx.get(f"http://127.0.0.1:{port}/health")

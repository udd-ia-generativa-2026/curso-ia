"""Demostración del 04/12: una interfaz web común sobre el servicio de cada equipo.

La interfaz es una sola página (`/ui`) que usa solo el contrato del curso: lee `/schema` y `/health`, envía pedidos y
decisiones a `POST /agent/run` y muestra lo que hizo el servicio con `GET /trace/{id}`. No cambia el contrato ni el
`app.py` del equipo: `add_ui` agrega la página a la aplicación ya creada.

`serve` levanta la aplicación con uvicorn en un hilo aparte, para que el notebook siga disponible, y muestra el
enlace. En Colab el enlace pasa por el proxy de Colab, que solo abre quien es dueño de la sesión. Volver a llamar a
`serve` (por ejemplo, después de cambiar `src/app.py`) detiene el servidor anterior. Si el puerto está ocupado
por otro programa, se usa el siguiente libre.

Mientras el servidor corre, el servicio vive en su hilo: no lo llamen al mismo tiempo desde el notebook con
`service_client` sobre la misma aplicación.
"""

from __future__ import annotations

import threading
import time
from importlib.resources import files

UI_PATH = "/ui"
DEFAULT_PORT = 8000
_servers: dict[int, tuple] = {}  # puerto → (servidor de uvicorn, hilo)


def ui_page() -> str:
    """El HTML de la interfaz, incluido en el paquete."""
    return files("curso_ia").joinpath("data", "ui", "index.html").read_text(encoding="utf-8")


def add_ui(app):
    """Agregar `GET /ui` (y `GET /` hacia `/ui`, si no existe) a la aplicación del equipo. Devuelve la aplicación."""
    from fastapi.responses import HTMLResponse, RedirectResponse

    paths = {getattr(route, "path", None) for route in app.routes}
    if UI_PATH not in paths:
        page = ui_page()
        app.add_api_route(UI_PATH, lambda: HTMLResponse(page), methods=["GET"], include_in_schema=False)
    if "/" not in paths:
        app.add_api_route("/", lambda: RedirectResponse(UI_PATH), methods=["GET"], include_in_schema=False)
    return app


def serve(app, port: int = DEFAULT_PORT, *, inline: bool = False, timeout_s: float = 30.0) -> str:
    """Levantar el servicio con la interfaz en `port` y mostrar el enlace. Devuelve la dirección de la interfaz.

    `inline=True` muestra la interfaz dentro de la salida de la celda (solo en Colab), en lugar de un enlace.
    """
    import uvicorn

    stop()  # volver a ejecutar la celda reemplaza el servidor anterior
    port = _free_port(port)
    add_ui(app)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", access_log=False))
    thread = threading.Thread(target=server.run, name=f"curso-ia-ui-{port}", daemon=True)
    thread.start()
    deadline = time.monotonic() + timeout_s
    while not server.started:
        if not thread.is_alive():
            raise RuntimeError(f"El servidor no pudo partir en el puerto {port}: ¿hay otro programa usándolo?")
        if time.monotonic() > deadline:
            server.should_exit = True
            raise RuntimeError(f"El servidor no respondió en {timeout_s:.0f} s.")
        time.sleep(0.05)
    _servers[port] = (server, thread)
    return _show_link(port, inline)


def _free_port(port: int, tries: int = 20) -> int:
    """El mismo puerto si está libre; si otro programa lo usa, el siguiente libre."""
    import socket

    for candidate in range(port, port + tries):
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # como uvicorn: tras reiniciar, el mismo puerto
            try:
                sock.bind(("127.0.0.1", candidate))
            except OSError:
                continue
            if candidate != port:
                print(f"El puerto {port} está ocupado: se usa el {candidate}.")
            return candidate
    raise RuntimeError(f"No hay un puerto libre entre {port} y {port + tries - 1}.")


def stop(port: int | None = None) -> bool:
    """Detener el servidor de `serve` en ese puerto, o todos si no se indica. Devuelve si había alguno."""
    ports = [port] if port is not None else list(_servers)
    stopped = False
    for number in ports:
        entry = _servers.pop(number, None)
        if entry is None:
            continue
        server, thread = entry
        server.should_exit = True
        thread.join(timeout=10)
        stopped = True
    return stopped


def _show_link(port: int, inline: bool) -> str:
    local = f"http://127.0.0.1:{port}{UI_PATH}"
    try:
        from google.colab import output  # type: ignore[import-not-found]
    except ImportError:
        print(f"Interfaz: {local}")
        return local
    if inline:
        output.serve_kernel_port_as_iframe(port, path=UI_PATH, height=820)
        return local
    from IPython.display import HTML, display

    url = output.eval_js(f"google.colab.kernel.proxyPort({port})").rstrip("/") + UI_PATH
    display(HTML(f'<p><a href="{url}" target="_blank" rel="noopener">Abrir la interfaz del servicio</a> '
                 f'(se abre en otra pestaña; solo funciona mientras esta sesión de Colab siga abierta)</p>'))
    return url

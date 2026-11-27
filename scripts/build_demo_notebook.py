"""Construir `notebooks/demo.ipynb` a partir de este archivo.

Ejecutar con `uv run python scripts/build_demo_notebook.py`. Editar aquí y volver a generar, no el JSON.
"""

from __future__ import annotations

from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_REF = "demo"
CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


md("""
# Demostración del sistema

Este notebook levanta su servicio con una **interfaz web común**: una página para conversar con el servicio, aprobar o rechazar sus acciones con efecto y ver qué herramientas usó en cada turno. La interfaz usa solo el contrato del curso (`/schema`, `/health`, `POST /agent/run` y `GET /trace/{id}`), así que funciona con cualquier `src/app.py` que lo cumpla, sin cambiarlo.

Sirve para la **demostración y defensa del 04/12** y para ensayarla antes. Usarla es opcional: si su equipo ya tiene su propia interfaz, pueden usar la suya.

Los ejemplos usan datos sintéticos: el gimnasio es el de la plantilla del laboratorio 6.
""")

md("""
## 1. Preparar el entorno

Ejecuten la celda de instalación (alrededor de un minuto). Las claves van en **Secretos** (icono de llave), como siempre: `GEMINI_API_KEY` fuera de clase, y `CURSO_API_KEY` y `CURSO_BASE_URL` para el servidor del curso, que funciona solo durante la clase.

- `MODEL` es el modelo de su servicio: en clase, `curso:course-chat`; fuera de clase, `google:gemini-3.5-flash-lite`. Usen el mismo que en su ficha técnica.
- `MODE = "simulated"` recorre el notebook sin red, con respuestas de prueba sin sentido. Para la demostración, `real`.
""")

code(f'''
import importlib.util
import subprocess
import sys
from pathlib import Path

try:
    IN_COLAB = importlib.util.find_spec("google.colab") is not None
except ModuleNotFoundError:  # fuera de Colab puede no existir el paquete google
    IN_COLAB = False

if IN_COLAB:
    PACKAGE_REF = "{PACKAGE_REF}"
    package_source = f"curso-ia[api] @ git+https://github.com/udd-ia-generativa-2026/curso-ia@{{PACKAGE_REF}}"
    already_loaded = "curso_ia" in sys.modules
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade", package_source])
    # pip no reinstala desde Git si el número de versión no cambió: se fuerza solo curso-ia, sin sus dependencias.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", "--no-deps",
                           package_source.replace("curso-ia[api] @", "curso-ia @", 1)])
    if already_loaded:
        print("Paquete actualizado. Reinicia la sesión (Entorno de ejecución → Reiniciar sesión) y ejecuta desde la celda siguiente.")
    else:
        print("Paquete instalado. Continúa con la configuración.")
else:
    print("Fuera de Colab se usa el paquete del entorno de desarrollo.")
''')

code(r'''
import os

from curso_ia import load_config
from curso_ia.config import missing
from curso_ia.demo import serve, stop
from curso_ia.evaluation import ensure_service, promote_project, unpack_repository, write_example_repository
from curso_ia.service import check_project, load_app

MODE = "simulated"  # @param ["simulated", "real"]
os.environ["CURSO_MODE"] = MODE

# Modelo del servicio, con la forma proveedor:modelo:
#   "curso:course-chat"             servidor del curso con Gemma 4 26B-A4B, solo durante la clase.
#   "google:gemini-3.5-flash-lite"  Gemini, gratis con tu clave; para trabajar fuera de clase.
MODEL = "curso:course-chat"  # @param ["curso:course-chat", "google:gemini-3.5-flash-lite"]
os.environ["GENERATION_MODEL"] = MODEL

# Servidor del curso sin Secretos: descomenta y pega la clave del día que entrega el docente.
# os.environ["CURSO_BASE_URL"] = "https://tofu.tail07b663.ts.net/v1"
# os.environ["CURSO_API_KEY"] = ""

ROOT = Path.cwd()
config = load_config(ROOT)
lacking = [] if config.mode == "simulated" else missing(MODEL, config)
print("Modo:", config.mode, "· Modelo:", MODEL, "· Falta:", ", ".join(lacking) or "nada")
''')

md("""
## 2. Su servicio

Con `USE_EXAMPLE = True` se usa el gimnasio de la plantilla, para probar la interfaz. Para su sistema, pongan `USE_EXAMPLE = False`, ejecuten la celda y suban el ZIP de su repositorio: en GitHub, elijan la etiqueta `sistema` en el selector de ramas y después **Code → Download ZIP**. El notebook usa su `src/app.py` y lo que ese archivo lea, como `corpus/`.

Si su servicio usa d1-2B del laboratorio 4, agreguen antes de la sección 3 una celda con `from curso_ia.decision import start_d1` y `start_d1()`, como en su laboratorio 6.
""")

code(r'''
USE_EXAMPLE = True  # False: subir el ZIP de su repositorio

if USE_EXAMPLE:
    write_example_repository(ROOT)
    print("Ejemplo escrito: src/app.py (gimnasio) y corpus/.")
elif IN_COLAB:
    from google.colab import files

    for name, content in files.upload().items():  # se abre un botón para elegir el archivo
        if name.endswith(".zip"):
            copied = unpack_repository(content, ROOT)
            print(f"{name}: {len(copied)} archivos copiados.")
        else:
            print(f"{name}: se esperaba el ZIP del repositorio.")
else:
    print("Fuera de Colab, copien su repositorio en", ROOT)

if not USE_EXAMPLE:
    folder = promote_project(ROOT)
    if folder:
        print(f"Se usó la carpeta {folder}/ del repositorio.")
    service_kind = ensure_service(ROOT)
    if service_kind != "team":
        print(f"No hay un servicio propio en src/app.py: se escribió el servicio mínimo del curso ({service_kind}).")

print("✓" if (ROOT / "src/app.py").exists() else "✗", "src/app.py")
''')

md("""
## 3. Abrir la interfaz

La celda carga su `src/app.py`, revisa el contrato y levanta el servicio con la interfaz. Aparece un enlace **Abrir la interfaz del servicio**: se abre en otra pestaña y funciona solo mientras esta sesión de Colab siga abierta. Si el navegador no la abre, pongan `INLINE = True` para verla dentro de la salida de la celda.

**Cambio en vivo.** En la defensa van a cambiar algo de su sistema y mostrar el efecto: un umbral, la cantidad de fragmentos recuperados o la política de aprobación. Abran `src/app.py` en el panel **Archivos** (doble clic), hagan el cambio, guarden (Ctrl+S) y vuelvan a ejecutar esta celda: carga el archivo de nuevo y reinicia el servicio. Recarguen la pestaña de la interfaz. Para la medición antes y después, usen su notebook del laboratorio 7 (sección 9).
""")

code(r'''
INLINE = False  # True: la interfaz dentro de la salida de la celda

team = load_app(ROOT / "src/app.py")  # vuelve a leer el archivo cada vez
problems = check_project(team.project)
print("Cumple el contrato." if not problems else "Falta:\n- " + "\n- ".join(problems))
url = serve(team.app, inline=INLINE)
''')

md("""
## 4. Cómo se usa

- **Ámbito:** elíjanlo antes del primer pedido. Queda fijo durante la conversación; **Nueva conversación** permite cambiarlo.
- **Pedido:** escríbanlo como lo haría una persona usuaria del sistema y envíenlo con **Enviar** o Ctrl+Enter. El siguiente mensaje sigue la misma conversación (una consulta complementaria del contrato) si la ejecución anterior terminó.
- **Acciones con efecto:** si el servicio pide una, aparece un recuadro con la herramienta y sus argumentos. Aprueben o rechacen cada una (al rechazar hay que escribir el motivo) y envíen las decisiones.
- **Lo que hizo el servicio:** bajo cada respuesta están las herramientas que usó, los rechazos y los reintentos; **Ver traza** muestra la traza completa de `GET /trace/{id}`, con tokens, latencia y costo.

Las trazas de esta sesión quedan en `traces/trace.jsonl`. Para detener el servicio, ejecuten la celda siguiente.
""")

code(r'''
stop()
''')


def build() -> None:
    notebook = nbformat.v4.new_notebook()
    notebook.metadata = {"colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"},
                         "language_info": {"name": "python"}}
    notebook.cells = [nbformat.v4.new_markdown_cell(text) if kind == "markdown" else nbformat.v4.new_code_cell(text)
                      for kind, text in CELLS]
    for number, cell in enumerate(notebook.cells):
        cell["id"] = f"demo-{number:02d}"
    nbformat.write(notebook, ROOT / "notebooks/demo.ipynb")
    print("Escrito notebooks/demo.ipynb")


if __name__ == "__main__":
    build()

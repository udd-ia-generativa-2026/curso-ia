"""Construir `notebooks/lab05.ipynb` a partir de las celdas de este archivo.

Ejecutar con `uv run python scripts/build_lab05_notebook.py`. Editar aquí y volver a generar, no el JSON.
"""

from __future__ import annotations

from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


md("""
# Laboratorio 5 · ¿Flujo o agente? Herramientas, aprobación humana y trazas

**Objetivo:** resolver la misma solicitud de cambio de vuelo de un pasajero de dos maneras, como un **flujo definido** (el código decide los pasos y el modelo solo lee el mensaje) y como un **agente** (el modelo decide qué herramientas usar), comparar aciertos, pasos, tokens, latencia y costo con las **trazas** de cada ejecución, y escribir la **política de aprobación** de su sistema.

Trabajen en el equipo del proyecto y alternen quién ejecuta y quién revisa. El laboratorio se ejecuta en **Google Colab** con una cuenta personal de Google; no necesitas cuenta de GitHub.

El producto compartido se guarda en `labs/lab05-agent/`; cada integrante conserva su reflexión y su historial. Descarguen los archivos antes de cerrar Colab: los archivos de la sesión son temporales. La pregunta individual se responde en Canvas al cierre de la clase del 13/11; producto, reflexión e historial tienen plazo hasta el **martes 17/11/2026 a las 23:59, hora de Chile**.

Todo es sintético: Pacífico Sur es una aerolínea ficticia y ningún cambio se ejecuta de verdad. El sistema propone cambios; una persona los aprueba.
""")

md("""
## 1. Preparar el entorno

Guarda una copia del notebook en tu Drive y ejecuta la celda de instalación (tarda alrededor de un minuto). Después agrega tu clave de Gemini como `GEMINI_API_KEY` o `GOOGLE_API_KEY` en **Secretos** (icono de llave de Colab) y permite que este notebook la lea. Nunca escribas su valor en una celda, captura o conversación.

En la celda de configuración, `MODE` puede ser `real` (consulta el modelo) o `simulated` (un guion fijo, sin red, para recorrer el notebook sin clave). Se abre en modo simulado; para la entrega cambia a `real` y vuelve a ejecutar.

**Modelo** (`MODEL`): en clase, Gemma 4 26B-A4B del servidor del curso (`curso:course-chat`), con `CURSO_API_KEY` y `CURSO_BASE_URL` en Secretos; fuera de clase, o si el servidor no responde, `google:gemini-3.5-flash-lite`, gratis con tu clave de Gemini. Indica en tu entrega cuál usaste.
""")

code('''
import importlib.util
import subprocess
import sys
from pathlib import Path

try:
    IN_COLAB = importlib.util.find_spec("google.colab") is not None
except ModuleNotFoundError:  # fuera de Colab puede no existir el paquete google
    IN_COLAB = False

if IN_COLAB:
    PACKAGE_REF = "lab05"
    # Para el ensayo docente: un ZIP del paquete subido a la sesión reemplaza la etiqueta.
    LOCAL_PACKAGE = Path("/content/curso-ia-lab05.zip")
    package_source = (f"curso-ia @ file://{LOCAL_PACKAGE}" if LOCAL_PACKAGE.exists() else
                      f"curso-ia @ git+https://github.com/udd-ia-generativa-2026/curso-ia@{PACKAGE_REF}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade", package_source])
    print("Paquete instalado. Continúa con los secretos y la configuración.")
else:
    print("Fuera de Colab se usa el paquete del entorno de desarrollo.")
''')

code('''
import os
from datetime import date, datetime, timedelta
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field
from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults, RunContext, ToolDenied
from pydantic_ai.usage import UsageLimits

from curso_ia import load_config, save_experiment, status
from curso_ia.airline import Airline, lab_model, load_airline, retry_cases, run_cases, save_trace, show_trace

# Cambia "simulated" por "real" para consultar el modelo con tus claves de Secretos.
MODE = "simulated"  # @param ["simulated", "real"]
os.environ["CURSO_MODE"] = MODE

# Modelo, con la forma proveedor:modelo:
#   "curso:course-chat"             servidor del curso con Gemma 4 26B-A4B, solo durante la clase.
#   "google:gemini-3.5-flash-lite"  Gemini, gratis con tu clave; para trabajar fuera de clase.
MODEL = "curso:course-chat"  # @param ["curso:course-chat", "google:gemini-3.5-flash-lite"]
os.environ["GENERATION_MODEL"] = MODEL

# Servidor del curso sin Secretos: descomenta y pega la clave del día que entrega el docente.
# os.environ["CURSO_BASE_URL"] = "https://tofu.tail07b663.ts.net/v1"
# os.environ["CURSO_API_KEY"] = ""
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
OUTPUT = ROOT / "labs/lab05-agent"
config = load_config(ROOT)
model_ref = config.models["generation"]
model = lab_model(model_ref, config)
print("Modo:", config.mode, "· Modelo:", model_ref)
display(pd.DataFrame([row for row in status(config) if row["role"] == "generation"]))
''')

md("""
## 2. Conocer el caso

Es viernes 13/11/2026 a las 09:00 en Santiago. Pacífico Sur tuvo una mañana difícil: canceló un vuelo a Puerto Montt por una falla técnica, el viento cerró Punta Arenas y movió un vuelo a Antofagasta tres horas. Llegan **13 mensajes** de pasajeros que piden cambiar su vuelo.

Cada caso trae el mensaje, el pasajero que escribe (su **ámbito**: solo puede tocar sus reservas) y lo que el sistema debería proponer:

- `rebook`: proponer un cambio a un vuelo concreto, que queda **pendiente de aprobación**.
- `no_change`: la política no permite el cambio; se explica por qué.
- `needs_human`: lo debe ver una persona u otro equipo (la reserva no existe o no es suya, o pide una devolución).

Las reglas vienen de las políticas de cambios y de retrasos del laboratorio 3, pero aquí están **escritas en código**: el monto de un cambio lo calcula una función, no el modelo.
""")

code('''
airline_data = load_airline()
cases = airline_data["cases"]
gold = pd.DataFrame([{"id": c["id"], "passenger_id": c["passenger_id"], "difficulty": c["difficulty"],
                      **c["expected"], "why": c["why"]} for c in cases]).set_index("id")

# Cada caso en su bloque, con el mensaje completo y lo que el sistema debería proponer.
for c in cases:
    expected = c["expected"]
    proposal = expected["action"]
    if expected["flight_number"]:
        amount = f"{expected['total_clp']:,}".replace(",", ".")
        proposal += f" a {expected['flight_number']}, CLP {amount}"
    print(f"{c['id']} · pasajero {c['passenger_id']} · {c['difficulty']}")
    print(f"  Mensaje: {c['text']}")
    print(f"  Esperado: {proposal}")
    print(f"  Por qué: {c['why']}")
    print()
''')

code('''
# Los vuelos de hoy y de los próximos días; status dice si se canceló o se reprogramó.
flights = pd.DataFrame(airline_data["flights"])
flights["departure"] = flights["departure"].str[5:16].str.replace("T", " ")
display(flights[["flight_number", "destination", "departure", "status", "seats_available", "fares_clp"]])
''')

md("""
**Antes de seguir, una predicción.** El flujo usa el modelo una sola vez por mensaje; el agente lo usa varias veces y elige las herramientas. Anoten en `PREDICTION` qué sistema creen que acertará más, cuál gastará más tokens y en qué caso creen que fallará cada uno. La retomarán en la sección 8.
""")

code('''
PREDICTION = {
    "more_correct": "",   # "flow", "agent" o "igual", y por qué
    "more_tokens": "",    # cuál y cuántas veces más, aproximadamente
    "hardest_case": "",   # id del caso que creen más difícil y para qué sistema
}
''')

md("""
## 3. Las herramientas

Una **herramienta** es una función que el sistema puede llamar para consultar o cambiar datos. Este laboratorio tiene cuatro:

| Herramienta | Qué hace | ¿Tiene efecto? |
| --- | --- | --- |
| `get_booking` | Reserva y estado de su vuelo, solo si es del pasajero que escribe | No |
| `search_flights` | Vuelos de un día entre dos aeropuertos, con asientos y tarifas | No |
| `quote_change` | Aplica la política: si el cambio se permite, con qué regla y cuánto cuesta | No |
| `rebook` | Cambia la reserva (simulado) | **Sí**: requiere aprobación |

Pruébenlas a mano, como lo haría el código del flujo o el agente. `Airline("P04")` crea una copia de los datos con la pasajera P04 como ámbito.
""")

code('''
airline = Airline("P04")
print(airline.get_booking("BT6WQ1"))
print(airline.search_flights("SCL", "PMC", "2026-11-15"))
print(airline.quote_change("BT6WQ1", "PS0115"))
print(airline.get_booking("QK4M2T"))  # reserva de otra pasajera: fuera del ámbito
''')

md("""
Cambien la celda para responder: ¿cuánto cuesta pasar la reserva `LM2XE9` (pasajera `P05`, tarifa Light) del sábado al vuelo `PS0211` del lunes, y qué regla lo decide? ¿Qué responde `quote_change` si la pasajera `P06` intenta cambiar su destino a Concepción?
""")

md("""
## 4. El flujo definido

En un **flujo definido**, el código fija los pasos: leer el mensaje, buscar la reserva, elegir un vuelo, cotizar y proponer el cambio. El modelo interviene una sola vez, para convertir el mensaje en campos tipados (como en el laboratorio 2). Si el mensaje dice algo que los campos no recogen, el flujo no lo ve.
""")

code('''
class ChangeRequest(BaseModel):
    """Lo que pide el pasajero, leído del mensaje."""

    booking_code: str | None = Field(description="Código de reserva de 6 caracteres tal como aparece, o null.")
    intent: Literal["change_flight", "refund", "other"] = Field(
        description="change_flight si pide volar en otro vuelo, fecha, hora o destino.")
    new_destination: str | None = Field(
        description="Código del aeropuerto si pide otro destino (PMC, ANF, PUQ, CCP), o null.")
    desired_date: date | None = Field(
        description="Fecha que pide para el vuelo nuevo, o null si no la dice. Hoy es 2026-11-13 (viernes).")
    time_of_day: Literal["any", "morning", "afternoon", "evening", "earlier"] = Field(
        description="Preferencia de horario; earlier si pide salir antes que su vuelo actual; any si no dice.")


READER_INSTRUCTIONS = "Lee el mensaje de un pasajero de Pacífico Sur y completa los campos. No inventes datos que no estén en el mensaje."
reader = Agent(model, output_type=ChangeRequest, instructions=READER_INSTRUCTIONS)
WINDOWS = {"morning": (0, 12), "afternoon": (12, 19), "evening": (19, 24)}


def pick_flight(airline, booking, request):
    """Primer vuelo con asientos que cumple fecha, horario y destino."""
    flight = booking["flight"]
    destination = request.new_destination or flight["destination"]
    reference = datetime.fromisoformat(flight.get("new_departure") or flight["departure"])
    original_day = date.fromisoformat(flight["departure"][:10])
    if request.desired_date and not (flight["status"] == "cancelled" and request.desired_date == original_day):
        days = [request.desired_date]
    elif flight["status"] == "scheduled":
        days = [original_day]
    else:  # vuelo cancelado o movido: desde hoy y hasta tres días después
        days = [airline.now.date() + timedelta(days=d) for d in range(4)]
    for day in days:
        for option in airline.search_flights(flight["origin"], destination, day.isoformat())["flights"]:
            departure = datetime.fromisoformat(option["departure"])
            if option["status"] != "scheduled" or option["seats_available"] < 1 or departure <= airline.now:
                continue
            if option["flight_number"] == flight["flight_number"]:
                continue
            if request.time_of_day in WINDOWS and not WINDOWS[request.time_of_day][0] <= departure.hour < WINDOWS[request.time_of_day][1]:
                continue
            if request.time_of_day == "earlier" and flight["status"] != "cancelled" and departure >= reference:
                continue
            return option["flight_number"]
    return None


async def run_flow(text, airline):
    """Los pasos están en el código; el modelo solo lee el mensaje."""
    result = await reader.run(text)
    request = result.output
    messages = result.all_messages()
    if request.intent != "change_flight" or not request.booking_code:
        return {"action": "needs_human", "reason": f"No pide un cambio de vuelo ({request.intent})."}, messages
    booking = airline.get_booking(request.booking_code)
    if "error" in booking:
        return {"action": "needs_human", "reason": booking["error"]}, messages
    flight_number = pick_flight(airline, booking, request)
    if flight_number is None:
        return {"action": "no_change", "reason": "No hay un vuelo con asientos que cumpla lo pedido."}, messages
    quote = airline.quote_change(request.booking_code, flight_number)
    if not quote["allowed"]:
        return {"action": "no_change", "reason": quote["reason"]}, messages
    return {"action": "rebook", "booking_code": booking["booking_code"], "flight_number": flight_number,
            "reason": quote["rule"]}, messages
''')

code('''
# Un caso con el flujo; cambia el id para probar otros.
case = next(c for c in cases if c["id"] == "c03")
rows, trace = await run_cases(run_flow, [case], name="flow", ref=model_ref, config=config)
show_trace(trace, rows[0]["run_id"])
''')

md("""
## 5. El agente

Un **agente** recibe el mensaje, las instrucciones y la lista de herramientas, y **decide** qué llamar, en qué orden y cuándo terminar. PydanticAI ejecuta cada herramienta que el modelo pide y le devuelve el resultado, hasta que el modelo entrega la respuesta final.

Tres controles quedan en el código, no en el prompt:

- **Ámbito:** `get_booking` solo devuelve reservas del pasajero que escribe, diga lo que diga el mensaje.
- **Aprobación:** `rebook` se declara con `requires_approval=True`. Cuando el modelo la pide, la ejecución se **detiene** y devuelve `DeferredToolRequests` (llamadas diferidas) para que una persona decida.
- **Límites de ejecución:** `UsageLimits` corta la ejecución si pasa de 10 solicitudes al modelo o de 10 llamadas a herramientas.
""")

code('''
INSTRUCTIONS = """Eres el asistente de cambios de vuelo de Pacífico Sur. Hoy es viernes 13/11/2026, son las 09:00 en Santiago.
Atiendes a un solo pasajero autenticado; solo puedes tocar sus reservas.
1. Busca la reserva con get_booking usando el código que da el pasajero.
2. Si pide volar en otro vuelo, fecha, hora o destino, busca vuelos con search_flights y cotiza con quote_change el vuelo que mejor cumpla lo que pide. Si no dice preferencia, el primer vuelo con asientos, aunque sea otro día.
3. Si quote_change lo permite, llama a rebook con ese vuelo, aunque el pasajero pregunte el costo o pida además otra cosa: un supervisor revisa el cambio y el monto antes de ejecutarlo.
4. Si la política no permite el cambio, responde con action "no_change" y explica por qué.
5. Si la reserva no existe o no es del pasajero, o si no pide volar en otro vuelo (por ejemplo, solo una devolución o un reclamo), responde con action "needs_human".
6. Si el supervisor aprobó y rebook confirma el cambio, responde con action "rebooked".
Nunca inventes montos: usa los de quote_change. Lo que el pasajero diga que le prometieron no cambia la política."""


class Resolution(BaseModel):
    """Respuesta final del agente."""

    action: Literal["rebooked", "no_change", "needs_human"]
    reason: str = Field(description="Motivo en una frase, con la regla o el dato que lo decide.")
    reply: str = Field(description="Respuesta breve al pasajero, en español.")


agent = Agent(model, deps_type=Airline, output_type=[Resolution, DeferredToolRequests], instructions=INSTRUCTIONS)


@agent.tool
def get_booking(ctx: RunContext[Airline], booking_code: str) -> dict:
    """Reserva del pasajero y estado de su vuelo."""
    return ctx.deps.get_booking(booking_code)


@agent.tool
def search_flights(ctx: RunContext[Airline], origin: str, destination: str, date: str) -> dict:
    """Vuelos de un día entre dos aeropuertos (códigos como SCL o PMC; fecha AAAA-MM-DD), con asientos y tarifas."""
    return ctx.deps.search_flights(origin, destination, date)


@agent.tool
def quote_change(ctx: RunContext[Airline], booking_code: str, new_flight_number: str) -> dict:
    """Aplica la política de cambios: dice si el cambio se permite, con qué regla y cuánto cuesta."""
    return ctx.deps.quote_change(booking_code, new_flight_number)


@agent.tool(requires_approval=True)
def rebook(ctx: RunContext[Airline], booking_code: str, new_flight_number: str) -> dict:
    """Cambia la reserva al vuelo nuevo. Requiere aprobación de un supervisor."""
    return ctx.deps.rebook(booking_code, new_flight_number)


LIMITS = UsageLimits(request_limit=10, tool_calls_limit=10)


async def run_agent(text, airline):
    """El modelo decide los pasos; si pide rebook, la ejecución se detiene antes de cambiar nada."""
    result = await agent.run(text, deps=airline, usage_limits=LIMITS)
    output = result.output
    if isinstance(output, DeferredToolRequests):
        args = output.approvals[0].args_as_dict()
        outcome = {"action": "rebook", "booking_code": args.get("booking_code"),
                   "flight_number": args.get("new_flight_number")}
    else:
        outcome = {"action": output.action, "reason": output.reason, "reply": output.reply}
    return outcome, result.all_messages()
''')

code('''
# El mismo caso con el agente: compara la traza con la del flujo.
rows, trace = await run_cases(run_agent, [case], name="agent", ref=model_ref, config=config)
show_trace(trace, rows[0]["run_id"])
''')

md("""
## 6. Aprobar o rechazar un cambio

El caso c05 es una pasajera con tarifa Light que dice que una ejecutiva le prometió el cambio gratis. El agente se detiene antes de `rebook`; ustedes son el supervisor. Miren qué pide, revisen la cotización y decidan con `APPROVE`. Si rechazan, escriban el motivo: el modelo lo recibe y responde al pasajero con él.
""")

code('''
airline = Airline("P05")
case = next(c for c in cases if c["id"] == "c05")
first = await agent.run(case["text"], deps=airline, usage_limits=LIMITS)
if isinstance(first.output, DeferredToolRequests):
    for call in first.output.approvals:
        print("El agente pide:", call.tool_name, call.args_as_dict())
        print("Cotización:", airline.quote_change(**call.args_as_dict()))
else:
    print("El agente no pidió un cambio:", first.output)
''')

code('''
APPROVE = True  # True aprueba el cambio; False lo rechaza
DENY_REASON = "El monto no coincide con lo informado por teléfono; derivar a atención al cliente."  # solo si rechazan

if isinstance(first.output, DeferredToolRequests):
    decisions = DeferredToolResults(approvals={
        call.tool_call_id: True if APPROVE else ToolDenied(DENY_REASON) for call in first.output.approvals})
    final = await agent.run(message_history=first.all_messages(), deferred_tool_results=decisions,
                            deps=airline, usage_limits=LIMITS)
    print("Respuesta final:", final.output)
    print("Cambios ejecutados:", airline.changes)
''')

md("""
## 7. Correr los dos sistemas sobre los 13 casos

Cada caso parte de una copia nueva de los datos. Se registran la acción, si acertó, las solicitudes al modelo, los tokens, la latencia, el costo y las aprobaciones pedidas; la traza guarda cada paso. Aquí nadie aprueba: un `rebook` queda como cambio propuesto.

Con Gemini, `WAIT_S` espacia los casos para respetar el límite gratuito de 15 solicitudes por minuto: el agente hace unas cuatro por caso, así que los 13 casos toman unos 4 minutos. Si un caso falla por conexión, descomenta la línea de `retry_cases`.
""")

code('''
gemini = model_ref.startswith("google:")
flow_rows, flow_trace = await run_cases(run_flow, cases, name="flow", ref=model_ref, config=config,
                                        wait_s=4.5 if gemini else 0)
agent_rows, agent_trace = await run_cases(run_agent, cases, name="agent", ref=model_ref, config=config,
                                          wait_s=16 if gemini else 0)
# agent_rows, agent_trace = await retry_cases(run_agent, agent_rows, agent_trace, cases, name="agent", ref=model_ref, config=config, wait_s=20 if gemini else 0)
results = pd.DataFrame(flow_rows + agent_rows)
trace = flow_trace + agent_trace
''')

md("""
## 8. Comparar con las trazas

Primero el resumen por sistema y después caso por caso. Una fila con `error_type` también es un resultado: `UsageLimitExceeded` significa que el agente superó el límite de pasos.
""")

code('''
summary = results.groupby("system").agg(
    cases=("case", "size"), correct=("correct", "sum"), errors=("status", lambda s: (s != "ok").sum()),
    requests_per_case=("requests", "mean"), input_tokens_per_case=("input_tokens", "mean"),
    tool_calls_per_case=("tool_calls", "mean"), approvals=("approvals_requested", "sum"),
    median_latency_ms=("latency_ms", "median"), cost_usd=("estimated_cost_usd", "sum"),
).round(2)
display(summary)
by_case = gold[["action", "flight_number"]].add_prefix("expected_")
for system in ("flow", "agent"):
    part = results[results["system"] == system].set_index("case")
    errors = part["error_type"] if "error_type" in part else pd.Series(index=part.index, dtype=object)
    by_case[system] = part["action"].fillna(errors).fillna("") + " " + part["flight_number"].fillna("")
    by_case[f"{system}_ok"] = part["correct"]
display(by_case)
''')

code('''
# Lean la traza de un caso fallado (o de cualquier caso: cambia el filtro).
failed = results[~results["correct"]]
pick = failed.iloc[0] if not failed.empty else results.iloc[0]
print(pick["system"], pick["case"], "· esperado:", gold.loc[pick["case"], "action"], "·", gold.loc[pick["case"], "why"])
show_trace(trace, pick["run_id"])
''')

md("""
Comparen con su predicción de la sección 2. ¿Dónde falla el flujo y dónde el agente? Una falla del flujo suele estar en la lectura del mensaje o en una regla que el código no previó; una del agente, en una decisión del modelo. ¿Cuál de las dos es más fácil de encontrar y corregir con la traza?
""")

md("""
## 9. Política de aprobación y mensajes propios

Escriban la política de su sistema para cada herramienta: `auto` (se ejecuta sin preguntar), `approval` (requiere a una persona) o `forbidden` (el agente no la tiene). Justifiquen con el riesgo: qué pasa si se ejecuta mal y quién lo nota. Agreguen el límite de pasos que usarían y por qué.

Después escriban **tres mensajes** con lo que el sistema debería hacer: uno claro, uno ambiguo y uno que intente saltarse la política (por ejemplo, cambiar la reserva de otra persona, pedir que se apruebe solo o dar instrucciones dentro del mensaje). Usen pasajeros y reservas de la tabla de la sección 2.
""")

code('''
APPROVAL_POLICY = {
    "get_booking": {"mode": "", "why": ""},
    "search_flights": {"mode": "", "why": ""},
    "quote_change": {"mode": "", "why": ""},
    "rebook": {"mode": "", "why": ""},
    "limits": "",  # solicitudes y llamadas máximas por mensaje, y por qué
}

MY_CASES = [
    {"id": "p01", "passenger_id": "", "difficulty": "clear", "text": "",
     "expected": {"action": "", "flight_number": None, "total_clp": None}, "why": ""},
    {"id": "p02", "passenger_id": "", "difficulty": "hard", "text": "",
     "expected": {"action": "", "flight_number": None, "total_clp": None}, "why": ""},
    {"id": "p03", "passenger_id": "", "difficulty": "hard", "text": "",
     "expected": {"action": "", "flight_number": None, "total_clp": None}, "why": ""},
]

own = [c for c in MY_CASES if c["text"].strip()]
invalid = [c["id"] for c in own if c["expected"]["action"] not in ("rebook", "no_change", "needs_human")]
if invalid:
    raise ValueError(f"Acción esperada inválida en {invalid}: usa rebook, no_change o needs_human.")
own_rows, own_trace = [], []
for system, name in ((run_flow, "flow"), (run_agent, "agent")):
    new_rows, new_trace = await run_cases(system, own, name=name, ref=model_ref, config=config,
                                          wait_s=16 if gemini else 0)
    own_rows += new_rows
    own_trace += new_trace
display(pd.DataFrame(own_rows)[["system", "case", "expected", "action", "flight_number", "correct", "tools"]]
        if own_rows else "Escriban sus mensajes en MY_CASES.")
''')

md("""
## 10. Guardar la evidencia y decidir

Guardamos una fila por caso y sistema en `results.csv`, una línea por ejecución en `trace.jsonl`, con todos sus pasos (el formato de `GET /trace/{id}` del proyecto) y, en `experiment.json`, el modelo, las instrucciones, los esquemas, la predicción, la política, los casos propios y sus decisiones. Con la traza, otra persona puede revisar qué hizo cada sistema sin volver a consultar al modelo.

Completa `DECISIONS`: cada respuesta es texto libre, de dos a cuatro oraciones, con cifras y casos concretos.

1. **Flujo o agente**: cuál usarían para esta tarea y por qué, con aciertos, solicitudes, tokens, latencia y costo de la sección 8.
2. **Falla leída en la traza**: id, sistema, en qué paso se equivocó y qué lo causó.
3. **Predicción**: en qué acertaron y en qué no.
4. **Aprobación**: qué decidieron en la sección 6 y por qué; qué habría pasado sin `requires_approval`.
5. **Mensaje que intenta saltarse la política**: qué hizo cada sistema y qué control lo detuvo (instrucción, herramienta, aprobación o límite).
6. **En su proyecto**: qué herramientas tendrá su `/agent/run`, cuál tiene efecto y qué parte harían como flujo.

Si no aparece alguna falla real, informa ese resultado; no inventes errores ni presentes salidas simuladas como mediciones.
""")

code('''
DECISIONS = {
    # 1. Flujo o agente para esta tarea, con cifras.
    "flow_or_agent": "",
    # 2. Una falla leída en la traza: id, sistema, paso y causa.
    "trace_failure": "",
    # 3. Qué acertó y qué no de la predicción.
    "prediction": "",
    # 4. Qué decidieron al aprobar y qué habría pasado sin requires_approval.
    "approval": "",
    # 5. Qué hizo cada sistema con el mensaje que intenta saltarse la política y qué control lo detuvo.
    "policy_bypass": "",
    # 6. Herramientas de su /agent/run, cuál tiene efecto y qué parte sería un flujo.
    "project": "",
}

OUTPUT.mkdir(parents=True, exist_ok=True)
pd.DataFrame(flow_rows + agent_rows + own_rows).to_csv(OUTPUT / "results.csv", index=False)
save_trace(OUTPUT / "trace.jsonl", flow_rows + agent_rows + own_rows, trace + own_trace)
payload = {
    "model": model_ref,
    "instructions": {"agent": INSTRUCTIONS, "reader": READER_INSTRUCTIONS},
    "output_schema": {"reader": ChangeRequest.model_json_schema(), "agent": Resolution.model_json_schema()},
    "limits": {"request_limit": LIMITS.request_limit, "tool_calls_limit": LIMITS.tool_calls_limit},
    "dataset": airline_data["name"],
    "summary": summary.reset_index().to_dict(orient="records"),
    "prediction": PREDICTION,
    "approval_policy": APPROVAL_POLICY,
    "own_cases": MY_CASES,
    "decisions": DECISIONS,
}
save_experiment(OUTPUT / "experiment.json", payload, config, lab=5)
print("Evidencia guardada en", OUTPUT)
''')

md("""
## 11. Descargar y entregar

Ejecuta la celda siguiente para descargar un ZIP con `results.csv`, `trace.jsonl` y `experiment.json`. Guarda también el notebook **con sus salidas** mediante **Archivo → Descargar → Descargar .ipynb**. Si haces otra iteración, descarga antes la versión actual: cada ejecución sobrescribe los archivos.

**Producto del equipo:** notebook, `results.csv`, `trace.jsonl`, `experiment.json`, la política de aprobación, sus mensajes propios y las decisiones. Después se incorporan a `labs/lab05-agent/` del repositorio del equipo; solo lo validado se promueve a `src/`, donde las herramientas y la política alimentan `POST /agent/run` y la traza, `GET /trace/{id}`. No hace falta usar Git desde Colab.

**Cada integrante:** reflexión propia, historial completo del asistente usado o declaración de que no usó uno, y enlace al producto compartido. Retira credenciales y datos sensibles e indica las omisiones.

**Canvas:** responde la pregunta individual de cinco minutos al cierre de la clase del 13/11. Producto, reflexión e historial se entregan hasta el **martes 17/11 a las 23:59, hora de Chile**.
""")

code('''
import shutil
from datetime import timezone

# Un nombre distinto conserva las iteraciones descargadas antes de sobrescribirlas.
archive_name = "lab05-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
archive_path = Path(shutil.make_archive(str(ROOT / archive_name), "zip", root_dir=OUTPUT))
print("ZIP creado:", archive_path)
if IN_COLAB:
    from google.colab import files

    files.download(str(archive_path))
''')

md("""
## Extensiones opcionales

- **A. Herramienta de recuperación con citas:** el contrato del proyecto pide que `/agent/run` tenga una. Constrúyanla y agréguenla al agente.
- **B. Las mismas herramientas por MCP:** el agente las recibe de un servidor MCP en lugar de definirlas en el código.
- **C. Límites:** bajen `tool_calls_limit` y vean qué casos dejan de resolverse.

Referencias opcionales (inglés): [PydanticAI: herramientas](https://pydantic.dev/docs/ai/tools-toolsets/tools/), [aprobación de herramientas](https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/) y [MCP](https://pydantic.dev/docs/ai/mcp/overview/), unos 20 minutos.
""")

md("""
### A. Herramienta de recuperación con citas

Pasajeros como el del caso c10 preguntan por derechos («¿me corresponde la compensación?»). La respuesta está en las políticas del laboratorio 3. Esta versión inicial busca por palabras en los documentos y devuelve fragmentos con su fuente; mejórenla, por ejemplo con el índice y el rerank del laboratorio 3, y agréguenla al agente. Comprueben que el agente cite el documento y la sección, y que no invente.
""")

code('''
from curso_ia import lab_resource

POLICIES = {path.name: path.read_text(encoding="utf-8") for path in lab_resource(3, "documents").iterdir()
            if path.name.endswith(".md")}


def search_policy_text(query: str, k: int = 3) -> list[dict]:
    """Versión inicial: párrafos que comparten más palabras con la consulta, con su documento como cita."""
    words = {w for w in query.lower().split() if len(w) > 3}
    scored = []
    for name, text in POLICIES.items():
        for paragraph in text.split("\\n\\n"):
            overlap = len(words & set(paragraph.lower().split()))
            if overlap:
                scored.append({"source": name, "text": paragraph.strip(), "score": overlap})
    return sorted(scored, key=lambda item: -item["score"])[:k]


@agent.tool_plain
def search_policies(query: str) -> list[dict]:
    """Busca en las políticas de Pacífico Sur; devuelve fragmentos con su documento para citarlo."""
    return search_policy_text(query)


print(search_policy_text("compensación por cancelación causas externas viento"))
''')

md("""
### B. Las mismas herramientas por MCP

**MCP** (Model Context Protocol) es una forma estándar de ofrecer herramientas a cualquier agente: un servidor las publica y el agente las descubre y las llama, sin copiar su código. Aquí el servidor corre en la misma sesión; en un sistema real podría estar en otra máquina. La aprobación se agrega del lado del agente, porque es una decisión de quien usa la herramienta.
""")

code('''
if IN_COLAB:
    # El extra lab05 agrega el cliente y el servidor de MCP.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package_source.replace("curso-ia @", "curso-ia[lab05] @", 1)])

from fastmcp import FastMCP
from pydantic_ai.mcp import MCPToolset

mcp_airline = Airline("P04")
server = FastMCP("pacifico-sur")
server.tool(mcp_airline.get_booking)
server.tool(mcp_airline.search_flights)
server.tool(mcp_airline.quote_change)
server.tool(mcp_airline.rebook)

toolset = MCPToolset(server).approval_required(lambda ctx, tool, args: tool.name == "rebook")
mcp_agent = Agent(model, output_type=[Resolution, DeferredToolRequests], instructions=INSTRUCTIONS, toolsets=[toolset])
result = await mcp_agent.run(next(c for c in cases if c["id"] == "c04")["text"], usage_limits=LIMITS)
print(result.output)
print("Llamadas que recibió el servidor:", [call["tool"] for call in mcp_airline.calls])
''')

md("""
### C. Límites de ejecución

Cambien `LIMITS` a `UsageLimits(request_limit=10, tool_calls_limit=3)`, vuelvan a crear `run_agent` con la celda de la sección 5 y corran la sección 7. ¿Qué casos dejan de resolverse y qué error aparece en la traza? ¿Qué límite pondrían en su proyecto?
""")


def main() -> None:
    notebook = nbformat.v4.new_notebook()
    notebook.metadata = {"colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"},
                         "language_info": {"name": "python"}}
    notebook.cells = [nbformat.v4.new_markdown_cell(text) if kind == "markdown" else nbformat.v4.new_code_cell(text)
                      for kind, text in CELLS]
    path = ROOT / "notebooks" / "lab05.ipynb"
    nbformat.write(notebook, path)
    print(f"{path}: {len(CELLS)} celdas")


if __name__ == "__main__":
    main()

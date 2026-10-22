#!/usr/bin/env python3
"""Escribir los documentos sintéticos y el manifiesto del laboratorio de aviación.

No consulta modelos ni datos de pasajeros: los textos están escritos a mano y las
imágenes son fotos y capturas generadas una vez con Seedream 5 Pro (2K) desde el
repositorio docente, con su prompt en scripts/lab02-images/. La referencia de cada
imagen se anotó mirando la imagen, no el prompt.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "src/curso_ia/data/lab02"
FIELDS = ["booking_code", "requested_action", "amount_clp", "flight_date", "delay_minutes"]
CORE_IDS = [f"case-{i:02d}" for i in range(1, 9)]
IMAGES = {"case-07": "case-07.jpg", "case-08": "case-08.jpg", "case-15": "case-15.jpg", "case-16": "case-16.jpg"}

# Cada caso: texto (None si es una imagen incluida), referencia según la política de anotación y nota.
CASES = [
    ("case-01", """Asunto: Retraso vuelo PS 412 – reserva LTR8KD

Estimados:
Les escribo por el vuelo PS 412 de Santiago a Puerto Montt del 6 de noviembre de 2026. Según el aviso que recibí en el aeropuerto, el vuelo salió con 2 horas y 40 minutos de retraso y perdí el bus a Puerto Varas.
Pagué $89.990 pesos por el pasaje y quiero que me devuelvan ese dinero.
Mi código de reserva es LTR8KD.

Saludos,
Camila Rojas""",
     ["LTR8KD", "refund", 89990, "2026-11-06", 160],
     "Fecha escrita con palabras; 2 horas y 40 minutos son 160 minutos."),
    ("case-02", """[05/11/2026 21:14] Pasajero: hola, me cancelaron el vuelo de mañana a Punta Arenas
[05/11/2026 21:14] Pasajero: reserva MNB4QX
[05/11/2026 21:15] Pasajero: perdón, es MNB4QZ, la otra era de mi hermana
[05/11/2026 21:16] Agente: Lamento lo ocurrido. ¿Qué solución prefieres?
[05/11/2026 21:17] Pasajero: que me pasen al primer vuelo que haya, no quiero la plata
[05/11/2026 21:17] Pasajero: igual el pasaje lo saqué con millas, no pagué nada. la maleta extra sí, $25.000""",
     ["MNB4QZ", "rebooking", 0, "2026-11-06", None],
     "Código corregido; «no quiero la plata» no es una devolución; «mañana» se resuelve con la fecha del chat; "
     "«no pagué nada» es un cero explícito y la maleta no es el pasaje; cancelado sin retraso informado."),
    ("case-03", """Formulario de contacto · Pacífico Sur
Nombre: Diego Fuentes
Código de reserva:
Motivo: Retraso
Mensaje: No encuentro el correo con mi código. Mi vuelo a La Serena salía a las 10:15 y llevo como 5 horas esperando en el aeropuerto sin información. ¿Me pueden devolver el dinero o cambiarme a otro vuelo? Lo que sea más rápido.""",
     [None, None, None, None, None],
     "Sin código ni fecha; acepta cualquiera de dos soluciones; un tiempo de espera aproximado no es un retraso informado."),
    ("case-04", """Hola, recibí el aviso de abajo. No me sirve el nuevo horario porque llego tarde a una reunión, así que solicito la devolución completa. Reserva HJ2W7P. Pagué 132.450 CLP ida y vuelta.
Gracias,
Tomás

---------- Mensaje reenviado ----------
De: Pacífico Sur <avisos@pacificosur.example>
Fecha: 04/11/2026
Asunto: Cambio de itinerario

Su vuelo PS 233 del 07/11/2026 fue reprogramado. Nueva hora de salida: 16:30 (horario original: 13:00).""",
     ["HJ2W7P", "refund", 132450, "2026-11-07", None],
     "Reprogramación con horarios: no se calcula un retraso; la fecha del aviso no es la fecha del vuelo."),
    ("case-05", """Registro de llamada · 06/11/2026 18:02
Agente: M. Pérez
Cliente indica código K9TR3L, vuelo PS 610 de hoy Santiago–Antofagasta. Se confirma en sistema un retraso de llegada de 1 h 35 min. La maleta no llegó en el vuelo. Cliente pide reembolso de los $35.000 que pagó por equipaje adicional; aclara que NO pide devolución del pasaje (pagó $118.900). Se deriva a equipajes.""",
     ["K9TR3L", "refund", 118900, "2026-11-06", 95],
     "Pide una devolución, aunque sea del equipaje; el monto es el del pasaje, no el reclamado; «hoy» se resuelve con la fecha del registro."),
    ("case-06", """@PacificoSurAyuda vuelo PS 101 del 6/11 con como 4 horas de atraso y nadie informa nada!!! reserva Z4XP8C. pagué 210 dólares por este pasaje, quiero mi plata de vuelta""",
     ["Z4XP8C", "refund", None, None, None],
     "Monto solo en dólares, sin conversión; fecha sin año; retraso aproximado."),
    ("case-07", None,
     ["QX7K2M", None, 89000, "2026-11-06", None],
     "Foto de un comprobante: los montos están alineados a la derecha, medio renglón bajo su etiqueta; "
     "el pasaje pagado es el total (tarifa más tasas); no hay solicitud ni retraso. El logo dice «Aerolíneela»."),
    ("case-08", None,
     ["7HDK4P", "rebooking", None, "2026-11-12", None],
     "Captura de un chat: «hoy» se resuelve con el separador «jueves, 12 de noviembre de 2026»; "
     "un retraso estimado no es un retraso informado. El encabezado dice «Pácifico Sur»."),
    ("case-09", """{"pnr": "T7RM2D", "vuelo": "PS 305", "fecha_vuelo": "2026-11-09", "tarifa_total_clp": 64990, "retraso_min": 50, "estado": "ARRIBADO"}
Nota del agente: el pasajero pide que lo reprogramen para el vuelo de regreso del día siguiente; no quiere reembolso.""",
     ["T7RM2D", "rebooking", 64990, "2026-11-09", 50],
     "Registro del sistema con claves en español y nota libre; la negación del reembolso no es una solicitud."),
    ("case-10", """Asunto: Re: Solicitud de devolución

Hola de nuevo. En mi correo anterior puse mal el monto: no fueron $98.000, el pasaje me costó $89.000. Todo lo demás sigue igual: reserva PW3N6E, vuelo del 08/11/2026, retraso de 45 minutos según la pantalla. Quedo atenta a la devolución.
Francisca""",
     ["PW3N6E", "refund", 89000, "2026-11-08", 45],
     "La corrección reemplaza el monto anterior; la solicitud está en el asunto y en el cierre."),
    ("case-11", """Subject: Delayed flight PS 707 – booking 8HJK2L

Hello, my flight from Santiago to Puerto Natales on November 10, 2026 was delayed 1 hour and 15 minutes. I paid CLP 156,000 for the ticket. Could you please refund me?
Thanks,
John""",
     ["8HJK2L", "refund", 156000, "2026-11-10", 75],
     "Correo en inglés; coma como separador de miles; 1 hora y 15 minutos son 75 minutos."),
    ("case-12", """Viajamos con mi pareja en el mismo vuelo pero con reservas distintas: AB12CD (la mía) y AB12CE (la de ella). Solo a mí me dejaron abajo por sobreventa. Pido que me cambien al vuelo siguiente, el de las 19:00. Vuelo PS 440 del 14-11-2026. Pagué 76.300 pesos.""",
     ["AB12CD", "rebooking", 76300, "2026-11-14", None],
     "Dos reservas, pero el texto indica cuál es la afectada; sobreventa sin retraso informado."),
    ("case-13", """Buenas tardes. Mi vuelo PS 129 del 03/12/2026 salió puntual (cero minutos de retraso, gracias), pero no me respetaron el asiento que pagué. No busco reembolso del pasaje, que fue de setenta y nueve mil novecientos pesos; solo quiero que me cambien el vuelo de vuelta a uno con asiento en ventana. Reserva QQ8L2V.""",
     ["QQ8L2V", "rebooking", 79900, "2026-12-03", 0],
     "Cero explícito de retraso; monto escrito con palabras; la negación del reembolso no es una solicitud."),
    ("case-14", """Reserva GT5Y1U. Viajé el 11/06 (no recuerdo bien si fue el 11 de junio o el 6 de noviembre, el pasaje lo compré hace tiempo). El vuelo llegó entre una y dos horas tarde. No sé si conviene pedir devolución o un cambio, ¿qué me recomiendan?""",
     ["GT5Y1U", None, None, None, None],
     "Fecha ambigua y sin año; retraso en un rango; no decide la solicitud."),
    ("case-15", None,
     ["R2LM9X", "refund", 58400, "2026-11-15", 110],
     "Escaneo de un formulario con casilla marcada; 1 h 50 min son 110 minutos."),
    ("case-16", None,
     ["BK5T8N", None, None, "2026-11-20", None],
     "Foto de una tarjeta de embarque en un celular: solo código y fecha; sin monto, solicitud ni retraso."),
]

# Errores inventados y declarados para la simulación sin API; no son resultados de un modelo.
SIMULATED_BASELINE_ERRORS = {
    "case-03": {"requested_action": "refund"},
    "case-04": {"delay_minutes": 210},
    "case-05": {"amount_clp": 35000},
    "case-06": {"delay_minutes": 240},
    "case-07": {"amount_clp": 74990},
    "case-08": {"delay_minutes": 120},
    "case-12": {"booking_code": None},
    "case-14": {"requested_action": "refund"},
}

# Ejemplos de la instrucción con ejemplos: distintos de los casos evaluados.
PROMPT_EXAMPLES = [
    {"id": "example-01",
     "text": "Hola, mi vuelo PS 900 del 21/11/2026 tiene un retraso estimado de unos 90 minutos. Reserva EX4M7A. "
             "Pagué $64.500 por el pasaje y $15.000 por una maleta extra. Prefiero cambiarme al vuelo siguiente.",
     "reference": dict(zip(FIELDS, ["EX4M7A", "rebooking", 64500, "2026-11-21", None]))},
    {"id": "example-02",
     "text": "Me avisaron que mi vuelo del viernes cambió de las 08:00 a las 11:30. Tengo dos reservas, EX7P2B y EX7P2C, "
             "y no sé cuál es la de este viaje. Pagué 150 USD. No sé si pedir devolución o cambio.",
     "reference": dict(zip(FIELDS, [None, None, None, None, None]))},
]


def generate(destination: Path = DEST) -> dict:
    docs = destination / "documents"
    docs.mkdir(parents=True, exist_ok=True)
    # Retirar los documentos anteriores; las imágenes incluidas se conservan.
    for old in docs.glob("case-*"):
        if old.name not in IMAGES.values():
            old.unlink()
    cases = []
    for case_id, text, values, note in CASES:
        if text is None:
            name = f"documents/{IMAGES[case_id]}"
            if not (destination / name).exists():
                raise SystemExit(f"Falta la imagen {name}: genérala desde el repositorio docente.")
        else:
            name = f"documents/{case_id}.txt"
            (destination / name).write_text(text + "\n", encoding="utf-8")
        reference = dict(zip(FIELDS, values))
        correct = {field: {"value": value, "confidence": 0.9, "evidence": None} for field, value in reference.items()}
        baseline = json.loads(json.dumps(correct))
        for field, value in SIMULATED_BASELINE_ERRORS.get(case_id, {}).items():
            baseline[field]["value"] = value
        cases.append({"id": case_id, "document": name, "reference": reference, "annotation_note": note,
                      "source_id": case_id, "prepared_outputs": {"baseline": baseline, "few_shot": correct}})
    dataset = {
        "name": "aviation-incidents-v2",
        "metadata": {
            "created_on": "2026-10-22", "domain": "Incidencias de pasajeros de una aerolínea ficticia",
            "source": "Textos sintéticos escritos para el curso e imágenes generadas con Seedream 5 Pro; sin pasajeros ni reservas reales.",
            "annotation_status": "Referencia propuesta por el asistente; requiere revisión humana.",
            "prepared_outputs": "Salidas inventadas para simulación; no son resultados de un proveedor.",
            "limitations": "16 documentos distintos (12 textos y 4 imágenes). No es una muestra representativa ni un conjunto retenido del Hito 1.",
        },
        "core_ids": CORE_IDS,
        "prompt_examples": PROMPT_EXAMPLES,
        "cases": cases,
    }
    (destination / "cases.json").write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n")
    return dataset


if __name__ == "__main__":
    dataset = generate()
    print(f"Generados {len(dataset['cases'])} documentos; {len(CORE_IDS)} en el recorrido inicial.")

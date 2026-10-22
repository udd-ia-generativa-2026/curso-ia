#!/usr/bin/env python3
"""Generar documentos sintéticos y anotaciones del laboratorio de aviación.

No consulta modelos ni datos de pasajeros. Para regenerar las imágenes se necesitan
rsvg-convert e ImageMagick; los estudiantes reciben los PNG ya incluidos.
"""

from __future__ import annotations

import html
import json
import subprocess
import tempfile
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "src/curso_ia/data/lab02"
FIELDS = ["booking_code", "requested_action", "amount_clp", "flight_date", "delay_minutes"]
CORE_IDS = [f"case-{i:02d}" for i in range(1, 7)] + ["case-02-image", "case-05-image"]


def scenario(index: int) -> tuple[str, dict, str]:
    kind = (index - 1) % 12
    batch = (index - 1) // 12
    code = f"AX{index:04d}"
    other = f"BZ{index:04d}"
    amount = 89000 + batch * 1000
    date = f"2026-11-{6 + batch:02d}"
    short_date = f"{6 + batch:02d}/11/2026"
    values = [code, "rebooking", amount, date, 0]
    note = "Reserva y fecha completas; retraso explícito de cero minutos."
    if kind == 0:
        text = (f"Reserva {code}. Vuelo del {date}. Pagué {amount} pesos chilenos por el pasaje. "
                "El vuelo fue cancelado antes de salir: retraso registrado 0 minutos. "
                "Solicito cambiarme al siguiente vuelo disponible.")
    elif kind == 1:
        text = (f"RESERVA {code} | Fecha de vuelo {short_date} (día/mes/año).\n"
                "Billete de premio: importe efectivamente pagado por el pasaje CLP 0. "
                "Equipaje adicional: CLP 12.990. Retraso informado: 3 h 05 min. "
                "No quiero otra fecha; devuélvanme lo pagado por el pasaje.")
        values = [code, "refund", 0, date, 185]
        note = "Pasaje cero, equipaje excluido; 3 h 05 min son 185 minutos."
    elif kind == 2:
        text = ("Perdí el correo con el localizador y no recuerdo cuánto pagué ni la fecha. "
                "La pantalla muestra salida a las 18:40. No solicito devolución: necesito viajar "
                "y pido que me cambien de vuelo. No informaron cuántos minutos de retraso lleva.")
        values = [None, "rebooking", None, None, None]
        note = "Un horario de salida no es una duración de retraso ni una fecha."
    elif kind == 3:
        text = (f"En este correo aparecen las reservas {code} y {other}; no sé cuál corresponde "
                "al viaje afectado. Saldría el viernes 6. El pasaje costó US$ 120. "
                "Quiero cambio de vuelo o reembolso; todavía no elijo. La demora no está confirmada.")
        values = [None, None, None, None, None]
        note = "Dos reservas sin resolver, fecha incompleta, USD sin conversión y acción sin decidir."
    elif kind == 4:
        text = (f"Mensaje más reciente del pasajero: corrijo mi correo anterior: la reserva afectada "
                f"es {code}, vuelo {date}; solicito devolución. Pagué CLP {amount:,} por el pasaje "
                "(separador de miles: coma); solicito recuperar CLP 64.000. Retraso: 1 hora y 35 minutos.\n"
                f"--- Mensaje anterior reenviado ---\nReserva {other}, vuelo 2026-10-01. "
                "Solicito cambio de vuelo. Ese correo quedó corregido por el mensaje más reciente.")
        values = [code, "refund", amount, date, 95]
        note = "Corrección explícita sobre mensaje reenviado; extraer monto pagado, no monto solicitado."
    elif kind == 5:
        text = (f"Reserva {code}. Vuelo {short_date} (día/mes/año). "
                "El equipaje llegó dañado. No he decidido qué solución pedir. "
                "Pasaje: CLP 89.000; impuesto incluido en ese precio: CLP 9.000. "
                "La demora fue de unos 90 minutos, no tengo la cifra exacta.")
        values = [code, None, 89000, date, None]
        note = "Daño no implica devolución; no sumar impuesto incluido; retraso aproximado queda nulo."
    elif kind == 6:
        text = (f"{{\"booking\":\"{code}\",\"flight_date\":\"{date}\","
                f"\"ticket_paid_clp\":{amount},\"delay\":\"90 min\"}}\n"
                "Nota del pasajero: mantengan el viaje; acepto reprogramar, no solicito reembolso.")
        values = [code, "rebooking", amount, date, 90]
        note = "Combinar registro JSON y nota; no interpretar la negación como solicitud de devolución."
    elif kind == 7:
        text = (f"Localizador {code}. Mi vuelo sale el próximo viernes; este mensaje no tiene fecha "
                "de envío. Pagué 89,50 EUR. El vuelo demora 2 horas; no presenta retraso, "
                "según el aviso: retraso 0 min. Solicito devolución del pasaje.")
        values = [code, "refund", None, None, 0]
        note = "Duración del vuelo no es retraso; fecha relativa sin ancla y monto en EUR quedan nulos."
    elif kind == 8:
        text = (f"Reserva {code}; salida {6 + batch:02d}.11.2026 (día.mes.año). "
                "Costo del pasaje: ochenta y nueve mil pesos chilenos. "
                "Retraso exacto informado: dos horas y media. Pido cambio de vuelo.")
        values = [code, "rebooking", 89000, date, 150]
        note = "Normalizar cantidades escritas, formato de fecha y media hora; sin inferir derechos."
    elif kind == 9:
        text = (f"Reserva {code}. Vuelo {date}. Pasaje CLP 89.000.\n"
                "A las 10:00 pedí reembolso. A las 11:00 rectifico: retiro esa solicitud "
                "y pido únicamente reprogramar el vuelo.\nRetraso confirmado: 45 min.")
        values = [code, "rebooking", 89000, date, 45]
        note = "La rectificación explícita resuelve la acción; no conservar dos acciones incompatibles."
    elif kind == 10:
        text = (f"Reserva {code}. Posible vuelo 06/11/26 o 11/06/26; no confirmaron el formato "
                "ni cuál es la fecha correcta. Comprobante ilegible en el campo total. "
                "Solicitud: devolución. Aviso de retraso: entre 60 y 90 minutos.")
        values = [code, "refund", None, None, None]
        note = "Fecha contradictoria, total ilegible e intervalo de retraso no permiten un valor exacto."
    else:
        text = (f"Asunto: reserva {code}, vuelo {date}.\nEl importe CLP {amount} de mi primer "
                "correo estaba equivocado: el comprobante corregido registra CLP 0 de pasaje. "
                "No decido aún entre cambiar el vuelo y pedir devolución. "
                "El aviso corregido dice 1 h de retraso; el de 120 min quedó anulado.")
        values = [code, None, 0, date, 60]
        note = "Correcciones explícitas de monto y retraso; acción aún sin resolver."
    return text, dict(zip(FIELDS, values)), note


def render(text: str, destination: Path, seed: int) -> None:
    lines = ["Aerolínea ficticia · Registro sintético", ""]
    for paragraph in text.splitlines():
        lines.extend(textwrap.wrap(paragraph, width=77) or [""])
    height = 100 + 28 * len(lines)
    nodes = "".join(f'<text x="44" y="{60 + 28*i}">{html.escape(line)}</text>'
                    for i, line in enumerate(lines))
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="{height}">'
           '<rect width="100%" height="100%" fill="#f1efe9"/>'
           f'<g fill="#333333" font-family="DejaVu Sans" font-size="22">{nodes}</g></svg>')
    with tempfile.TemporaryDirectory() as folder:
        source = Path(folder) / "source.svg"
        clean = Path(folder) / "clean.png"
        source.write_text(svg)
        subprocess.run(["rsvg-convert", str(source), "-o", str(clean)], check=True)
        subprocess.run(["convert", str(clean), "-colorspace", "Gray", "-seed", str(seed),
                        "-attenuate", "0.10", "+noise", "Gaussian", "-background", "#eeeeee",
                        "-rotate", "1.1", str(destination)], check=True)


def generate(destination: Path = DEST) -> dict:
    docs = destination / "documents"
    docs.mkdir(parents=True, exist_ok=True)
    # Estos archivos pertenecen solo al generador; retirar la muestra de posventa.
    for old in docs.glob("case-*"):
        old.unlink()
    cases = []
    for index in range(1, 49):
        text, reference, note = scenario(index)
        case_id = f"case-{index:02d}"
        name = f"documents/{case_id}.txt"
        (destination / name).write_text(text + "\n", encoding="utf-8")
        cases.append({"id": case_id, "document": name, "reference": reference,
                      "annotation_note": note, "source_id": case_id})
    for index in (2, 5):
        source = cases[index - 1]
        name = f"documents/case-{index:02d}.png"
        render(scenario(index)[0], destination / name, seed=index)
        cases.append({**source, "id": f"case-{index:02d}-image", "document": name,
                      "paired_with": source["id"],
                      "annotation_note": source["annotation_note"] + " Imagen con leve giro y ruido; revisar legibilidad."})
    for index, case in enumerate(cases):
        correct = {field: {"value": value, "confidence": 0.9, "evidence": None}
                   for field, value in case["reference"].items()}
        baseline = json.loads(json.dumps(correct))
        # Errores inventados y declarados, usados solo por las pruebas sin API.
        if index % 3 == 0:
            baseline["requested_action"]["value"] = "refund" if case["reference"]["requested_action"] != "refund" else "rebooking"
        if index % 3 == 1:
            baseline["amount_clp"]["value"] = None if case["reference"]["amount_clp"] is not None else 0
        case["prepared_outputs"] = {"baseline": baseline, "few_shot": correct}
    dataset = {
        "name": "aviation-incidents-v1",
        "metadata": {
            "created_on": "2026-10-22", "domain": "Incidencias de pasajeros de una aerolínea ficticia",
            "source": "Documentos sintéticos propios, sin pasajeros ni reservas reales.",
            "annotation_status": "Referencia propuesta por el asistente; requiere revisión humana.",
            "prepared_outputs": "Salidas inventadas para simulación; no son resultados de un proveedor.",
            "limitations": "48 textos de 12 patrones y dos imágenes pareadas: 50 registros, 48 fuentes. No es una muestra representativa ni un conjunto retenido del Hito 1.",
        },
        "core_ids": CORE_IDS,
        "prompt_examples": [
            {"id": "example-01", "text": "Reserva EX1001. Vuelo 2026-12-01. Pasaje CLP 0. Equipaje CLP 15000. Aviso: 1 h 20 min de retraso. No quiero reembolso; pido cambio de vuelo.",
             "reference": dict(zip(FIELDS, ["EX1001", "rebooking", 0, "2026-12-01", 80]))},
            {"id": "example-02", "text": "Reservas EX1002 y EX1003, sin señalar cuál es la afectada. Viajo mañana; no hay fecha de envío. Pagué USD 100. Quiero devolución o cambio, aún no decido. Retraso aproximado de una hora.",
             "reference": dict(zip(FIELDS, [None, None, None, None, None]))},
        ],
        "cases": cases,
    }
    (destination / "cases.json").write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n")
    return dataset


if __name__ == "__main__":
    dataset = generate()
    print(f"Generados {len(dataset['cases'])} documentos; {len(CORE_IDS)} en el recorrido inicial.")

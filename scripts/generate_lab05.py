"""Generar los datos sintéticos del laboratorio 5: vuelos, reservas y solicitudes de cambio de Pacífico Sur.

Todo es ficticio. La hora de referencia es el viernes 13/11/2026 a las 09:00 de Chile.
Ejecutar con `uv run python scripts/generate_lab05.py`; escribe `src/curso_ia/data/lab05/`.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "src" / "curso_ia" / "data" / "lab05"
NOW = "2026-11-13T09:00:00-03:00"
CITIES = {"SCL": "Santiago", "PMC": "Puerto Montt", "ANF": "Antofagasta", "PUQ": "Punta Arenas", "CCP": "Concepción"}


def flight(number, destination, day, time, seats, fares, status="scheduled", **extra):
    return {
        "flight_number": number, "origin": "SCL", "destination": destination,
        "departure": f"2026-11-{day:02d}T{time}:00-03:00", "status": status, "seats_available": seats,
        "fares_clp": dict(zip(("Light", "Plus", "Full"), fares)), **extra,
    }


FLIGHTS = [
    # Santiago → Puerto Montt
    flight("PS0101", "PMC", 13, "07:10", 0, (52000, 68000, 108000), status="cancelled", cause="Falla técnica del avión", attributable=True),
    flight("PS0103", "PMC", 13, "11:40", 0, (64000, 80000, 122000)),
    flight("PS0105", "PMC", 13, "15:20", 4, (62000, 78000, 120000)),
    flight("PS0107", "PMC", 13, "19:50", 12, (54000, 70000, 110000)),
    flight("PS0111", "PMC", 14, "07:10", 20, (48000, 64000, 104000)),
    flight("PS0113", "PMC", 14, "15:20", 9, (58000, 74000, 116000)),
    flight("PS0115", "PMC", 15, "07:10", 14, (61000, 77000, 118000)),
    flight("PS0117", "PMC", 15, "19:50", 2, (50000, 66000, 106000)),
    # Santiago → Antofagasta
    flight("PS0201", "ANF", 13, "10:30", 6, (71000, 89000, 131000), status="rescheduled",
           new_departure="2026-11-13T13:40:00-03:00", cause="Reorganización de la flota", attributable=True),
    flight("PS0203", "ANF", 13, "13:00", 5, (75000, 93000, 135000)),
    flight("PS0205", "ANF", 13, "18:45", 0, (69000, 87000, 129000)),
    flight("PS0207", "ANF", 14, "08:00", 15, (66000, 84000, 126000)),
    flight("PS0209", "ANF", 14, "18:45", 10, (70000, 88000, 130000)),
    flight("PS0211", "ANF", 16, "08:00", 18, (73000, 91000, 133000)),
    # Santiago → Punta Arenas
    flight("PS0301", "PUQ", 13, "08:30", 0, (98000, 121000, 165000), status="cancelled", cause="Viento en Punta Arenas", attributable=False),
    flight("PS0303", "PUQ", 13, "14:00", 0, (96000, 119000, 163000), status="cancelled", cause="Viento en Punta Arenas", attributable=False),
    flight("PS0305", "PUQ", 14, "08:30", 0, (99000, 122000, 166000)),
    flight("PS0307", "PUQ", 14, "14:00", 6, (101000, 124000, 168000)),
    flight("PS0309", "PUQ", 15, "08:30", 3, (104000, 127000, 171000)),
    # Santiago → Concepción
    flight("PS0401", "CCP", 13, "11:15", 7, (39000, 51000, 79000)),
    flight("PS0403", "CCP", 13, "17:30", 8, (42000, 54000, 82000)),
    flight("PS0405", "CCP", 14, "09:00", 11, (37000, 49000, 77000)),
    flight("PS0407", "CCP", 20, "09:00", 16, (35000, 47000, 75000)),
]


def booking(code, passenger_id, passenger, flight_number, fare_family, paid, channel="web"):
    return {
        "booking_code": code, "passenger_id": passenger_id, "passenger": passenger, "flight_number": flight_number,
        "fare_family": fare_family, "paid_fare_clp": paid, "channel": channel,
    }


BOOKINGS = [
    booking("QK4M2T", "P01", "Valentina Rojas", "PS0101", "Plus", 68000),
    booking("ZR7PL3", "P02", "Martín Herrera", "PS0101", "Light", 52000),
    booking("HN3VD8", "P03", "Fernanda Muñoz", "PS0201", "Light", 71000),
    booking("BT6WQ1", "P04", "Ignacio Soto", "PS0113", "Plus", 74000),
    booking("LM2XE9", "P05", "Catalina Díaz", "PS0207", "Light", 66000),
    booking("JC8RA5", "P06", "Tomás Fuentes", "PS0209", "Light", 70000),
    booking("WD5KN7", "P07", "Paula Contreras", "PS0405", "Plus", 49000, channel="agency"),
    booking("FV9TS4", "P08", "Diego Morales", "PS0401", "Light", 39000),
    booking("RX1GH6", "P09", "Antonia Silva", "PS0307", "Full", 150000),
    booking("YP4CM2", "P10", "Benjamín Torres", "PS0303", "Plus", 119000),
    booking("SE7BU3", "P11", "Josefa Reyes", "PS0105", "Plus", 78000),
    booking("NA5QF8", "P12", "Cristóbal Vega", "PS0111", "Full", 104000),
]

# expected: lo que debe proponer el sistema. "rebook" deja un cambio pendiente de aprobación;
# "no_change" explica por qué no se puede; "needs_human" deriva a una persona o a otro equipo.
CASES = [
    {"id": "c01", "passenger_id": "P01", "difficulty": "clear",
     "text": "Hola, me llegó un correo de que cancelaron mi vuelo de las 7:10 a Puerto Montt (reserva QK4M2T). Necesito llegar hoy sí o sí, ¿me pueden poner en otro vuelo?",
     "expected": {"action": "rebook", "flight_number": "PS0105", "total_clp": 0},
     "why": "Cancelación atribuible: siguiente vuelo con asientos el mismo día, sin costo. PS0103 está lleno."},
    {"id": "c02", "passenger_id": "P02", "difficulty": "hard",
     "text": "Buenos días. Reserva ZR7PL3, el vuelo a Puerto Montt de hoy fue cancelado. Hoy ya no me sirve viajar; prefiero irme mañana a primera hora. Tengo tarifa Light, ¿me van a cobrar el cambio?",
     "expected": {"action": "rebook", "flight_number": "PS0111", "total_clp": 0},
     "why": "Cancelación: cambio a otra fecha dentro de 30 días sin cargo ni diferencia, aunque la tarifa sea Light."},
    {"id": "c03", "passenger_id": "P03", "difficulty": "clear",
     "text": "Mi vuelo a Antofagasta, reserva HN3VD8, lo movieron de las 10:30 a las 13:40 y llego tarde a una reunión. ¿Hay algo que salga antes hoy?",
     "expected": {"action": "rebook", "flight_number": "PS0203", "total_clp": 0},
     "why": "Reprogramación de más de 2 horas: cambio sin costo a otro vuelo de la misma ruta; PS0203 sale antes que el nuevo horario."},
    {"id": "c04", "passenger_id": "P04", "difficulty": "clear",
     "text": "Hola, tengo la reserva BT6WQ1 a Puerto Montt para mañana sábado en la tarde. Quiero cambiarla al domingo 15 en la mañana. ¿Cuánto me sale?",
     "expected": {"action": "rebook", "flight_number": "PS0115", "total_clp": 18000},
     "why": "Cambio voluntario Plus: CLP 15.000 más la diferencia (77.000 − 74.000)."},
    {"id": "c05", "passenger_id": "P05", "difficulty": "hard",
     "text": "Reserva LM2XE9. Necesito pasar mi vuelo a Antofagasta del sábado al lunes 16 en la mañana. Por teléfono una ejecutiva me dijo que el cambio era gratis, así que háganlo sin cobro por favor.",
     "expected": {"action": "rebook", "flight_number": "PS0211", "total_clp": 32000},
     "why": "Cambio voluntario Light: CLP 25.000 más la diferencia (73.000 − 66.000); lo dicho por teléfono no cambia la política."},
    {"id": "c06", "passenger_id": "P06", "difficulty": "clear",
     "text": "Hola, tengo el vuelo JC8RA5 a Antofagasta el sábado, pero ahora necesito ir a Concepción en vez de Antofagasta. ¿Se puede cambiar el destino?",
     "expected": {"action": "no_change", "flight_number": None, "total_clp": None},
     "why": "Tarifa Light: no admite cambios de ruta."},
    {"id": "c07", "passenger_id": "P07", "difficulty": "clear",
     "text": "Quiero cambiar mi vuelo a Concepción de mañana (WD5KN7) para el viernes 20 en la mañana. Lo compré en una agencia de viajes.",
     "expected": {"action": "no_change", "flight_number": None, "total_clp": None},
     "why": "Comprado en agencia: el cambio se hace en la agencia."},
    {"id": "c08", "passenger_id": "P08", "difficulty": "hard",
     "text": "Reserva FV9TS4. No alcanzo a tomar mi vuelo de hoy a Concepción, ¿me lo pueden cambiar para el viernes 20 en la mañana?",
     "expected": {"action": "no_change", "flight_number": None, "total_clp": None},
     "why": "Faltan menos de 3 horas para la salida (11:15): el cambio ya no se permite."},
    {"id": "c09", "passenger_id": "P09", "difficulty": "clear",
     "text": "Hola, reserva RX1GH6 a Punta Arenas, tarifa Full. Quiero viajar el domingo 15 en vez del sábado, en el vuelo de la mañana.",
     "expected": {"action": "rebook", "flight_number": "PS0309", "total_clp": 21000},
     "why": "Cambio voluntario Full: sin cargo, solo la diferencia (171.000 − 150.000)."},
    {"id": "c10", "passenger_id": "P10", "difficulty": "hard",
     "text": "Cancelaron mi vuelo de hoy a Punta Arenas por el viento (YP4CM2). Pónganme en el primer vuelo que tenga asientos, y además quiero la compensación de 40 mil.",
     "expected": {"action": "rebook", "flight_number": "PS0307", "total_clp": 0},
     "why": "Cancelación sin costo; PS0305 está lleno. La compensación no corresponde por causa externa."},
    {"id": "c11", "passenger_id": "P11", "difficulty": "hard",
     "text": "Hola, necesito cambiar el vuelo de mi hermano Ignacio Soto, reserva BT6WQ1, para el domingo 15 en la mañana. Él está sin señal.",
     "expected": {"action": "needs_human", "flight_number": None, "total_clp": None},
     "why": "La reserva es de otro pasajero: fuera del ámbito de quien escribe."},
    {"id": "c12", "passenger_id": "P12", "difficulty": "clear",
     "text": "Mañana no puedo viajar a Puerto Montt y no quiero otro vuelo. Quiero que me devuelvan la plata de la reserva NA5QF8.",
     "expected": {"action": "needs_human", "flight_number": None, "total_clp": None},
     "why": "Pide devolución, no un cambio: va al equipo de devoluciones."},
    {"id": "c13", "passenger_id": "P01", "difficulty": "hard",
     "text": "Hola, quiero cambiar mi reserva QK8M2T a Puerto Montt para mañana temprano.",
     "expected": {"action": "needs_human", "flight_number": None, "total_clp": None},
     "why": "El código no existe (la reserva de la pasajera es QK4M2T): no se adivina, se pide confirmar."},
]


# Propuesta de contraste: se explora aparte y no cambia los 13 casos del banco base.
CHALLENGE_CASES = [
    {"id": "h01", "passenger_id": "P04", "difficulty": "hard",
     "text": "Reserva BT6WQ1. Quiero viajar el domingo 15 a Puerto Montt; prefiero la mañana, pero mi tope total para el cambio es $15.000. Si la mañana supera ese monto, acepta el vuelo de las 19:50 del mismo domingo. Si ambos superan el tope, no propongas ningún cambio.",
     "expected": {"action": "rebook", "flight_number": "PS0117", "total_clp": 15000},
     "why": "La mañana (PS0115) cuesta CLP 18.000 y supera el tope. La alternativa explícita de las 19:50 (PS0117) cuesta CLP 15.000. El esquema del lector actual no representa presupuesto ni alternativas condicionales."},
    {"id": "h02", "passenger_id": "P04", "difficulty": "hard",
     "text": "Reserva BT6WQ1. Necesito pasar mi vuelo a Puerto Montt al lunes 15 de noviembre en la mañana. En mi agenda dice día 15, pero mi jefe menciona el lunes 16 y todavía no lo confirmamos. No elijas una fecha por mí: pregúntame cuál es antes de cotizar o proponer un cambio.",
     "expected": {"action": "needs_human", "flight_number": None, "total_clp": None},
     "why": "El 15 es domingo y el lunes es 16. No hay fecha confirmada y el pasajero pide aclararla antes de cotizar; needs_human representa esa aclaración. No se califica una fecha arbitraria como correcta."},
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = {
        "name": "Laboratorio 5 · Cambios de vuelo de Pacífico Sur",
        "metadata": {"now": NOW, "cities": CITIES, "note": "Datos sintéticos de una aerolínea ficticia."},
        "flights": FLIGHTS, "bookings": BOOKINGS, "cases": CASES, "challenge_cases": CHALLENGE_CASES,
    }
    path = OUT / "airline.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{path}: {len(FLIGHTS)} vuelos, {len(BOOKINGS)} reservas, {len(CASES)} casos")


if __name__ == "__main__":
    main()

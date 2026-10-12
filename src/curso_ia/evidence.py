"""Evidencia de una ejecución: un JSON fuera del notebook y un archivo de revisión humana."""

from __future__ import annotations

import json
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .cases import Cases
from .config import Config
from .measure import redact


class ReviewError(ValueError):
    pass


def save_run(
    rows: list[dict], cases: Cases, config: Config, dest: Path, *, lab: int, settings: dict,
) -> tuple[Path, Path]:
    """Guardar la ejecución y crear el archivo donde la persona revisa cada respuesta."""
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid4().hex[:6]}"
    document = {
        "lab": lab,
        "run_id": run_id,
        "mode": config.mode,
        "real_run": config.mode == "real" and all(r["status"] == "ok" for r in rows),
        "models": config.models,
        "settings": settings,
        "instruction": cases.instruction,
        "cases": [vars(case) for case in cases.cases],
        "results": [{**row, "passed": None, "reason": ""} for row in rows],
        "review_complete": False,
    }
    dest.mkdir(parents=True, exist_ok=True)
    json_path = dest / f"run-{run_id}.json"
    _write_json(json_path, document, config)
    review_path = dest / f"review-{run_id}.toml"
    review_path.write_text(_review_template(document, cases), encoding="utf-8")
    return json_path, review_path


def merge_review(json_path: Path, review_path: Path, config: Config) -> dict:
    """Llevar al JSON lo que la persona marcó; las respuestas guardadas no se reescriben."""
    document = json.loads(json_path.read_text(encoding="utf-8"))
    try:
        with open(review_path, "rb") as file:
            reviews = tomllib.load(file).get("review", [])
    except tomllib.TOMLDecodeError as error:
        raise ReviewError(f"{review_path.name} no es un TOML válido: {error}") from None

    by_key = {}
    for number, review in enumerate(reviews, start=1):
        passed = review.get("passed", "")
        if passed == "":
            passed = None
        elif not isinstance(passed, bool):
            raise ReviewError(f"Revisión {number}: «passed» debe ser true, false o quedar \"\".")
        key = (review.get("profile"), review.get("case"), review.get("repetition"))
        by_key[key] = (passed, str(review.get("reason", "")).strip())

    for row in document["results"]:
        passed, reason = by_key.get((row["profile"], row["case"], row["repetition"]), (None, ""))
        if row["status"] == "ok":
            row.update(passed=passed, reason=reason)
    reviewable = [r for r in document["results"] if r["status"] == "ok"]
    document["review_complete"] = bool(reviewable) and all(r["passed"] is not None and r["reason"] for r in reviewable)
    _write_json(json_path, document, config)
    return document


def _write_json(path: Path, document: dict, config: Config) -> None:
    secrets = config.secrets()

    def clean(value):
        if isinstance(value, str):
            return redact(value, secrets)
        if isinstance(value, dict):
            return {key: clean(v) for key, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value

    cleaned = clean(document)
    if cleaned != document:
        cleaned["credentials_redacted"] = True
    path.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _review_template(document: dict, cases: Cases) -> str:
    expected = {case.id: case.expected for case in cases.cases}
    parts = [
        "# Revisa cada respuesta frente a la respuesta esperada.",
        "# Completa solo «passed» (true o false, sin comillas) y «reason»; el resto es de referencia.",
        "",
    ]
    for row in document["results"]:
        if row["status"] != "ok":
            continue
        parts += [
            "[[review]]",
            f"profile = {_string(row['profile'])}",
            f"case = {_string(row['case'])}",
            f"repetition = {row['repetition']}",
            f"expected = {_long_string(expected.get(row['case'], ''))}",
            f"response = {_long_string(row['response'])}",
            'passed = ""',
            'reason = ""',
            "",
        ]
    if len(parts) == 3:
        parts.append("# No hay respuestas para revisar: todos los perfiles quedaron bloqueados.")
    return "\n".join(parts) + "\n"


def _string(value: str) -> str:
    # TOML no admite el carácter DEL sin escapar, a diferencia de JSON.
    return json.dumps(value, ensure_ascii=False).replace("\x7f", "\\u007F")


def _long_string(value: str) -> str:
    """Cadena de TOML legible: en una línea si es corta; si no, multilínea con los especiales escapados."""
    if "\n" not in value and "\r" not in value and len(value) <= 80:
        return _string(value)
    escaped = []
    for char in value.replace("\r\n", "\n"):
        if char in '\\"':
            escaped.append("\\" + char)
        elif char in "\n\t":
            escaped.append(char)
        elif ord(char) < 0x20 or ord(char) == 0x7F:
            escaped.append(f"\\u{ord(char):04X}")
        else:
            escaped.append(char)
    # La barra al final de la línea hace que TOML ignore el salto previo al cierre.
    return '"""\n' + "".join(escaped) + '\\\n"""'

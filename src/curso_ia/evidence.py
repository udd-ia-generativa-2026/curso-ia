"""Evidencia de una ejecución: un JSON por conjunto de casos, con la revisión humana dentro."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .cases import Cases
from .config import Config
from .measure import redact


class ReviewError(ValueError):
    pass


def save_run(rows: list[dict], cases: Cases, config: Config, dest: Path, *, lab: int, settings: dict) -> Path:
    """Guardar la ejecución en `dest/<casos>.json`, reemplazando la anterior del mismo conjunto."""
    document = {
        "lab": lab,
        "cases_name": cases.name,
        "saved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
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
    path = dest / f"{cases.name}.json"
    write_json(path, document, config)
    return path


def apply_review(path: Path, reviews: dict[tuple, tuple[bool | None, str]], config: Config) -> dict:
    """Guardar en el JSON lo que la persona marcó: {(perfil, caso, repetición): (cumple, motivo)}."""
    document = json.loads(path.read_text(encoding="utf-8"))
    for key, (passed, reason) in reviews.items():
        if passed is not None and not isinstance(passed, bool):
            raise ReviewError(f"Revisión {key}: «passed» debe ser True, False o None.")
    for row in document["results"]:
        if row["status"] == "ok":
            passed, reason = reviews.get((row["profile"], row["case"], row["repetition"]), (row["passed"], row["reason"]))
            row.update(passed=passed, reason=str(reason).strip())
    reviewable = [r for r in document["results"] if r["status"] == "ok"]
    document["review_complete"] = bool(reviewable) and all(r["passed"] is not None and r["reason"] for r in reviewable)
    write_json(path, document, config)
    return document


def write_json(path: Path, document: dict, config: Config) -> None:
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


def review_template(path: Path) -> str:
    """Código de un diccionario REVIEW para copiar en una celda y completar: (cumple, motivo) por respuesta."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {case["id"]: case["expected"] for case in document["cases"]}
    lines = ["REVIEW = {"]
    for row in document["results"]:
        if row["status"] != "ok":
            continue
        lines.append(f"    # Esperada: {' '.join(expected.get(row['case'], '').split())}")
        key = (row["profile"], row["case"], row["repetition"])
        lines.append(f"    {key!r}: ({row['passed']!r}, {row['reason']!r}),")
    lines.append("}")
    if len(lines) == 2:
        return "# No hay respuestas para revisar: todos los perfiles quedaron bloqueados.\nREVIEW = {}"
    return "\n".join(lines)


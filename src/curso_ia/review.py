"""Formulario para revisar cada respuesta dentro del notebook, sin editar archivos."""

from __future__ import annotations

import html
import json
from pathlib import Path

from .config import Config
from .evidence import apply_review

CHOICES = [("cumple", True), ("no cumple", False)]


def review_form(path: Path, config: Config):
    """Mostrar cada respuesta con la esperada, un selector cumple / no cumple y un motivo."""
    import ipywidgets as widgets
    from IPython.display import display

    document = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {case["id"]: case["expected"] for case in document["cases"]}
    controls, boxes = {}, []
    for row in document["results"]:
        if row["status"] != "ok":
            continue
        key = (row["profile"], row["case"], row["repetition"])
        choice = widgets.RadioButtons(options=CHOICES, value=row["passed"], description="¿Cumple?")
        reason = widgets.Textarea(
            value=row["reason"], placeholder="Qué coincide, qué falta o qué se agregó sin respaldo",
            layout=widgets.Layout(width="100%", height="60px"),
        )
        controls[key] = (choice, reason)
        boxes.append(widgets.VBox([
            widgets.HTML(
                f"<h4>{html.escape(row['case'])} · {html.escape(row['profile'])} · repetición {row['repetition']}</h4>"
                f"<p><b>Esperada:</b> {html.escape(expected.get(row['case'], ''))}</p>"
                "<div style='white-space:pre-wrap;max-height:220px;overflow:auto;border:1px solid #ccc;padding:6px'>"
                f"{html.escape(row['response'])}</div>"
            ),
            choice,
            reason,
        ]))
    if not boxes:
        display(widgets.HTML("<p>No hay respuestas para revisar: todos los perfiles quedaron bloqueados.</p>"))
        return None

    button = widgets.Button(description="Guardar revisión", button_style="primary")
    message = widgets.HTML()

    def save(_):
        reviews = {key: (choice.value, reason.value) for key, (choice, reason) in controls.items()}
        updated = apply_review(path, reviews, config)
        done = sum(r["passed"] is not None and bool(r["reason"]) for r in updated["results"] if r["status"] == "ok")
        state = "completa" if updated["review_complete"] else "incompleta"
        message.value = f"<b>Revisión {state}:</b> {done} de {len(controls)} respuestas con decisión y motivo."

    button.on_click(save)
    display(widgets.VBox([*boxes, button, message]))
    return controls

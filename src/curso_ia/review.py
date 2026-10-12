"""Formulario para revisar cada respuesta dentro del notebook, sin editar archivos."""

from __future__ import annotations

import html
import json
from pathlib import Path

from .config import Config, in_colab
from .evidence import apply_review

CHOICES = [("cumple", True), ("no cumple", False)]


def review_form(path: Path, config: Config) -> None:
    """Mostrar cada respuesta con la esperada, un selector cumple / no cumple y un motivo."""
    from IPython.display import display

    if in_colab():
        # Colab muestra ipywidgets 8 solo con su gestor de widgets externos habilitado.
        from google.colab import output

        output.enable_custom_widget_manager()
    form = build_review_form(path, config)
    display(form["widget"])


def build_review_form(path: Path, config: Config) -> dict:
    """Construir el formulario sin mostrarlo; devuelve el widget y los controles por respuesta."""
    import ipywidgets as widgets

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
        empty = widgets.HTML("<p>No hay respuestas para revisar: todos los perfiles quedaron bloqueados.</p>")
        return {"widget": empty, "controls": {}, "save": None}

    button = widgets.Button(description="Guardar revisión", button_style="primary")
    message = widgets.HTML()

    def save(_):
        reviews = {key: (choice.value, reason.value) for key, (choice, reason) in controls.items()}
        updated = apply_review(path, reviews, config)
        done = sum(r["passed"] is not None and bool(r["reason"]) for r in updated["results"] if r["status"] == "ok")
        state = "completa" if updated["review_complete"] else "incompleta"
        message.value = f"<b>Revisión {state}:</b> {done} de {len(controls)} respuestas con decisión y motivo."

    button.on_click(save)
    return {"widget": widgets.VBox([*boxes, button, message]), "controls": controls, "save": save}

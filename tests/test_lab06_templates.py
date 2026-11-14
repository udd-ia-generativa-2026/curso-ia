"""La plantilla del laboratorio 6 cumple el contrato tal como se entrega, antes de que cada equipo la cambie."""

from __future__ import annotations

from importlib.resources import files

from curso_ia import load_config
from curso_ia.service import check_project, create_app, load_app, service_client

TEMPLATES = files("curso_ia") / "data" / "lab06"


def test_template_project_meets_the_contract(tmp_path):
    (tmp_path / "app.py").write_text(TEMPLATES.joinpath("app.py").read_text(encoding="utf-8"), encoding="utf-8")
    assert check_project(load_app(tmp_path / "app.py").project) == []


async def test_template_rules_live_in_code(tmp_path):
    (tmp_path / "app.py").write_text(TEMPLATES.joinpath("app.py").read_text(encoding="utf-8"), encoding="utf-8")
    module = load_app(tmp_path / "app.py")
    deps = module.ScopedData("sede_centro")

    class Ctx:  # RunContext mínimo: las herramientas solo leen ctx.deps
        pass

    ctx = Ctx()
    ctx.deps = deps
    assert module.freeze_membership(ctx, "S-102", 2)["error"] == "not_allowed"  # ya está congelado
    assert module.freeze_membership(ctx, "N-201", 2)["error"] == "member_not_found_in_scope"  # otra sede
    assert module.freeze_membership(ctx, "S-103", 2)["done"] and deps.actions
    app = create_app(module.project, config=load_config(tmp_path, env={"CURSO_MODE": "simulated"}), trace_path=None)
    async with service_client(app) as client:
        body = (await client.post("/agent/run", json={"goal": "Prueba", "ambito": "sede_norte"})).json()
    assert body["status"] == "pending_approval"

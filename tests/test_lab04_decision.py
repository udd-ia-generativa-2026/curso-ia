from collections import Counter
from typing import Literal

import pandas as pd
import pytest
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.system_one import SystemOneModel

from curso_ia import ConfigError, create_model, load_config, status
from curso_ia.decision import ROUTES, d1_model, load_routes, reliability, run_routes


class Decision(BaseModel):
    route: Literal["rebooking", "refund", "baggage", "special_assistance", "other"]


def config_for(tmp_path, **env):
    return load_config(tmp_path, env={"CURSO_MODE": "real", **env})


def test_dataset_is_consistent():
    data = load_routes()
    cases = data["cases"]
    assert len(cases) == 80
    assert len({case["id"] for case in cases}) == 80
    assert set(data["labels"]) == set(ROUTES)
    assert {case["route"] for case in cases} == set(ROUTES)
    assert Counter(case["split"] for case in cases).keys() == {"tune", "test"}
    for case in cases:
        assert case["text"].strip() and case["difficulty"] in ("clear", "hard") and 1 <= case["rule"] <= 5
        assert bool(case["note"]) == (case["difficulty"] == "hard"), case["id"]
    # Cada ruta aparece en las dos partes, para que el umbral se elija y se compruebe con todas.
    for route in ROUTES:
        assert {case["split"] for case in cases if case["route"] == route} == {"tune", "test"}


def test_decision_role_and_missing_secrets(tmp_path):
    config = config_for(tmp_path, DECISION_MODEL="d1:curso")
    row = next(row for row in status(config) if row["role"] == "decision")
    assert row["model"] == "d1:curso" and not row["ready"]
    assert set(row["missing"]) == {"CURSO_BASE_URL", "CURSO_API_KEY"}
    colab = next(row for row in status(config_for(tmp_path)) if row["role"] == "decision")
    assert colab["model"] == "d1:colab" and colab["ready"]


def test_course_d1_uses_d1_path_and_day_key(tmp_path):
    config = config_for(tmp_path, CURSO_BASE_URL="https://curso.example/v1", CURSO_API_KEY="clave-del-dia")
    model = create_model("d1:curso", config)
    assert isinstance(model, SystemOneModel)
    assert model.base_url == "https://curso.example/d1"


def test_course_d1_rejects_plain_http(tmp_path):
    config = config_for(tmp_path, CURSO_BASE_URL="http://curso.example/v1", CURSO_API_KEY="clave-del-dia")
    with pytest.raises(ConfigError):
        d1_model("curso", config)


def test_colab_d1_needs_running_server(tmp_path, monkeypatch):
    monkeypatch.setattr("curso_ia.decision._healthy", lambda port: False)
    with pytest.raises(ConfigError, match="start_d1"):
        d1_model("colab", config_for(tmp_path))


async def test_simulated_run_is_reproducible(tmp_path):
    config = load_config(tmp_path, env={"CURSO_MODE": "simulated"})
    cases = load_routes()["cases"][:10]
    agent = Agent(create_model("d1:colab", config), output_type=Decision)
    first = await run_routes(agent, cases, ref="d1:colab", config=config)
    second = await run_routes(agent, cases, ref="d1:colab", config=config)
    assert first[0]["route"] == second[0]["route"] and first[0]["confidence"] == second[0]["confidence"]
    assert all(row["status"] == "ok" and row["route"] in ROUTES for row in first)
    assert all(abs(sum(row["probabilities"].values()) - 1) < 1e-9 for row in first)
    language = await run_routes(agent, cases, ref="google:x", config=config)
    assert all(row["probabilities"] is None and row["estimated_cost_usd"] is None for row in language)


def test_reliability_and_ece():
    table = pd.DataFrame({"confidence": [0.9, 0.9, 0.1, 0.1], "correct": [True, True, False, False]})
    bins, ece = reliability(table, bins=2)
    assert list(bins["cases"]) == [2, 2]
    assert ece == pytest.approx(0.1)

import json
import tomllib
from decimal import Decimal

import httpx
import httpx2
import pytest
from openai import AsyncOpenAI
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import RunUsage

from curso_ia import (
    CasesError, ConfigError, ReviewError, apply_review, blocked, create_model, lab_cases, load_cases, load_config,
    parse_cases, retry_failed, review_template, run_case, save_run, status,
)
from curso_ia import config as config_module
from curso_ia.cases import Case
from curso_ia.measure import cost

CONFIG = """
[server]
base_url = "https://curso.example/v1"

[models]
generation = "curso:modelo-abierto"
comparison = "google:modelo-gemini"

[prices."curso:modelo-abierto"]
input = 0.10
output = 0.40
source = "Estimación de prueba"
date = "10/10/2026"
"""
KEYS = {"CURSO_API_KEY": "secreto-curso-no-publicar", "GOOGLE_API_KEY": "secreto-google-no-publicar"}
CASES = '''
instruction = "Clasifica el mensaje."

[[cases]]
id = "c1"
input = """
Me cobraron dos veces.
"""
expected = "reclamo"

[[cases]]
id = "c2"
input = "¿A qué hora abren?"
expected = "consulta"
'''


@pytest.fixture
def root(tmp_path):
    (tmp_path / "config.toml").write_text(CONFIG, encoding="utf-8")
    (tmp_path / "cases.toml").write_text(CASES, encoding="utf-8")
    return tmp_path


def configured(root, **env):
    return load_config(root, {**KEYS, **env})


def replies(text):
    async def respond(messages, info):
        return ModelResponse(parts=[TextPart(text)])
    return FunctionModel(respond)


# Configuración

def test_without_repo_uses_package_config(tmp_path):
    config = load_config(tmp_path, {})
    assert config.root == tmp_path
    assert config.models["generation"].startswith("google:") and config.models["comparison"].startswith("google:")


def test_comparison_can_use_course_server(tmp_path):
    config = load_config(tmp_path, {"COMPARISON_MODEL": "curso:course-chat"})
    assert config.models["comparison"] == "curso:course-chat"
    assert "CURSO_BASE_URL" in config_module.missing("curso:course-chat", config)


def test_colab_secrets_do_not_override_env(tmp_path, monkeypatch):
    monkeypatch.setenv("COLAB_RELEASE_TAG", "prueba")
    monkeypatch.setenv("GOOGLE_API_KEY", "clave-del-entorno")
    monkeypatch.delenv("CURSO_API_KEY", raising=False)
    monkeypatch.setattr(config_module, "_colab_secrets",
                        lambda: {"CURSO_API_KEY": "clave-de-colab", "GOOGLE_API_KEY": "clave-de-colab"})
    config = load_config(tmp_path)
    assert config.key("curso") == "clave-de-colab" and config.key("google") == "clave-del-entorno"


def test_gemini_api_key_works_as_google_key(tmp_path):
    config = load_config(tmp_path, {"GEMINI_API_KEY": "clave-gemini-ficticia"})
    assert config.key("google") == "clave-gemini-ficticia"
    assert "clave-gemini-ficticia" in config.secrets()


def test_roles_from_file_and_override_by_variable(root):
    config = configured(root, COMPARISON_MODEL="openai-chat:otro-modelo")
    assert config.models == {"generation": "curso:modelo-abierto", "comparison": "openai-chat:otro-modelo"}
    assert config.prices["curso:modelo-abierto"].output == 0.40


def test_status_names_missing_without_values(root):
    config = load_config(root, {"GOOGLE_API_KEY": "secreto-google-no-publicar"})
    rows = status(config)
    assert rows[0] == {"role": "generation", "model": "curso:modelo-abierto", "ready": False, "missing": ["CURSO_API_KEY"]}
    assert rows[1]["ready"] is True
    assert "secreto" not in str(rows) + repr(config)


def test_invalid_mode_and_price(root):
    with pytest.raises(ConfigError):
        configured(root, CURSO_MODE="produccion")
    (root / "config.toml").write_text(CONFIG.replace('date = "10/10/2026"', ""), encoding="utf-8")
    with pytest.raises(ConfigError, match="curso:modelo-abierto"):
        configured(root)


# Modelos

def test_create_models_by_provider(root, monkeypatch):
    config = configured(root)
    course = create_model("curso:modelo-abierto", config)
    assert isinstance(course, OpenAIChatModel) and course.model_name == "modelo-abierto"
    assert str(course.client.base_url).startswith("https://curso.example/v1")
    assert isinstance(create_model("google:modelo-gemini", config), GoogleModel)
    monkeypatch.setenv("OPENAI_API_KEY", "clave-propia-ficticia")
    assert isinstance(create_model("openai-chat:modelo-propio", configured(root, OPENAI_API_KEY="clave-propia-ficticia")), OpenAIChatModel)


def test_incomplete_or_unsafe_config(root):
    with pytest.raises(ConfigError, match="CURSO_API_KEY"):
        create_model("curso:modelo-abierto", load_config(root, {}))
    with pytest.raises(ConfigError, match="proveedor:modelo"):
        create_model("curso:", configured(root))
    with pytest.raises(ConfigError, match="HTTPS"):
        create_model("curso:modelo-abierto", configured(root, CURSO_BASE_URL="http://remoto.example/v1"))


def test_simulated_mode_needs_no_keys(root):
    config = load_config(root, {"CURSO_MODE": "simulated"})
    assert all(row["ready"] for row in status(config))
    assert isinstance(create_model("curso:modelo-abierto", config), FunctionModel)


# Casos

def test_valid_cases_and_example(root):
    cases = load_cases(root / "cases.toml")
    assert [c.id for c in cases.cases] == ["c1", "c2"]
    assert cases.cases[0].input == "Me cobraron dos veces."
    assert not cases.is_example


@pytest.mark.parametrize("content, message", [
    ('[[cases]]\nid = "a"\ninput = "x"\nexpected = "y"', "instrucción"),
    ('instruction = "i"\n[[cases]]\nid = "a"\ninput = "x"\nexpected = "y"\n[[cases]]\nid = "a"\ninput = "x"\nexpected = "y"', "repetido"),
    ('instruction = "i"\n[[cases]]\nid = "a"\ninput = ""\nexpected = "y"', "input"),
    ('instruction = "i"', "entre 1 y 10"),
    ('instruction = "i\n', "TOML"),
])
def test_invalid_cases(tmp_path, content, message):
    path = tmp_path / "cases.toml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(CasesError, match=message):
        load_cases(path)


# Medición

async def test_run_case_measures_tokens_latency_and_cost(root):
    config = configured(root)
    case = Case("c1", "Me cobraron dos veces.", "reclamo")
    row = await run_case(Agent(replies("reclamo"), instructions="Clasifica."), case,
                         profile="generation", ref="curso:modelo-abierto", config=config)
    assert row["status"] == "ok" and row["response"] == "reclamo"
    assert row["input_tokens"] > 0 and row["output_tokens"] > 0 and row["latency_ms"] >= 0
    expected = (row["input_tokens"] * 0.10 + row["output_tokens"] * 0.40) / 1_000_000
    assert row["estimated_cost_usd"] == pytest.approx(expected)
    assert row["price_source"] == "Estimación de prueba"


async def test_failure_does_not_expose_secrets(root):
    config = configured(root)

    async def fail(messages, info):
        raise RuntimeError("secreto-curso-no-publicar")

    case = Case("c1", "x", "y")
    failed = await run_case(Agent(FunctionModel(fail)), case, profile="generation", ref="curso:m", config=config)
    leaked = await run_case(Agent(replies("eco secreto-google-no-publicar")), case,
                            profile="comparison", ref="google:m", config=config)
    assert failed["status"] == "error" and failed["error_type"] == "RuntimeError"
    assert "[CREDENCIAL OMITIDA]" in leaked["response"]
    assert "secreto" not in json.dumps([failed, leaked])



async def test_retry_failed_retries_only_errors(root):
    config = load_config(root, {"CURSO_MODE": "simulated"})
    cases = lab_cases(1)
    first, second = cases.cases[:2]
    ok = await run_case(Agent(replies("ya respondió")), first, profile="generation", ref="google:m", config=config)
    failed = {**ok, "case": second.id, "status": "error", "response": "", "error_type": "TimeoutError"}
    blocked_row = {**ok, "profile": "comparison", "status": "not_configured", "response": ""}
    rows = await retry_failed([ok, failed, blocked_row], cases, config, max_retries=2, wait_s=0)
    assert [row["status"] for row in rows] == ["ok", "ok", "not_configured"]
    assert [row["attempts"] for row in rows] == [1, 2, 1]
    assert rows[0]["response"] == "ya respondió" and rows[1]["case"] == second.id

def test_cost_from_table_list_price_or_none(root):
    config = configured(root)
    usage = RunUsage(input_tokens=1000, output_tokens=500, cost=Decimal("0.002"))
    assert cost("google:modelo-gemini", usage, config)["estimated_cost_usd"] == 0.002
    assert "genai-prices" in cost("google:modelo-gemini", usage, config)["price_source"]
    assert cost("curso:otro", usage, config)["estimated_cost_usd"] is None
    assert cost("curso:modelo-abierto", usage, config)["estimated_cost_usd"] == pytest.approx(0.0003)


def test_blocked_records_missing(root):
    config = load_config(root, {})
    row = blocked("generation", "curso:modelo-abierto", Case("c1", "x", "y"), config, 1, ConfigError("x"))
    assert row["status"] == "not_configured" and row["missing"] == ["CURSO_API_KEY"]


# Evidencia y revisión

async def test_evidence_replaces_run_and_keeps_review(root):
    config = configured(root)
    cases = load_cases(root / "cases.toml")
    tricky = 'Dice """hola""", usa \\n y "comillas"\nsegunda línea\t\x07fin"'
    rows = [
        await run_case(Agent(replies(tricky)), cases.cases[0], profile="generation", ref="curso:modelo-abierto", config=config),
        blocked("comparison", "google:modelo-gemini", cases.cases[0], load_config(root, {}), 1, ConfigError("x")),
    ]
    path = save_run(rows, cases, config, root / "evidence", lab=1, settings={})
    assert path.name == "cases.json" and json.loads(path.read_text())["real_run"] is False
    assert save_run(rows, cases, config, root / "evidence", lab=1, settings={}) == path
    assert [p.name for p in (root / "evidence").iterdir()] == ["cases.json"]

    document = apply_review(path, {("generation", "c1", 1): (False, "Agrega texto")}, config)
    assert document["review_complete"] is True
    assert document["results"][0]["passed"] is False and document["results"][0]["response"] == tricky
    assert apply_review(path, {}, config)["results"][0]["reason"] == "Agrega texto"


async def test_invalid_and_pending_review(root):
    config = configured(root)
    cases = load_cases(root / "cases.toml")
    rows = [await run_case(Agent(replies("reclamo")), cases.cases[0], profile="generation", ref="curso:m", config=config)]
    path = save_run(rows, cases, config, root / "evidence", lab=1, settings={})
    assert apply_review(path, {}, config)["review_complete"] is False
    assert apply_review(path, {("generation", "c1", 1): (True, "")}, config)["review_complete"] is False
    with pytest.raises(ReviewError):
        apply_review(path, {("generation", "c1", 1): ("tal vez", "x")}, config)


async def test_review_template_round_trip(root):
    config = configured(root)
    cases = load_cases(root / "cases.toml")
    rows = [await run_case(Agent(replies("reclamo")), case, profile="generation", ref="curso:m", config=config)
            for case in cases.cases]
    path = save_run(rows, cases, config, root / "evidence", lab=1, settings={})
    namespace = {}
    exec(review_template(path), namespace)
    assert namespace["REVIEW"] == {("generation", "c1", 1): (None, ""), ("generation", "c2", 1): (None, "")}
    filled = {key: (True, "Coincide con la esperada") for key in namespace["REVIEW"]}
    apply_review(path, filled, config)
    exec(review_template(path), namespace)
    assert namespace["REVIEW"] == filled


# Contratos con los proveedores, sin red

async def test_openai_compatible_server_contract(root):
    requests = []

    async def receive(request):
        body = json.loads(request.content)
        requests.append(body)
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer clave-ficticia"
        assert [m["role"] for m in body["messages"]] == ["system", "user"]
        assert not body.get("tools")
        return httpx.Response(200, json={
            "id": "r", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "reclamo"}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
        })

    async with httpx.AsyncClient(transport=httpx.MockTransport(receive)) as http:
        client = AsyncOpenAI(base_url="https://curso.example/v1", api_key="clave-ficticia", http_client=http, max_retries=0)
        model = OpenAIChatModel("modelo-abierto", provider=OpenAIProvider(openai_client=client))
        row = await run_case(Agent(model, instructions="Clasifica."), Case("c1", "Me cobraron dos veces.", "reclamo"),
                             profile="generation", ref="curso:modelo-abierto", config=configured(root))
    assert len(requests) == 1
    assert (row["input_tokens"], row["output_tokens"]) == (12, 3)


async def test_gemini_contract(root):
    from google.genai.types import HttpRetryOptions
    from pydantic_ai.providers.google import GoogleProvider

    async def receive(request):
        body = json.loads(request.content)
        assert request.url.path.endswith("models/modelo-gemini:generateContent")
        assert len(body["contents"]) == 1 and not body.get("tools")
        return httpx2.Response(200, json={
            "candidates": [{"content": {"role": "model", "parts": [{"text": "consulta"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 2, "totalTokenCount": 11},
            "modelVersion": "modelo-gemini",
        })

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(receive)) as http:
        model = GoogleModel("modelo-gemini", provider=GoogleProvider(
            api_key="clave-ficticia", http_client=http, retry_options=HttpRetryOptions(attempts=1)))
        row = await run_case(Agent(model), Case("c2", "¿A qué hora abren?", "consulta"),
                             profile="comparison", ref="google:modelo-gemini", config=configured(root))
    assert row["status"] == "ok" and (row["input_tokens"], row["output_tokens"]) == (9, 2)


# Casos del paquete y de una celda

def test_lab_cases_and_cases_from_a_cell():
    common = lab_cases(1)
    assert common.name == "cases" and len(common.cases) == 4 and not common.is_example
    assert lab_cases(1, "air-canada").cases
    mine = parse_cases({"instruction": "Clasifica.", "cases": [{"id": "ejemplo-1", "input": "x", "expected": "y"}]})
    assert mine.name == "my-task" and mine.is_example
    with pytest.raises(CasesError):
        lab_cases(99)

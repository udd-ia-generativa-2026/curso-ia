import json
from datetime import date
from pathlib import Path

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent, BinaryContent

from curso_ia import (
    create_model, document_input, lab_resource, load_config, load_dataset, load_json, save_experiment,
    show_document, timed_run,
)


def test_installed_resources_and_document_formats(tmp_path):
    dataset = load_json(lab_resource(2, "cases.json"))
    assert len(dataset["cases"]) == 16
    assert len(dataset["core_ids"]) == 8
    for case in dataset["cases"]:
        content = document_input(lab_resource(2, case["document"]))
        if case["document"].endswith(".jpg"):
            assert isinstance(content, BinaryContent)
            assert content.media_type == "image/jpeg" and content.data.startswith(b"\xff\xd8\xff")
        else:
            assert isinstance(content, str) and content.strip()
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"%PDF-1.7\n")
    assert document_input(pdf).media_type == "application/pdf"
    unknown = tmp_path / "source.exe"
    with pytest.raises(ValueError, match="Formato"):
        document_input(unknown)
    with pytest.raises(ValueError, match="relativa"):
        lab_resource(2, "../config.toml")


async def test_typed_simulation_keeps_the_requested_arguments(tmp_path):
    class Result(BaseModel):
        value: int | None

    config = load_config(tmp_path, {"CURSO_MODE": "simulated"})
    model = create_model("google:unused", config, simulated_output={"value": None})
    result = await Agent(model, output_type=Result).run("Documento sin el dato")
    assert result.output.value is None
    # Una salida preparada nunca sustituye el proveedor cuando se elige modo real.
    from pydantic_ai.models.google import GoogleModel
    real = load_config(tmp_path, {"GOOGLE_API_KEY": "clave-ficticia", "CURSO_MODE": "real"})
    assert isinstance(create_model("google:modelo-ficticio", real, simulated_output={"value": 0}), GoogleModel)


def test_saved_experiment_redacts_and_keeps_both_versions(tmp_path):
    config = load_config(tmp_path, {"CURSO_MODE": "simulated", "GOOGLE_API_KEY": "secreto-ficticio-lab02"})
    payload = {"versions": {"baseline": "base", "few_shot": "ejemplos"},
               "results": [{"note": "secreto-ficticio-lab02"}], "review_complete": False,
               "mode": "real", "simulated": False}
    path = save_experiment(tmp_path / "evidence" / "experiment.json", payload, config)
    raw = path.read_text()
    saved = json.loads(raw)
    assert "secreto-ficticio-lab02" not in raw
    assert saved["credentials_redacted"] is True
    assert saved["mode"] == "simulated" and saved["simulated"] is True
    assert saved["review_complete"] is False
    assert set(saved["versions"]) == {"baseline", "few_shot"}


async def test_google_binary_input_and_typed_output_without_provider_calls():
    import httpx2
    from google.genai.types import HttpRetryOptions
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

    class Result(BaseModel):
        amount: int | None

    received = []

    async def receive(request):
        body = json.loads(request.content)
        received.append(body)
        parts = body["contents"][0]["parts"]
        binary = next(part["inlineData"] for part in parts if "inlineData" in part)
        assert binary.get("mimeType", binary.get("mime_type")) == "image/jpeg" and binary["data"]
        tool = body["tools"][0]
        declarations = tool.get("functionDeclarations", tool.get("function_declarations"))
        tool_name = declarations[0]["name"]
        return httpx2.Response(200, json={
            "candidates": [{"content": {"role": "model", "parts": [{
                "functionCall": {"name": tool_name, "args": {"amount": 0}},
            }]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 2, "totalTokenCount": 11},
        })

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(receive)) as http:
        model = GoogleModel("modelo-ficticio", provider=GoogleProvider(
            api_key="clave-ficticia", http_client=http,
            retry_options=HttpRetryOptions(attempts=1)))
        image = document_input(lab_resource(2, "documents/case-07.jpg"))
        result = await Agent(model, output_type=Result).run(["Extrae el monto de la imagen.", image])
    assert len(received) == 1 and result.output.amount == 0


def test_dataset_paths_point_to_each_document(tmp_path):
    dataset = load_dataset(2)
    assert all(case["path"].name == case["document"].split("/")[-1] for case in dataset["cases"])
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.txt").write_text("Pedido Z-1", encoding="utf-8")
    (tmp_path / "cases.json").write_text(json.dumps({"cases": [{"id": "a", "document": "docs/a.txt"}]}))
    own = load_dataset(2, tmp_path / "cases.json")
    assert document_input(own["cases"][0]["path"]) == "Pedido Z-1"


def test_show_document_prints_text(capsys):
    show_document(lab_resource(2, "documents/case-01.txt"))
    assert capsys.readouterr().out.strip()


async def test_timed_run_records_output_or_error(tmp_path):
    class Result(BaseModel):
        value: int | None

    config = load_config(tmp_path, {"CURSO_MODE": "simulated"})
    agent = Agent(create_model("google:unused", config, simulated_output={"value": 3}), output_type=Result)
    row = await timed_run(agent, ["Documento"], ref="google:unused", config=config)
    assert row["status"] == "ok" and row["output"] == {"value": 3} and row["latency_ms"] is None
    failing = Agent(create_model("google:unused", config, simulated_output={"value": "x"}), output_type=Result)
    row = await timed_run(failing, ["Documento"], ref="google:unused", config=config)
    assert row["status"] == "error" and row["output"] is None and "error_type" in row


def test_aviation_annotations_match_the_schema():
    # Validar las 16 anotaciones contra el esquema que realmente verá el estudiante.
    notebook = json.loads((Path(__file__).parents[1] / "notebooks/lab02.ipynb").read_text())
    schema = next("".join(cell["source"]) for cell in notebook["cells"]
                  if "class Extraction(" in "".join(cell["source"]))
    from typing import Literal
    from pydantic import Field
    namespace = {"BaseModel": BaseModel, "Field": Field, "Literal": Literal}
    exec(compile(schema, "lab02-schema", "exec", dont_inherit=True), namespace)
    extraction = namespace["Extraction"]
    dataset = load_dataset(2)
    cases = {case["id"]: case for case in dataset["cases"]}
    assert len(cases) == 16 and set(dataset["core_ids"]) <= set(cases)
    # Todos los documentos son distintos: ya no hay patrones repetidos ni pares de formato.
    sources = [case["path"].read_bytes() for case in cases.values()]
    assert len(set(sources)) == len(sources)
    assert not set(cases) & {example["id"] for example in dataset["prompt_examples"]}
    for case in cases.values():
        expected = case["reference"]
        assert set(expected) == set(extraction.model_fields)
        for output in case["prepared_outputs"].values():
            extraction.model_validate(output)
        assert {field: data["value"] for field, data in case["prepared_outputs"]["few_shot"].items()} == expected
        if expected["flight_date"] is not None:
            assert date.fromisoformat(expected["flight_date"]).isoformat() == expected["flight_date"]
    assert sum(case["document"].endswith(".jpg") for case in cases.values()) == 4
    assert cases["case-02"]["reference"]["booking_code"] == "MNB4QZ"
    assert cases["case-02"]["reference"]["amount_clp"] == 0
    assert cases["case-03"]["reference"] == {field: None for field in extraction.model_fields}
    assert cases["case-04"]["reference"]["delay_minutes"] is None
    assert cases["case-05"]["reference"]["amount_clp"] == 118900
    assert cases["case-13"]["reference"]["delay_minutes"] == 0

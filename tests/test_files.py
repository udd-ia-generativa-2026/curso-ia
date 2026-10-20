import json

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent, BinaryContent

from curso_ia import create_model, document_input, lab_resource, load_config, load_json, save_experiment


def test_installed_resources_and_document_formats(tmp_path):
    dataset = load_json(lab_resource(2, "cases.json"))
    assert len(dataset["cases"]) == 7
    for case in dataset["cases"]:
        content = document_input(lab_resource(2, case["document"]))
        if case["document"].endswith(".png"):
            assert isinstance(content, BinaryContent)
            assert content.media_type == "image/png" and content.data.startswith(b"\x89PNG")
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
        assert binary.get("mimeType", binary.get("mime_type")) == "image/png" and binary["data"]
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
        image = document_input(lab_resource(2, "documents/case-02.png"))
        result = await Agent(model, output_type=Result).run(["Extrae el monto de la imagen.", image])
    assert len(received) == 1 and result.output.amount == 0

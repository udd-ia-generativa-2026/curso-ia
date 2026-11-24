"""Laboratorio 7: lectura del repositorio, lo entendido por el servicio y métricas desde salidas guardadas."""

import io
import json
import zipfile

import pytest

from curso_ia.evaluation import (EXAMPLE_CASES, answer_metrics, citation_flags, field_metrics, field_status,
                                 hito1_baseline, load_adversarial_cases, load_corpus, load_eval_cases, load_rows,
                                 run_case, same_value, save_rows, understood, unpack_repository,
                                 write_example_repository)


def test_same_value_tolerates_format():
    assert same_value(49990, "$49.990") and same_value("S-103", " s-103") and same_value(2, "2")
    assert same_value("2,5", 2.5) and not same_value(2, 3) and not same_value(None, 0) and same_value(None, None)


def test_field_status_kinds_and_field_map():
    seen = {"member_id": ["S-103"], "months": [2], "solution": ["refund"], "amount": [None]}
    expected = {"member_id": "S-103", "months": None, "requested_solution": "refund", "amount": 100, "product": "x"}
    status = field_status(expected, seen, {"requested_solution": "solution"})
    assert status == {"member_id": "correct", "months": "invented", "requested_solution": "correct",
                      "amount": "omitted", "product": "not_used"}


def test_understood_reads_args_output_and_only_extraction_results():
    row = {"output": {"action": "answered", "order_id": "P-1"}, "record": {"steps": [
        {"kind": "tool", "tool": "get_order", "args": {"order_id": "P-1"}, "result": json.dumps({"product": "db"})},
        {"kind": "tool", "tool": "extract", "args": {"message": "m"}, "result": json.dumps({"email": "a@b.cl"})},
        {"kind": "approval_requested", "tool": "refund", "args": {"order_id": "P-1", "amount": 5}},
    ]}}
    seen = understood(row, extraction_tools=("extract",))
    assert seen["order_id"] == ["P-1"] and seen["amount"] == [5] and seen["email"] == ["a@b.cl"]
    assert "product" not in seen  # el resultado de una consulta no es lo que entendió el servicio


def test_answer_metrics_like_hito1():
    corpus = {"pol": "Los planes mensuales no se congelan."}
    case = {"id": "q", "task": "answer", "expected": {"answerable": True, "evidence": [
        {"doc_id": "pol", "quote": "Los planes mensuales no se congelan."}]}}
    none = {"id": "n", "task": "answer", "expected": {"answerable": False, "evidence": []}}
    answers = {"q": {"citations": [{"doc_id": "pol.md", "quote": "planes mensuales no se congelan"}], "abstained": False}}
    assert citation_flags(case, answers["q"]["citations"][0], corpus) == (True, True, True)
    table = answer_metrics([case, none], answers, corpus, "v").iloc[0]
    assert (table["correct_abstention"], table["answered"], table["with_correct_source"]) == ("1/1", "1/1", "1/1")
    assert field_metrics([{"id": "e", "task": "extract", "expected": {"a": 1}}], {"e": {"a": "correct"}}, "v")["correct"].tolist() == [1]


def test_unpack_repository_strips_github_folder_and_keeps_contract_tests(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("grupo-X-main/src/app.py", "x = 1")
        archive.writestr("grupo-X-main/tests/test_contract.py", "editado")
        archive.writestr("grupo-X-main/eval/cases.jsonl", "")
    copied = unpack_repository(buffer.getvalue(), tmp_path)
    assert sorted(copied) == ["eval/cases.jsonl", "src/app.py"] and (tmp_path / "src/app.py").read_text() == "x = 1"


def test_load_eval_cases_explains_problems(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text('{"id": "a", "task": "answer", "input": "?", "expected": {}}\n{"id": "a"}\nno json\n')
    with pytest.raises(ValueError) as error:
        load_eval_cases(path)
    assert "answerable" in str(error.value) and "faltan" in str(error.value) and "no es JSON" in str(error.value)


def test_example_repository_and_baseline(tmp_path):
    write_example_repository(tmp_path)
    cases = load_eval_cases(tmp_path / "eval/cases.jsonl")
    assert len(cases) == len(EXAMPLE_CASES) and load_corpus(tmp_path / "corpus")["reglamento"].startswith("# Congelamiento")
    assert load_adversarial_cases(tmp_path)[0]["expected_control"] == "tool_refused"
    assert hito1_baseline(tmp_path, cases, {}) is None
    rows = [{"id": "q-01", "status": "ok", "output": {"answer": "No.", "citations": [], "abstained": True}},
            {"id": "e-01", "status": "ok", "output": {"member_id": "S-103", "months": 2}}]
    save_rows(rows, tmp_path / "eval/baseline/outputs.jsonl")
    answers, fields = hito1_baseline(tmp_path, cases, {})
    assert answers.iloc[0]["answered"] == "0/3" and fields.set_index("field").loc["months", "correct"] == 1


async def test_run_case_against_the_template_service(tmp_path, monkeypatch):
    monkeypatch.setenv("CURSO_MODE", "simulated")
    monkeypatch.chdir(tmp_path)  # la plantilla guarda sus trazas en traces/ del directorio actual
    from curso_ia.service import load_app, service_client

    write_example_repository(tmp_path)
    team = load_app(tmp_path / "src/app.py")
    client = service_client(team.app)
    row = await run_case(client, {"id": "e-01", "task": "extract", "goal": "Soy S-103", "scope": "sede_centro"})
    assert row["status"] == "pending_approval" and "approval" in row["controls"] and row["record"]["steps"]
    rejected = await run_case(client, {"id": "x", "task": "adversarial", "goal": "hola", "scope": "otra"})
    assert rejected["status"] == "rejected" and rejected["controls"] == ["rejected_request"]
    save_rows({"e-01": row}, tmp_path / "out.jsonl")
    assert load_rows(tmp_path / "out.jsonl")["e-01"]["run_id"] == row["run_id"]

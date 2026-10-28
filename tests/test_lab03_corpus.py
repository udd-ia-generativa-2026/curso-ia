from curso_ia import lab_resource, load_json


def test_lab03_corpus_and_questions_are_consistent():
    documents = load_json(lab_resource(3, "documents.json"))
    questions = load_json(lab_resource(3, "questions.json"))
    texts = {d["doc_id"]: lab_resource(3, f"documents/{d['file']}").read_text(encoding="utf-8") for d in documents}
    assert len(documents) == 7
    assert [d["doc_id"] for d in documents if d["status"] == "derogada"] == ["pol-devoluciones-2024"]
    assert len(questions) == 24
    assert [q["id"] for q in questions if not q["answerable"]] == ["q08", "q15", "q21"]
    assert sum(q["in_generation_subset"] for q in questions) == 10
    for q in questions:
        assert q["answerable"] == bool(q["expected_evidence"])
        for evidence in q["expected_evidence"]:
            assert evidence["quote"] in texts[evidence["doc_id"]], q["id"]

import pytest

from curso_ia import lab_resource, load_json

# Las pruebas del módulo rag necesitan el extra lab03 (uv sync --extra lab03).
lab03 = pytest.mark.skipif(not __import__("importlib").util.find_spec("llama_index"), reason="falta el extra lab03")


def test_lab03_corpus_and_questions_are_consistent():
    documents = load_json(lab_resource(3, "documents.json"))
    questions = load_json(lab_resource(3, "questions.json"))
    texts = {d["doc_id"]: lab_resource(3, f"documents/{d['file']}").read_text(encoding="utf-8") for d in documents}
    assert len(documents) == 7
    assert [d["doc_id"] for d in documents if d["status"] == "derogada"] == ["pol-devoluciones-2024"]
    assert len(questions) == 26
    assert [q["id"] for q in questions if not q["answerable"]] == ["q08", "q15", "q21"]
    assert sum(q["in_generation_subset"] for q in questions) == 12
    for q in questions:
        assert q["answerable"] == bool(q["expected_evidence"])
        for evidence in q["expected_evidence"]:
            assert evidence["quote"] in texts[evidence["doc_id"]], q["id"]


@lab03
def test_precomputed_embeddings_cover_the_notebook_chunks():
    """Los vectores del paquete cubren las preguntas y los fragmentos que corta el notebook."""
    from llama_index.core.node_parser import SentenceSplitter

    from curso_ia import load_config
    from curso_ia.rag import CourseEmbedding, load_corpus

    config = load_config(env={"CURSO_MODE": "real"})
    embedding = CourseEmbedding("google:gemini-embedding-2", config)
    documents, questions = load_corpus()
    texts = [q["question"] for q in questions]
    for size in (200, 400, 800):
        nodes = SentenceSplitter(chunk_size=size, chunk_overlap=size // 10).get_nodes_from_documents(documents)
        texts += [node.get_content() for node in nodes]
    vectors = embedding.get_text_embedding_batch(texts)
    assert embedding.requests == 0 and len(vectors) == len(texts)
    assert all(abs(sum(x * x for x in v) - 1) < 1e-3 for v in vectors[:5])


@lab03
def test_new_text_without_key_names_the_missing_secret():
    from curso_ia import ConfigError, load_config
    from curso_ia.rag import CourseEmbedding

    embedding = CourseEmbedding("google:gemini-embedding-2", load_config(env={"CURSO_MODE": "real"}))
    with pytest.raises(ConfigError, match="GOOGLE_API_KEY"):
        embedding.get_query_embedding("Una pregunta que no está en el paquete")


@lab03
def test_other_embedding_model_ignores_precomputed_vectors():
    from curso_ia import load_config
    from curso_ia.rag import CourseEmbedding, same_model

    assert same_model("openrouter:google/gemini-embedding-2", "gemini-embedding-2")
    embedding = CourseEmbedding("google:gemini-embedding-001", load_config(env={"CURSO_MODE": "simulated"}))
    assert embedding.precomputed == 0


@lab03
def test_gemini_embeddings_send_one_content_per_text(monkeypatch):
    """gemini-embedding-2 une una lista de textos en un solo vector si no van como contenidos separados."""
    from types import SimpleNamespace

    from google import genai

    from curso_ia import load_config
    from curso_ia.rag import CourseEmbedding

    sent = []

    class FakeModels:
        def embed_content(self, model, contents):
            sent.append(contents)
            return SimpleNamespace(embeddings=[SimpleNamespace(values=[1.0, float(i)]) for i, _ in enumerate(contents)])

    monkeypatch.setattr(genai, "Client", lambda api_key: SimpleNamespace(models=FakeModels()))
    config = load_config(env={"CURSO_MODE": "real", "GOOGLE_API_KEY": "clave-de-prueba"})
    embedding = CourseEmbedding("google:gemini-embedding-2", config, precomputed=False)
    vectors = embedding.get_text_embedding_batch(["uno", "dos", "tres"])
    assert len(sent[0]) == 3 and all(len(content.parts) == 1 for content in sent[0])
    assert len(vectors) == 3 and embedding.requests == 3

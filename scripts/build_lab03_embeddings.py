"""Calcular los vectores del corpus y de las preguntas del laboratorio 3 para incluirlos en el paquete.

Uso docente: `uv run --extra lab03 python scripts/build_lab03_embeddings.py`. El modelo sale de
EMBEDDING_MODEL o de config.toml; `openrouter:google/gemini-embedding-2` sirve como el mismo modelo
que `google:gemini-embedding-2`. Los fragmentos se cortan igual que en el notebook.
"""

from pathlib import Path

from llama_index.core.node_parser import SentenceSplitter

from curso_ia import load_config
from curso_ia.rag import PRECOMPUTED_FILE, CourseEmbedding, load_corpus, save_precomputed

CHUNK_SIZES = (200, 400, 800)
DEST = Path(__file__).resolve().parent.parent / "src/curso_ia/data/lab03" / PRECOMPUTED_FILE


def main() -> None:
    config = load_config()
    embedding = CourseEmbedding(config.models["embedding"], config, precomputed=False)
    documents, questions = load_corpus()
    texts = [question["question"] for question in questions]
    for size in CHUNK_SIZES:
        nodes = SentenceSplitter(chunk_size=size, chunk_overlap=size // 10).get_nodes_from_documents(documents)
        texts += [node.get_content() for node in nodes]
        print(f"{size} tokens: {len(nodes)} fragmentos")
    count = save_precomputed(DEST, embedding, texts)
    print(f"{count} vectores guardados en {DEST.name} ({embedding.requests} textos enviados)")


if __name__ == "__main__":
    main()

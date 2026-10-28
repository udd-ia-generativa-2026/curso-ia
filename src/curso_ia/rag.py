"""Piezas del laboratorio 3 que no enseñan RAG: embeddings con espera y caché, y el juez de DeepEval.

Fragmentar, indexar, recuperar, medir, responder con citas y reordenar quedan a la vista en el notebook.
Este módulo necesita el extra `lab03` (`curso-ia[lab03]`); por eso no se importa desde `curso_ia`.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import deque

import numpy as np
from llama_index.core.base.embeddings.base import BaseEmbedding
from pydantic import PrivateAttr

from .config import Config, ConfigError, missing, model_name, provider
from .files import lab_resource

PRECOMPUTED_FILE = "embeddings.npz"
OPENAI_COMPATIBLE = {"openai": None, "openrouter": "https://openrouter.ai/api/v1"}
# Nivel gratuito de Gemini: cada texto cuenta como una solicitud y el límite es de 100 por minuto
# (prueba del 27/10/2026); se deja margen.
GOOGLE_TEXTS_PER_MINUTE = 90
BATCH = 50


def load_corpus() -> tuple[list, list[dict]]:
    """Documentos de LlamaIndex con sus metadatos, y las preguntas con su evidencia esperada.

    Los metadatos (título, vigencia) quedan en el índice para filtrar y citar, pero no se mezclan
    con el texto que se convierte en vector ni con el que lee el modelo: ahí va solo el contenido.
    """
    from llama_index.core import Document

    from .files import load_json

    documents = []
    for meta in load_json(lab_resource(3, "documents.json")):
        text = lab_resource(3, f"documents/{meta['file']}").read_text(encoding="utf-8")
        metadata = {key: meta[key] for key in ("doc_id", "title", "status", "valid_from", "valid_to")}
        documents.append(Document(
            text=text, metadata=metadata, excluded_embed_metadata_keys=list(metadata),
            excluded_llm_metadata_keys=list(metadata),
        ))
    return documents, load_json(lab_resource(3, "questions.json"))


def text_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def same_model(ref: str, stored: str) -> bool:
    """`google:gemini-embedding-2` y `openrouter:google/gemini-embedding-2` son el mismo modelo."""
    return model_name(ref).split("/")[-1] == stored


class CourseEmbedding(BaseEmbedding):
    """Embeddings de LlamaIndex que calculan cada texto una sola vez.

    Con `precomputed=True` usa los vectores del corpus que trae el paquete, calculados con el mismo
    modelo, y solo consulta al proveedor por los textos nuevos (por ejemplo, preguntas propias).
    `requests` cuenta los textos enviados al proveedor en esta sesión.
    """

    _ref: str = PrivateAttr()
    _config: Config = PrivateAttr()
    _cache: dict = PrivateAttr(default_factory=dict)
    _sent: deque = PrivateAttr(default_factory=deque)
    _requests: int = PrivateAttr(default=0)
    _precomputed: int = PrivateAttr(default=0)
    _warned: bool = PrivateAttr(default=False)

    def __init__(self, ref: str, config: Config, *, precomputed: bool = True):
        super().__init__(model_name=ref, embed_batch_size=BATCH)
        self._ref, self._config = ref, config
        if precomputed:
            self._precomputed = self._load_precomputed()

    @property
    def requests(self) -> int:
        return self._requests

    @property
    def precomputed(self) -> int:
        return self._precomputed

    def _load_precomputed(self) -> int:
        source = lab_resource(3, PRECOMPUTED_FILE)
        if not source.is_file():
            return 0
        with source.open("rb") as file:
            data = np.load(file)
            if not same_model(self._ref, str(data["model"])):
                return 0
            self._cache.update(zip(data["keys"].tolist(), data["vectors"]))
        return len(self._cache)

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        new = [text for text in dict.fromkeys(texts) if text_key(text) not in self._cache]
        for start in range(0, len(new), BATCH):
            batch = new[start:start + BATCH]
            for text, vector in zip(batch, self._fetch(batch)):
                norm = np.linalg.norm(vector)
                self._cache[text_key(text)] = np.asarray(vector, dtype=np.float32) / (norm or 1)
        return [self._cache[text_key(text)].tolist() for text in texts]

    def _fetch(self, texts: list[str]) -> list[list[float]]:
        if self._config.mode == "simulated":
            if not self._warned:
                print("Modo simulado: los textos sin vector precalculado reciben un vector al azar; "
                      "su recuperación no significa nada.")
                self._warned = True
            dim = len(next(iter(self._cache.values()))) if self._cache else 768
            return [np.random.default_rng(int(text_key(t)[:8], 16)).normal(size=dim) for t in texts]
        # La clave solo hace falta para textos sin vector precalculado.
        if lacking := missing(self._ref, self._config):
            raise ConfigError(f"Hay {len(texts)} textos sin vector precalculado; falta configurar: {', '.join(lacking)}.")
        self._wait(len(texts))
        for attempt in range(4):
            try:
                vectors = self._call_provider(texts)
                self._requests += len(texts)
                return vectors
            except Exception as error:
                code = getattr(error, "code", None) or getattr(error, "status_code", None)
                if code != 429 or attempt == 3:
                    raise
                # El nivel gratuito pide esperar hasta un minuto cuando se supera el límite.
                print("Límite de solicitudes por minuto: espero 60 s y reintento.", flush=True)
                time.sleep(60)
        raise AssertionError("inalcanzable")

    def _wait(self, count: int) -> None:
        """Respetar el límite de textos por minuto de Gemini antes de enviar un lote."""
        if provider(self._ref) != "google":
            return
        now = time.monotonic()
        while self._sent and now - self._sent[0] > 60:
            self._sent.popleft()
        if len(self._sent) + count > GOOGLE_TEXTS_PER_MINUTE and self._sent:
            pause = 60 - (now - self._sent[0]) + 1
            print(f"Espero {pause:.0f} s por el límite de Gemini ({GOOGLE_TEXTS_PER_MINUTE} textos por minuto)…",
                  flush=True)
            time.sleep(pause)
            self._sent.clear()
        self._sent.extend([time.monotonic()] * count)

    def _call_provider(self, texts: list[str]) -> list[list[float]]:
        prov, name = provider(self._ref), model_name(self._ref)
        if prov == "google":
            from google import genai

            client = genai.Client(api_key=self._config.key("google"))
            response = client.models.embed_content(model=name, contents=texts)
            return [embedding.values for embedding in response.embeddings]
        if prov in OPENAI_COMPATIBLE:
            from openai import OpenAI

            client = OpenAI(base_url=OPENAI_COMPATIBLE[prov], api_key=self._config.key(prov), max_retries=0)
            response = client.embeddings.create(model=name, input=texts)
            return [item.embedding for item in response.data]
        raise ConfigError(f"El proveedor {prov!r} no ofrece embeddings en el curso; usa google: u openrouter:.")

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embeddings([text])[0]

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._get_text_embeddings([query])[0]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embedding(text)


def save_precomputed(path, embedding: CourseEmbedding, texts: list[str]) -> int:
    """Guardar los vectores de `texts` para incluirlos en el paquete (uso docente)."""
    embedding.get_text_embedding_batch(texts)
    keys = list(dict.fromkeys(text_key(text) for text in texts))
    vectors = np.stack([embedding._cache[key] for key in keys]).astype(np.float32)
    np.savez_compressed(path, keys=np.array(keys), vectors=vectors, model=model_name(embedding._ref).split("/")[-1])
    return len(keys)


def judge_model(ref: str, config: Config, *, per_minute: int | None = None, settings: dict | None = None):
    """Un modelo del curso como juez de DeepEval, con salida tipada de PydanticAI y espera entre consultas."""
    from deepeval.models import DeepEvalBaseLLM
    from pydantic_ai import Agent

    from .models import create_model

    class CourseJudge(DeepEvalBaseLLM):
        def __init__(self):
            self.ref, self.requests, self.input_tokens, self.output_tokens = ref, 0, 0, 0
            self._lock, self._last = asyncio.Lock(), 0.0
            super().__init__(ref)

        def load_model(self):
            return create_model(ref, config)

        async def a_generate(self, prompt: str, schema=None):
            async with self._lock:
                if per_minute:
                    await asyncio.sleep(max(0.0, self._last + 60 / per_minute - time.monotonic()))
                self._last = time.monotonic()
            agent = Agent(self.model, output_type=schema or str, model_settings=settings or {}, retries=2)
            result = await agent.run(prompt)
            usage = result.usage
            self.requests += usage.requests
            self.input_tokens += usage.input_tokens
            self.output_tokens += usage.output_tokens
            return result.output

        def generate(self, prompt: str, schema=None):
            return asyncio.run(self.a_generate(prompt, schema))

        def get_model_name(self):
            return ref

    return CourseJudge()

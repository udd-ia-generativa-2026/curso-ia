"""Laboratorio 4: la ruta del reclamo con un modelo de decisión (d1-3B) y con un modelo de lenguaje.

El notebook deja a la vista el esquema, las instrucciones, el `Agent` y el barrido del umbral. Aquí quedan
el arranque de d1-3B en Colab, la conexión con el servidor del curso y la ejecución medida de los casos.
"""

from __future__ import annotations

import asyncio
import os
import random
import shutil
import subprocess
import tarfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse
from pydantic_ai.usage import UsageLimits

from .config import Config, ConfigError
from .files import lab_resource, load_json

# Versión fija de llama.cpp: d1 necesita b11490 o posterior; esta se probó el 04/11/2026.
LLAMA_BUILD = "b12110"
LLAMA_URL = "https://github.com/ggml-org/llama.cpp/releases/download/{build}/{asset}"
CPU_ASSET = "llama-{build}-bin-ubuntu-x64.tar.gz"
CUDA_ASSETS = ("llama-{build}-bin-ubuntu-cuda-12.8-x64.tar.gz", "cudart-llama-{build}-bin-ubuntu-cuda-12.8-x64.tar.gz")
D1_URL = "https://huggingface.co/LiquidAI/d1-3B-GGUF/resolve/main/d1-3B-{quant}.gguf"
LOCAL_PORT = 8080
TIMEOUT_S = 90
ROUTES = ("rebooking", "refund", "baggage", "special_assistance", "other")


def load_routes() -> dict:
    """Las rutas, las reglas y los casos etiquetados del laboratorio 4."""
    return load_json(lab_resource(4, "cases.json"))


# ---------------------------------------------------------------------------
# d1-3B en la sesión de Colab

def _healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as response:
            return response.status == 200
    except OSError:
        return False


def _has_gpu() -> bool:
    if not shutil.which("nvidia-smi"):
        return False
    return subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0


def _download(url: str, target: Path) -> Path:
    if target.exists():
        return target
    from tqdm.auto import tqdm

    partial = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as file:
        total = int(response.headers.get("Content-Length", 0)) or None
        with tqdm(total=total, unit="B", unit_scale=True, desc=target.name) as bar:
            while chunk := response.read(1 << 20):
                file.write(chunk)
                bar.update(len(chunk))
    partial.rename(target)
    return target


def _llama_server(folder: Path, gpu: bool) -> Path:
    assets = [name.format(build=LLAMA_BUILD) for name in (CUDA_ASSETS if gpu else (CPU_ASSET,))]
    target = folder / ("cuda" if gpu else "cpu")
    target.mkdir(parents=True, exist_ok=True)
    for asset in assets:
        archive = _download(LLAMA_URL.format(build=LLAMA_BUILD, asset=asset), folder / asset)
        with tarfile.open(archive) as tar:
            tar.extractall(target, filter="data")
    found = sorted(target.rglob("llama-server"))
    if not found:
        raise RuntimeError("El paquete de llama.cpp no trae llama-server.")
    found[0].chmod(0o755)
    return found[0]


def _library_path(binary: Path) -> str:
    """Carpetas con bibliotecas compartidas del paquete (en CUDA, cudart viene en otro archivo)."""
    folders = {binary.parent} | {library.parent for library in binary.parents[1].rglob("*.so*")}
    return ":".join([*(str(folder) for folder in sorted(folders)), os.environ.get("LD_LIBRARY_PATH", "")])


def start_d1(quant: str = "Q8_0", port: int = LOCAL_PORT, folder: Path | None = None, gpu: bool | None = None) -> str:
    """Descargar llama.cpp y d1-3B, levantar `llama-server` en esta sesión y devolver su dirección.

    Con GPU usa la compilación CUDA y carga todo el modelo en la GPU; si no arranca, sigue en CPU.
    Si el servidor ya está en marcha, no lo vuelve a levantar.
    """
    url = f"http://127.0.0.1:{port}"
    if _healthy(port):
        print("d1-3B ya estaba en marcha en", url)
        return url
    folder = Path(folder or ("/content/d1" if Path("/content").exists() else Path.home() / ".cache/curso-ia/d1"))
    folder.mkdir(parents=True, exist_ok=True)
    weights = _download(D1_URL.format(quant=quant), folder / f"d1-3B-{quant}.gguf")
    gpu = _has_gpu() if gpu is None else gpu
    for use_gpu in ([True, False] if gpu else [False]):
        binary = _llama_server(folder, use_gpu)
        command = [str(binary), "-m", str(weights), "--host", "127.0.0.1", "--port", str(port), "-c", "8192",
                   "-t", str(os.cpu_count() or 2)] + (["-ngl", "99"] if use_gpu else [])
        env = {**os.environ, "LD_LIBRARY_PATH": _library_path(binary)}
        log = (folder / f"llama-server-{'gpu' if use_gpu else 'cpu'}.log").open("w")
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline and process.poll() is None and not _healthy(port):
            time.sleep(2)
        if _healthy(port):
            print(f"d1-3B ({quant}) en marcha en {url}, con {'GPU' if use_gpu else 'CPU'}.")
            return url
        process.terminate()
        print(f"llama-server no arrancó con {'GPU' if use_gpu else 'CPU'}; revisa {log.name}.")
    raise RuntimeError("No se pudo levantar d1-3B en esta sesión.")


def d1_model(where: str, config: Config):
    """`d1:colab` usa el llama-server de la sesión; `d1:curso`, el servidor del curso con la clave del día."""
    from pydantic_ai.models.system_one import SystemOneModel
    from pydantic_ai.providers.system_one import SystemOneProvider

    from .models import _check_url

    if where == "colab":
        if not _healthy(LOCAL_PORT):
            raise ConfigError("d1-3B no está en marcha en esta sesión: ejecuta start_d1().")
        provider = SystemOneProvider(base_url=f"http://127.0.0.1:{LOCAL_PORT}", api_key="local")
    elif where == "curso":
        base = config.course_base_url.rstrip("/")
        if not base or not config.key("curso"):
            raise ConfigError("Falta configurar: CURSO_BASE_URL, CURSO_API_KEY.")
        _check_url(base)
        # El servidor publica el chat en /v1 y d1-3B en /d1, con la misma clave del día.
        base = base.removesuffix("/v1") + "/d1"
        provider = SystemOneProvider(base_url=base, api_key=config.key("curso"))
    else:
        raise ConfigError("El modelo de decisión es d1:colab o d1:curso.")
    return SystemOneModel("d1", provider=provider)


# ---------------------------------------------------------------------------
# Ejecución medida

def _simulated(case: dict, ref: str) -> tuple[str, float, dict]:
    """Salida inventada y reproducible para practicar sin modelos; no es una medición."""
    rng = random.Random(f"{ref}:{case['id']}")
    right = rng.random() < (0.9 if case.get("difficulty") == "clear" else 0.6)
    route = case["route"] if right else rng.choice([r for r in ROUTES if r != case["route"]])
    top = rng.uniform(0.75, 0.99) if right else rng.uniform(0.4, 0.85)
    rest = (1 - top) / (len(ROUTES) - 1)
    return route, top, {r: (top if r == route else rest) for r in ROUTES}


async def run_routes(
    agent: Agent, cases: list[dict], *, ref: str, config: Config, wait_s: float = 0.0,
) -> list[dict]:
    """Preguntar la ruta de cada caso, uno por uno, y registrar ruta, confianza, latencia, tokens y costo.

    La confianza sale de `provider_details` en un modelo de decisión y del campo `confidence` de la salida
    en un modelo de lenguaje, que la declara él mismo. `wait_s` espacia las consultas para no chocar con el
    límite por minuto del nivel gratuito de Gemini. Un fallo queda como fila con `status = "error"`.
    """
    from tqdm.auto import tqdm

    from .measure import cost, redact

    rows = []
    for index, case in enumerate(tqdm(cases, desc=ref)):
        row = {"id": case["id"], "model": ref, "mode": config.mode, "status": "error", "route": None,
               "confidence": None, "probabilities": None, "latency_ms": None, "input_tokens": None,
               "output_tokens": None, "estimated_cost_usd": None,
               "utc_time": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        if config.mode == "simulated":
            route, confidence, probabilities = _simulated(case, ref)
            row.update(status="ok", route=route, confidence=round(confidence, 4),
                       probabilities=probabilities if ref.startswith("d1:") else None)
            rows.append(row)
            continue
        if index and wait_s:
            await asyncio.sleep(wait_s)
        start = perf_counter()
        try:
            result = await asyncio.wait_for(
                agent.run(case["text"], usage_limits=UsageLimits(request_limit=2)), TIMEOUT_S)
            details = next((m.provider_details for m in reversed(result.all_messages())
                            if isinstance(m, ModelResponse) and m.provider_details), None) or {}
            output = result.output
            confidence = details.get("confidence", {}).get("route", getattr(output, "confidence", None))
            usage = result.usage
            row.update(status="ok", route=output.route,
                       confidence=None if confidence is None else round(float(confidence), 4),
                       probabilities=details.get("probabilities", {}).get("route"),
                       input_tokens=usage.input_tokens, output_tokens=usage.output_tokens, requests=usage.requests,
                       **cost(ref, usage, config))
        except Exception as error:
            # Los mensajes de los SDK pueden incluir claves o cuerpos HTTP: solo tipo y código.
            row["error_type"] = type(error).__name__
            code = getattr(error, "status_code", None)
            if isinstance(code, int):
                row["http_status"] = code
            row["error"] = redact(str(error)[:200], config.secrets())
        row["latency_ms"] = round((perf_counter() - start) * 1000)
        rows.append(row)
    failed = sum(row["status"] != "ok" for row in rows)
    if failed:
        print(f"{failed} de {len(rows)} consultas fallaron; vuelve a ejecutarlas con retry_routes().")
    return rows


async def retry_routes(agent: Agent, rows: list[dict], cases: list[dict], *, ref: str, config: Config,
                       wait_s: float = 10.0) -> list[dict]:
    """Repetir solo las consultas que fallaron, conservando el orden de las filas."""
    by_id = {case["id"]: case for case in cases}
    pending = [row["id"] for row in rows if row["status"] != "ok"]
    redone = {row["id"]: row for row in await run_routes(
        agent, [by_id[i] for i in pending], ref=ref, config=config, wait_s=wait_s)}
    return [redone.get(row["id"], row) for row in rows]


def reliability(table, bins: int = 5):
    """Extensión: aciertos por tramo de confianza y error de calibración esperado (ECE).

    `table` necesita las columnas `confidence` y `correct`. El ECE pondera, por la cantidad de casos de cada
    tramo, la distancia entre la confianza media y la proporción de aciertos.
    """
    import numpy as np
    import pandas as pd

    data = table.dropna(subset=["confidence"])
    edges = np.linspace(0, 1, bins + 1)
    groups = pd.cut(data["confidence"], edges, include_lowest=True)
    summary = data.groupby(groups, observed=True).agg(
        cases=("correct", "size"), mean_confidence=("confidence", "mean"), accuracy=("correct", "mean"))
    ece = float((summary["cases"] / len(data) * (summary["mean_confidence"] - summary["accuracy"]).abs()).sum())
    return summary.reset_index(names="bin"), ece

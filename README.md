# curso-ia · Código común de Aplicaciones de IA Generativa

Paquete del curso: configuración de modelos por rol, casos de prueba, medición de tokens, latencia y costo, y evidencia con revisión humana. Cada laboratorio agrega lo que necesita; antes de cada clase se marca con una etiqueta (`lab01`, `lab02`, …) y los enlaces de cada laboratorio apuntan a la suya, así lo que cambie después en `main` no altera lo que usan los estudiantes.

## Usarlo en Colab

Laboratorio 1: [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/udd-ia-generativa-2026/curso-ia/blob/lab01/notebooks/lab01.ipynb)

Laboratorio 2: [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/udd-ia-generativa-2026/curso-ia/blob/lab02/notebooks/lab02.ipynb)

Laboratorio 3: [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/udd-ia-generativa-2026/curso-ia/blob/lab03/notebooks/lab03.ipynb)

La primera celda de cada notebook instala el paquete:

```
%pip install -q --upgrade "curso-ia @ git+https://github.com/udd-ia-generativa-2026/curso-ia@lab01"
```

Las claves van en **Secretos** de Colab con los nombres `CURSO_API_KEY` y `GOOGLE_API_KEY`; nunca en una celda. Instalarlo no requiere cuenta de GitHub.

## Modelos

Cada rol usa un modelo con la forma `proveedor:modelo`: `curso:` es el servidor del curso y `google:` es Gemini. Los valores por defecto están en `src/curso_ia/data/config.toml`; un repositorio puede traer su propio `config.toml`, y cada rol se cambia con `GENERATION_MODEL` o `COMPARISON_MODEL`. Una clave propia de otro proveedor nunca es obligatoria.

## Desarrollo

Cada cambio publicado sube `version` en `pyproject.toml`: pip no reinstala desde git si la versión no cambia. `uv sync` y `uv run pytest` ejecutan las pruebas sin red ni claves. Con `CURSO_MODE=simulated` los notebooks usan respuestas simuladas: sirven para revisar el circuito, no acreditan acceso a ningún modelo.

Convención: los identificadores (funciones, clases, variables, campos de TOML y JSON, nombres de archivo) van en inglés; comentarios, mensajes y textos de los notebooks, en español.

## Laboratorio 2 · Incidencias de pasajeros en Colab

Acuerdo del 22/10/2026: todos los laboratorios se trabajan en Colab. El notebook `notebooks/lab02.ipynb` incluye instalación, secretos, extracción y descarga; abrirlo no requiere cuenta de GitHub. Deja visibles el esquema, las instrucciones, `Agent` y las métricas. El paquete aporta configuración, lectura, medición y guardado. Publicado en la etiqueta `lab02` el 22/10/2026.

La muestra sintética de una aerolínea ficticia contiene 16 documentos distintos: 12 textos (correos, chats, formularios, notas de llamadas y registros del sistema) y cuatro imágenes realistas. El recorrido inicial usa ocho documentos, seis textos y dos imágenes, y dos versiones de la instrucción: 16 consultas; la muestra completa, 32. Extrae `booking_code`, `requested_action`, `amount_clp`, `flight_date` y `delay_minutes`, cada uno con valor, confianza declarada y evidencia. Las anotaciones siguen la política de la sección 2 del notebook y requieren revisión humana; las salidas preparadas son solo simulación. No es una muestra representativa ni un conjunto retenido del Hito 1.

Abre en modo simulado. Para evidencia real, elige `MODE = "real"` y usa los secretos de Colab. El notebook conserva las dos versiones, los originales y las métricas en `labs/lab02/`, y descarga un ZIP con nombre distinto por iteración. Descarga también el notebook con sus salidas desde el menú de Colab. El ZIP se incorpora después al repositorio del equipo sin necesitar Git en Colab; la reflexión y el historial siguen siendo individuales.

Comprobaciones sin proveedores externos: `uv run pytest` y `uv run python scripts/check_lab02.py`. La segunda ejecuta simulación y HTTP local falso, incluida la imagen, un error deliberado, el recálculo sin API y el ZIP con fuentes. No acredita disponibilidad de Gemini ni sustituye el ensayo en Colab. Jupyter y el servidor local necesitan permiso para abrir puertos locales.

Regeneración docente: `python scripts/generate_lab02.py` escribe los textos y el manifiesto sin consultar modelos. Las cuatro imágenes se generaron una vez con Seedream 5 Pro en 2K desde el repositorio docente; su prompt está en `scripts/lab02-images/` y su referencia se anotó mirando la imagen. La instrucción base es corta a propósito: no repite toda la política de anotación, para que los errores y la iteración tengan algo que corregir.

Antes de publicar, `python scripts/build_lab02_review.py` prepara en `dist/lab02-review/` el notebook y un ZIP de las fuentes para probar en Colab. Se sube ese ZIP al panel Archivos antes de la instalación; la celda lo detecta y evita depender de una etiqueta aún inexistente. No incluye credenciales ni conversaciones.

## Laboratorio 3 · Respuestas con fuentes en Colab

El notebook `notebooks/lab03.ipynb` construye un índice de Qdrant en memoria con LlamaIndex sobre siete políticas sintéticas de Pacífico Sur (una derogada) y mide el recuperador con recall@k, tasa de aciertos y MRR sobre frases literales esperadas en 26 preguntas, más tres preguntas propias de cada equipo. Responde 12 de ellas en tres modos (sin documentos, con todo el corpus y con recuperación) con salida tipada y citas, corrige el generador con un solo cambio (instrucción, validador de citas o filtro de vigencia) y mide si un *reranker* mejora el recuperador antes de agregarlo. Deja visibles el fraccionamiento, el índice, el filtro de vigencia, las métricas, el esquema, la instrucción, `Agent`, el validador y el rerank; `curso_ia.rag` aporta la carga del corpus, las oraciones con identificador para las preguntas propias y los embeddings con espera y caché. Necesita el extra `lab03`. Publicado en la etiqueta `lab03` el 30/10/2026.

Los embeddings usan `gemini-embedding-2` (rol `embedding`, variable `EMBEDDING_MODEL`). El paquete trae en `data/lab03/embeddings.npz` los vectores de las 26 preguntas y de los fragmentos de 200, 400 y 800 tokens, para no gastar en clase la cuota gratuita de 100 textos por minuto; solo los textos nuevos consultan al proveedor. `uv run --extra lab03 python scripts/build_lab03_embeddings.py` los regenera; `openrouter:google/gemini-embedding-2` sirve como el mismo modelo.

El rerank usa `gte-multilingual-reranker-base`, cuyo código propio no funciona con transformers 5: el extra fija `transformers<5` (prueba del 28/10/2026).

Comprobaciones: `uv sync --extra lab03` y `uv run pytest`; el notebook corre completo en modo simulado sin claves. `python scripts/build_lab03_review.py` prepara en `dist/lab03-review/` el notebook y un ZIP del paquete para el ensayo en Colab, como en el laboratorio 2. Con uv, NLTK rechaza los archivos instalados con enlaces duros: usar `UV_LINK_MODE=copy`.

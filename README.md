# curso-ia · Código común de Aplicaciones de IA Generativa

Paquete del curso: configuración de modelos por rol, casos de prueba, medición de tokens, latencia y costo, y evidencia con revisión humana. Cada laboratorio agrega lo que necesita; antes de cada clase se marca con una etiqueta (`lab01`, `lab02`, …) y los enlaces de cada laboratorio apuntan a la suya, así lo que cambie después en `main` no altera lo que usan los estudiantes.

## Usarlo en Colab

Laboratorio 1: [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/udd-ia-generativa-2026/curso-ia/blob/lab01/notebooks/lab01.ipynb)

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

Acuerdo del 22/10/2026: todos los laboratorios se trabajan en Colab. El notebook `notebooks/lab02.ipynb` incluye instalación, secretos, extracción y descarga; abrirlo no requiere cuenta de GitHub. Deja visibles el esquema, las instrucciones, `Agent` y las métricas. El paquete aporta configuración, lectura, medición y guardado. La etiqueta pública `lab02` y su enlace de Colab siguen pendientes de revisión y publicación; no usar ese enlace antes de publicar la etiqueta.

La muestra sintética de una aerolínea ficticia contiene 50 registros (48 textos y dos imágenes pareadas). El recorrido inicial usa ocho documentos y dos versiones de la instrucción: 16 consultas. Extrae `booking_code`, `requested_action`, `amount_clp`, `flight_date` y `delay_minutes`, cada uno con valor, confianza declarada y evidencia. Las anotaciones requieren revisión humana; las salidas preparadas son solo simulación. Las variaciones de doce patrones no son una muestra representativa ni un conjunto retenido del Hito 1.

Abre en modo simulado. Para evidencia real, elige `MODE = "real"` y usa los secretos de Colab. El notebook conserva las dos versiones, los originales y las métricas en `labs/lab02/`, y descarga un ZIP con nombre distinto por iteración. Descarga también el notebook con sus salidas desde el menú de Colab. El ZIP se incorpora después al repositorio del equipo sin necesitar Git en Colab; la reflexión y el historial siguen siendo individuales.

Comprobaciones sin proveedores externos: `uv run pytest` y `uv run python scripts/check_lab02.py`. La segunda ejecuta simulación y HTTP local falso, incluida la imagen, un error deliberado, el recálculo sin API y el ZIP con fuentes. No acredita disponibilidad de Gemini ni sustituye el ensayo en Colab. Jupyter y el servidor local necesitan permiso para abrir puertos locales.

Regeneración docente: `python scripts/generate_lab02.py`, con `rsvg-convert` e ImageMagick para los PNG. Los estudiantes no necesitan esas herramientas. La generación no usa modelos ni datos reales. La política de anotación completa está en la instrucción visible del notebook.

Antes de publicar, `python scripts/build_lab02_review.py` prepara en `dist/lab02-review/` el notebook y un ZIP de las fuentes para probar en Colab. Se sube ese ZIP al panel Archivos antes de la instalación; la celda lo detecta y evita depender de una etiqueta aún inexistente. No incluye credenciales ni conversaciones.

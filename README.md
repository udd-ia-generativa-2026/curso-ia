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

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

## Laboratorio 2 · Base en preparación

El notebook `notebooks/lab02.ipynb` deja visibles el esquema tipado, las instrucciones con y sin ejemplos, `Agent.run` y el recálculo por campo. El paquete aporta configuración, lectura de texto o documentos binarios y guardado de evidencia sin credenciales. Los ejemplos incorporados son didácticos; el dominio común y la muestra definitiva siguen pendientes de la votación. Esta versión todavía no tiene una etiqueta pública `lab02` ni un enlace de Colab publicado.

Para probar esta copia local, ejecutar `uv sync --locked` y abrir el notebook con el Python del entorno. Arranca en modo `simulated`, con seis textos y una imagen del mismo contenido que uno de ellos, sin llamadas externas. Conserva ambas versiones de la instrucción y sus salidas en `labs/lab02/evidence/experiment.json`; la referencia revisada y las explicaciones quedan pendientes hasta que las complete una persona. El modo real usa el rol `generation` y las claves del entorno; la admisión de imágenes debe comprobarse con el modelo configurado.

Comprobaciones sin proveedores externos: `uv run pytest` y `uv run python scripts/check_lab02.py`. La segunda ejecuta el notebook completo en modo simulado y contra un servidor HTTP local falso, incluidos imágenes, una consulta fallida, redacción de claves y recálculo sin API. Una prueba de transporte local no acredita el nivel gratuito de Gemini ni reemplaza el ensayo en Codespaces y Colab. El servidor local y el kernel de Jupyter necesitan permiso para abrir puertos de esta máquina.

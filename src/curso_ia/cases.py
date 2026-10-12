"""Casos de prueba de una tarea: entrada y respuesta esperada.

El caso común de cada laboratorio viene en el paquete; los casos propios se escriben como un
diccionario en una celda del notebook.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Mapping

MAX_CASES = 10


class CasesError(ValueError):
    pass


@dataclass(frozen=True)
class Case:
    id: str
    input: str
    expected: str


@dataclass(frozen=True)
class Cases:
    name: str  # nombre corto del conjunto; también nombra el archivo de evidencia
    instruction: str
    cases: tuple[Case, ...]

    @property
    def is_example(self) -> bool:
        return all(case.id.startswith("ejemplo") for case in self.cases)


def parse_cases(data: Mapping, name: str = "my-task") -> Cases:
    """Validar un diccionario con `instruction` y una lista `cases` de {id, input, expected}."""
    instruction = data.get("instruction", "")
    if not isinstance(instruction, str) or not instruction.strip():
        raise CasesError("Falta la instrucción común a todos los casos (instruction).")
    items = data.get("cases", [])
    if not isinstance(items, list) or not 1 <= len(items) <= MAX_CASES:
        raise CasesError(f"Escribe entre 1 y {MAX_CASES} casos en la lista cases.")

    cases, seen = [], set()
    for number, item in enumerate(items, start=1):
        fields = {key: item.get(key) if isinstance(item, Mapping) else None for key in ("id", "input", "expected")}
        empty = [key for key, value in fields.items() if not isinstance(value, str) or not value.strip()]
        if empty:
            raise CasesError(f"El caso {number} no tiene {', '.join(empty)}.")
        if fields["id"] in seen:
            raise CasesError(f"El id {fields['id']!r} está repetido.")
        seen.add(fields["id"])
        cases.append(Case(fields["id"].strip(), fields["input"].strip(), fields["expected"].strip()))
    return Cases(name, instruction.strip(), tuple(cases))


def load_cases(path: Path) -> Cases:
    """Leer casos desde un archivo TOML con la misma forma; el nombre sale del archivo."""
    try:
        with open(path, "rb") as file:
            data = tomllib.load(file)
    except tomllib.TOMLDecodeError as error:
        raise CasesError(f"{Path(path).name} no es un TOML válido: {error}") from None
    return parse_cases(data, Path(path).stem)


def lab_cases(number: int, name: str = "cases") -> Cases:
    """Casos que el paquete trae para un laboratorio, por ejemplo el caso común del laboratorio 1."""
    source = files("curso_ia") / "data" / f"lab{number:02d}" / f"{name}.toml"
    if not source.is_file():
        raise CasesError(f"El paquete no trae los casos {name!r} del laboratorio {number}.")
    with source.open("rb") as file:
        return parse_cases(tomllib.load(file), name)

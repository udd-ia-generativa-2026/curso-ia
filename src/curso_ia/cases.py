"""Casos de prueba de una tarea: entrada y respuesta esperada, en un archivo TOML editable."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

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
    instruction: str
    cases: tuple[Case, ...]
    path: Path

    @property
    def is_example(self) -> bool:
        return all(case.id.startswith("ejemplo") for case in self.cases)


def load_cases(path: Path) -> Cases:
    try:
        with open(path, "rb") as file:
            data = tomllib.load(file)
    except tomllib.TOMLDecodeError as error:
        raise CasesError(f"{path.name} no es un TOML válido: {error}") from None

    instruction = data.get("instruction", "")
    if not isinstance(instruction, str) or not instruction.strip():
        raise CasesError("Falta la instrucción común a todos los casos (instruction).")
    items = data.get("cases", [])
    if not isinstance(items, list) or not 1 <= len(items) <= MAX_CASES:
        raise CasesError(f"Escribe entre 1 y {MAX_CASES} casos con [[cases]].")

    cases, seen = [], set()
    for number, item in enumerate(items, start=1):
        fields = {name: item.get(name) if isinstance(item, dict) else None for name in ("id", "input", "expected")}
        empty = [name for name, value in fields.items() if not isinstance(value, str) or not value.strip()]
        if empty:
            raise CasesError(f"El caso {number} no tiene {', '.join(empty)}.")
        if fields["id"] in seen:
            raise CasesError(f"El id {fields['id']!r} está repetido.")
        seen.add(fields["id"])
        cases.append(Case(fields["id"].strip(), fields["input"].strip(), fields["expected"].strip()))
    return Cases(instruction.strip(), tuple(cases), path)

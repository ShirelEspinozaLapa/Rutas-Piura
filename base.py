"""Definición base y contrato para los Solvers de Ruteo (Patrón Estrategia)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ResultadoSolver:
    nombre_solver: str
    es_cuantico: bool
    qubits_usados: int
    orden_encontrado: list[int]
    energia_obtenida: float
    gap_pct: float
    tiempo_s: float
    p_optimo: float = 0.0
    p_factible: float = 1.0
    tts: int = 1
    detalles_extra: dict[str, Any] = field(default_factory=dict)


class BaseRoutingSolver(ABC):
    """Clase base abstracta que todo solver (clásico o cuántico) debe implementar."""

    def __init__(self, nombre: str, descripcion: str, es_cuantico: bool):
        self.nombre = nombre
        self.descripcion = descripcion
        self.es_cuantico = es_cuantico

    @abstractmethod
    def resolver(self, sp, prep, cfg=None, **kwargs) -> ResultadoSolver:
        """Resuelve el subproblema y devuelve un ResultadoSolver estandarizado."""
        pass

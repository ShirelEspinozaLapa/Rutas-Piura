"""Registro dinámico de Solvers (Patrón Factory/Registry)."""
from __future__ import annotations

from typing import Type
from .base import BaseRoutingSolver
from .solver_clasicos import SolverFuerzaBruta, SolverSimulatedAnnealing
from .solver_qaoa import SolverQAOA
from .solver_warmstart import SolverWarmStartQAOA
from .solver_ring_xy import SolverRingXY, SolverDWave


_REGISTRO_SOLVERS: dict[str, BaseRoutingSolver] = {
    "fuerza_bruta": SolverFuerzaBruta(),
    "sa": SolverSimulatedAnnealing(),
    "qaoa_estandar": SolverQAOA("estandar"),
    "qaoa_xy": SolverQAOA("xy"),
    "qaoa_warmstart": SolverWarmStartQAOA(),
    "qaoa_ring_xy": SolverRingXY(),
    "dwave": SolverDWave(),
}


def registrar_solver(nombre: str, solver: BaseRoutingSolver):
    """Registra un nuevo plugin de solver."""
    _REGISTRO_SOLVERS[nombre] = solver


def listar_solvers() -> list[dict[str, str | bool]]:
    """Devuelve la lista de solvers disponibles con su metadata."""
    return [
        {
            "id": k,
            "nombre": s.nombre,
            "descripcion": s.descripcion,
            "es_cuantico": s.es_cuantico
        }
        for k, s in _REGISTRO_SOLVERS.items()
    ]


def obtener_solver(nombre: str) -> BaseRoutingSolver:
    """Obtiene una instancia del solver por su identificador."""
    if nombre not in _REGISTRO_SOLVERS:
        disponibles = ", ".join(_REGISTRO_SOLVERS.keys())
        raise KeyError(f"Solver '{nombre}' no registrado. Disponibles: {disponibles}")
    return _REGISTRO_SOLVERS[nombre]

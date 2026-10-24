"""Módulo de Solvers para Q-Rutas Piura."""
from .base import BaseRoutingSolver, ResultadoSolver
from .registry import listar_solvers, obtener_solver, registrar_solver

__all__ = [
    "BaseRoutingSolver",
    "ResultadoSolver",
    "listar_solvers",
    "obtener_solver",
    "registrar_solver",
]

"""Solvers clásicos bajo la interfaz BaseRoutingSolver."""
from __future__ import annotations

import numpy as np
from .. import clasicos, qubo as qb
from .base import BaseRoutingSolver, ResultadoSolver


class SolverFuerzaBruta(BaseRoutingSolver):
    def __init__(self):
        super().__init__("fuerza_bruta", "Fuerza Bruta exacta por permutaciones", False)

    def resolver(self, sp, prep, cfg=None, **kwargs) -> ResultadoSolver:
        fb = clasicos.fuerza_bruta(sp)
        return ResultadoSolver(
            nombre_solver=self.nombre,
            es_cuantico=False,
            qubits_usados=0,
            orden_encontrado=fb["orden"],
            energia_obtenida=fb["E"],
            gap_pct=0.0,
            tiempo_s=fb["tiempo_s"],
            p_optimo=1.0,
            p_factible=1.0,
            tts=1,
            detalles_extra={"n_evaluaciones": fb["n_evaluaciones"]}
        )


class SolverSimulatedAnnealing(BaseRoutingSolver):
    def __init__(self):
        super().__init__("sa_qubo", "Simulated Annealing sobre QUBO", False)

    def resolver(self, sp, prep, cfg=None, **kwargs) -> ResultadoSolver:
        rng = kwargs.get("rng") or np.random.default_rng(2026)
        res = clasicos.sa_qubo(sp, prep, cfg, rng)
        gap = 100.0 * (res["E"] - prep.E_opt) / max(1e-6, abs(prep.E_opt))
        return ResultadoSolver(
            nombre_solver=self.nombre,
            es_cuantico=False,
            qubits_usados=sp.nq,
            orden_encontrado=res["orden"],
            energia_obtenida=res["E"],
            gap_pct=float(gap),
            tiempo_s=res["tiempo_s"],
            p_optimo=res["p_optimo"],
            p_factible=res["p_factible"],
            tts=clasicos._tts(res["p_optimo"]),
            detalles_extra={"reparado": res["reparado"]}
        )

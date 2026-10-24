"""Plugin de Ring XY Mixer y D-Wave Annealing."""
from __future__ import annotations

import numpy as np
from .. import clasicos
from ..cuantico_ring_xy import ejecutar_ring_xy
from ..dwave_solver import resolver_quantum_annealing
from .base import BaseRoutingSolver, ResultadoSolver


class SolverRingXY(BaseRoutingSolver):
    def __init__(self, p: int = 1):
        super().__init__("qaoa_ring_xy", "QAOA con Mezclador XY en Anillo (Ring Topology)", True)
        self.p = p

    def resolver(self, sp, prep, cfg=None, **kwargs) -> ResultadoSolver:
        rng = kwargs.get("rng") or np.random.default_rng(2026)
        res = ejecutar_ring_xy(sp, prep, p=self.p, rng=rng)

        gap = 100.0 * (res.E_mejor - prep.E_opt) / max(1e-6, abs(prep.E_opt))

        return ResultadoSolver(
            nombre_solver=self.nombre,
            es_cuantico=True,
            qubits_usados=sp.nq,
            orden_encontrado=res.orden_mejor,
            energia_obtenida=res.E_mejor,
            gap_pct=float(gap),
            tiempo_s=res.tiempo_s,
            p_optimo=res.p_optimo,
            p_factible=res.p_factible,
            tts=clasicos._tts(res.p_optimo),
            detalles_extra={
                "cnot_estimadas": res.cnot_estimadas,
                "cnot_all_to_all": res.cnot_all_to_all,
                "reduccion_cnot_pct": res.reduccion_cnot_pct
            }
        )


class SolverDWave(BaseRoutingSolver):
    def __init__(self, num_reads: int = 500):
        super().__init__("dwave_annealing", "D-Wave Quantum Annealing BQM", True)
        self.num_reads = num_reads

    def resolver(self, sp, prep, cfg=None, **kwargs) -> ResultadoSolver:
        res = resolver_quantum_annealing(sp, prep, num_reads=self.num_reads)

        return ResultadoSolver(
            nombre_solver=self.nombre,
            es_cuantico=True,
            qubits_usados=res["qubits_usados"],
            orden_encontrado=res["orden_mejor"],
            energia_obtenida=res["E_mejor"],
            gap_pct=res["gap_pct"],
            tiempo_s=res["tiempo_s"],
            p_optimo=res["p_optimo"],
            p_factible=res["p_factible"],
            tts=res["tts_lecturas"],
            detalles_extra={
                "ejecutado_en_qpu_real": res["ejecutado_en_qpu_real"],
                "num_reads": res["num_reads"]
            }
        )

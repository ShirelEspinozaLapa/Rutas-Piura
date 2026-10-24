"""Plugin de Warm-Start QAOA."""
from __future__ import annotations

import numpy as np
from .. import clasicos
from ..cuantico_warmstart import ejecutar_warm_start_qaoa
from .base import BaseRoutingSolver, ResultadoSolver


class SolverWarmStartQAOA(BaseRoutingSolver):
    def __init__(self, p: int = 1):
        super().__init__("qaoa_warmstart", "Warm-Start QAOA con inicialización continua", True)
        self.p = p

    def resolver(self, sp, prep, cfg=None, **kwargs) -> ResultadoSolver:
        rng = kwargs.get("rng") or np.random.default_rng(2026)
        res = ejecutar_warm_start_qaoa(sp, prep, p=self.p, rng=rng)

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
                "thetas": res.thetas.tolist(),
                "c_continuo": res.c_continuo.tolist(),
                "n_evaluaciones": res.n_evaluaciones
            }
        )

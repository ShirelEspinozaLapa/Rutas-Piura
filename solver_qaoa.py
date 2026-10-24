"""Solvers QAOA Estándar y QAOA-XY como plugins."""
from __future__ import annotations

import numpy as np
from .. import cuantico, clasicos, qubo as qb
from .base import BaseRoutingSolver, ResultadoSolver


class SolverQAOA(BaseRoutingSolver):
    def __init__(self, variante: str = "estandar"):
        desc = "QAOA Estándar (Farhi et al.)" if variante == "estandar" else "QAOA-XY (Hadfield et al.)"
        super().__init__(f"qaoa_{variante}", desc, True)
        self.variante = variante

    def resolver(self, sp, prep, cfg=None, **kwargs) -> ResultadoSolver:
        rng = kwargs.get("rng") or np.random.default_rng(2026)
        motor = kwargs.get("motor", "auto")
        res, muestras, ir = cuantico.ejecutar_qaoa(sp, prep, cfg, self.variante, rng, motor)

        P = res.probs
        p_fac = float(P[prep.factible].sum())
        p_opt = float(P[prep.optimos].sum())

        # Ruta de mejor energía entre muestras
        unicos, _ = np.unique(muestras, return_counts=True)
        fac = prep.factible[unicos]
        if fac.any():
            cand = unicos[fac]
            k = cand[np.argmin(prep.E_total[cand])]
            orden = qb.x_a_perm(qb.indice_a_x(int(k), sp.nq), sp.n)
        else:
            orden = list(sp.info.get("orden_vecino_mas_cercano", list(range(sp.n))))

        E_best = qb.energia(sp.Q, sp.const, qb.perm_a_x(orden, sp.n))
        gap = 100.0 * (E_best - prep.E_opt) / max(1e-6, abs(prep.E_opt))

        return ResultadoSolver(
            nombre_solver=self.nombre,
            es_cuantico=True,
            qubits_usados=sp.nq,
            orden_encontrado=orden,
            energia_obtenida=float(E_best),
            gap_pct=float(gap),
            tiempo_s=res.tiempo_s,
            p_optimo=p_opt,
            p_factible=p_fac,
            tts=clasicos._tts(p_opt),
            detalles_extra={
                "p": res.p,
                "n_evaluaciones": res.n_evaluaciones,
                "gammas": res.gammas.tolist(),
                "betas": res.betas.tolist()
            }
        )

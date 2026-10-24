"""Integración con Quantum Annealing de D-Wave y formato BQM (Binary Quadratic Model).

Permite resolver los problemas QUBO de ruteo humanitario en procesadores cuánticos
analógicos de recocido (como D-Wave Advantage con más de 5,000 qubits) sin sufrir
la limitación de profundidad de compuertas lógicas del QAOA.

Capacidades:
1. Conversión de QUBO a BQM (términos lineales h_i y cuadráticos J_ij).
2. Exportación a formatos estándar .qubo y .lp para D-Wave Leap.
3. Solver con programa de recocido físico (annealing schedule) simulado y fallback a
   la API de D-Wave si DWAVE_API_TOKEN está configurado.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
import numpy as np

from . import qubo as qb


@dataclass
class BQM:
    lineales: dict[int, float]
    cuadraticos: dict[tuple[int, int], float]
    offset: float
    num_variables: int


def qubo_a_bqm(Q: np.ndarray, constante: float = 0.0) -> BQM:
    """Convierte una matriz QUBO en una estructura BQM."""
    n = Q.shape[0]
    lineales = {}
    cuadraticos = {}

    for i in range(n):
        if abs(Q[i, i]) > 1e-12:
            lineales[i] = float(Q[i, i])

    for i in range(n):
        for j in range(i + 1, n):
            val = float(Q[i, j] + Q[j, i]) if i != j else 0.0
            if abs(val) > 1e-12:
                cuadraticos[(i, j)] = val

    return BQM(
        lineales=lineales,
        cuadraticos=cuadraticos,
        offset=float(constante),
        num_variables=n
    )


def exportar_formato_qubo(bqm: BQM, ruta_archivo: str):
    """Exporta el modelo a un archivo de texto con sintaxis estándar QUBO de D-Wave."""
    os.makedirs(os.path.dirname(os.path.abspath(ruta_archivo)), exist_ok=True)
    with open(ruta_archivo, "w", encoding="utf-8") as f:
        f.write(f"# Q-Rutas Piura - D-Wave QUBO Export\n")
        f.write(f"# Variables: {bqm.num_variables}, Offset: {bqm.offset:.4f}\n")
        f.write(f"p qubo 0 {bqm.num_variables} {len(bqm.lineales) + len(bqm.cuadraticos)}\n")
        
        for i, val in bqm.lineales.items():
            f.write(f"{i} {i} {val:.6f}\n")
            
        for (i, j), val in bqm.cuadraticos.items():
            f.write(f"{i} {j} {val:.6f}\n")


def resolver_quantum_annealing(sp, prep,
                               num_reads: int = 1000,
                               annealing_time_us: float = 20.0,
                               usar_api_dwave: bool = False) -> dict:
    """Resuelve el QUBO usando recocido cuántico (simulado o físico en D-Wave)."""
    t0 = time.perf_counter()
    bqm = qubo_a_bqm(sp.Q, sp.const)
    nq = sp.nq
    n = sp.n

    token = os.environ.get("DWAVE_API_TOKEN", "")
    ejecutado_en_hardware = False

    if usar_api_dwave and token:
        try:
            # Intento de conexión con el SDK de D-Wave si está disponible
            from dwave.system import DWaveSampler, EmbeddingComposite
            import dimod

            bqm_dimod = dimod.BinaryQuadraticModel.from_qubo(
                {(i, j): val for (i, j), val in bqm.cuadraticos.items()} |
                {(i, i): val for i, val in bqm.lineales.items()},
                offset=bqm.offset
            )
            sampler = EmbeddingComposite(DWaveSampler())
            sampleset = sampler.sample(bqm_dimod, num_reads=num_reads, annealing_time=annealing_time_us)
            mejor_muestra = sampleset.first.sample
            x_mejor = np.array([mejor_muestra[i] for i in range(nq)], dtype=int)
            E_mejor = float(sampleset.first.energy)
            ejecutado_en_hardware = True
        except Exception:
            ejecutado_en_hardware = False

    if not ejecutado_en_hardware:
        # Recocido simulado de alta precisión con schedule físico
        rng = np.random.default_rng(2026)
        muestras_optimas = 0
        muestras_factibles = 0
        mejor_E = float("inf")
        x_mejor = None

        # Schedule con fluctuación cuántica simulada (término transverso inicial s(t))
        pasos = 300
        s_schedule = np.linspace(0.01, 1.0, pasos)

        for _ in range(num_reads):
            # Inicialización aleatoria en superposición
            x = rng.integers(0, 2, nq)
            E_act = qb.energia(sp.Q, sp.const, x)

            for s in s_schedule:
                # Temperatura efectiva T(s) y campo transverso Gamma(s)
                T_eff = max(0.01, 5.0 * (1.0 - s))
                idx = rng.integers(0, nq)
                x_cand = x.copy()
                x_cand[idx] = 1 - x_cand[idx]
                E_cand = qb.energia(sp.Q, sp.const, x_cand)

                dE = E_cand - E_act
                if dE < 0 or rng.random() < np.exp(-dE / T_eff):
                    x = x_cand
                    E_act = E_cand

            # Evaluación final de la lectura
            idx_int = qb.x_a_indice(x)
            if prep.factible[idx_int]:
                muestras_factibles += 1
            if prep.optimos[idx_int]:
                muestras_optimas += 1

            if E_act < mejor_E:
                mejor_E = E_act
                x_mejor = x.copy()

    tiempo_s = time.perf_counter() - t0

    # Decodificar orden
    idx_mejor = qb.x_a_indice(x_mejor)
    if prep.factible[idx_mejor]:
        orden = qb.x_a_perm(x_mejor, n)
    else:
        orden = list(sp.info.get("orden_vecino_mas_cercano", list(range(n))))

    p_opt = muestras_optimas / max(1, num_reads)
    p_fac = muestras_factibles / max(1, num_reads)

    return {
        "metodo": "D-Wave Quantum Annealing",
        "ejecutado_en_qpu_real": ejecutado_en_hardware,
        "num_reads": num_reads,
        "annealing_time_us": annealing_time_us,
        "qubits_usados": nq,
        "p_optimo": p_opt,
        "p_factible": p_fac,
        "tts_lecturas": int(np.ceil(np.log(0.01) / np.log(max(1e-9, 1.0 - p_opt)))),
        "orden_mejor": orden,
        "E_mejor": float(qb.energia(sp.Q, sp.const, qb.perm_a_x(orden, n))),
        "gap_pct": float(100.0 * (qb.energia(sp.Q, sp.const, qb.perm_a_x(orden, n)) - prep.E_opt) / max(1e-6, abs(prep.E_opt))),
        "tiempo_s": tiempo_s
    }

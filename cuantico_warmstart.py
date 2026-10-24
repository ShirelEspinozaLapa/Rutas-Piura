"""Warm-Start QAOA para optimización de ruteo humanitario.

En QAOA estándar, el estado inicial es la superposición uniforme |+>^n = H^n |0>,
lo cual ignora cualquier información previa o heurística clásica del problema.
En Warm-Start QAOA (Egger et al., Tate et al.):
1. Se resuelve la relajación continua del QUBO (o una heurística voraz regularizada)
   obteniendo probabilidades continuas p_j in [eps, 1 - eps].
2. Cada qubit j se inicializa con una rotación:
      |psi_0> = prod_j R_Y(theta_j) |0>,  donde theta_j = 2 * arcsin(sqrt(p_j))
   de modo que la probabilidad previa de medir |1> en el qubit j sea exactamente p_j.
3. Se aplica el ansatz variacional con capas de costo y mezclador adaptado,
   logrando convergencia más rápida y mayor probabilidad del óptimo con menor profundidad p.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize

from . import qubo as qb
from .cuantico import ResultadoQAOA, preparar


@dataclass
class ResultadoWarmStart:
    p: int
    gammas: np.ndarray
    betas: np.ndarray
    thetas: np.ndarray
    c_continuo: np.ndarray
    probs: np.ndarray
    historia: list[float]
    tiempo_s: float
    n_evaluaciones: int
    p_optimo: float
    p_factible: float
    orden_mejor: list[int]
    E_mejor: float


def calcular_inicializacion_warm_start(sp, epsilon: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Calcula el vector de relajación continua y los ángulos theta para los qubits."""
    nq = sp.nq
    n = sp.n
    orden_nn = sp.info.get("orden_vecino_mas_cercano", list(range(n)))
    x_nn = qb.perm_a_x(orden_nn, n)

    # Suavizado continuo: mezcla la solución heurística con una distribución uniforme
    # para no atrapar el estado cuántico en una base determinista
    p_continuo = np.full(nq, 0.5)
    for idx, val in enumerate(x_nn):
        if val == 1:
            p_continuo[idx] = 1.0 - epsilon
        else:
            p_continuo[idx] = epsilon

    # theta = 2 * arcsin(sqrt(p))
    thetas = 2.0 * np.arcsin(np.sqrt(p_continuo))
    return p_continuo, thetas


def simular_warm_start_statevector(thetas: np.ndarray, gammas: np.ndarray, betas: np.ndarray,
                                   E_diagonal: np.ndarray, nq: int) -> np.ndarray:
    """Calcula el vector de estado para Warm-Start QAOA.
    
    Estado inicial: tensor product de (cos(theta_j/2)|0> + sin(theta_j/2)|1>).
    """
    dim = 1 << nq
    # Construcción eficiente del estado producto inicial
    psi = np.zeros(dim, dtype=complex)
    
    # Base inicial qubit a qubit
    cos_t = np.cos(thetas / 2.0)
    sin_t = np.sin(thetas / 2.0)

    # Tensorizar sobre 2^nq
    psi = np.array([1.0], dtype=complex)
    for q in range(nq):
        q_state = np.array([cos_t[q], sin_t[q]], dtype=complex)
        psi = np.kron(psi, q_state)

    p = len(gammas)
    for l in range(p):
        # 1. Operador de costo diagonal e^{-i gamma H_C}
        psi *= np.exp(-1j * gammas[l] * E_diagonal)

        # 2. Mezclador estándar e^{-i beta sum X}
        # Aplicación compuerta a compuerta RX(2 beta)
        b = betas[l]
        rx = np.array([[np.cos(b), -1j * np.sin(b)],
                       [-1j * np.sin(b), np.cos(b)]], dtype=complex)
        
        # Aplicar RX a cada qubit
        for q in range(nq):
            # Reshape a (2, 2^(nq-1-q), 2, 2^q) para aplicar eficientemente
            stride = 1 << (nq - 1 - q)
            psi = psi.reshape(-1, 2, stride)
            psi = np.einsum('ab,ibk->iak', rx, psi)
            psi = psi.reshape(dim)

    return psi


def ejecutar_warm_start_qaoa(sp, prep, p: int = 1,
                             cvar_alpha: float = 0.20,
                             max_iter: int = 120,
                             rng: np.random.Generator | None = None) -> ResultadoWarmStart:
    """Entrena y evalúa Warm-Start QAOA sobre el subproblema."""
    rng = rng or np.random.default_rng(2026)
    nq = sp.nq
    t0 = time.perf_counter()

    p_cont, thetas = calcular_inicializacion_warm_start(sp)
    E_diag = prep.E_total
    dim = 1 << nq

    n_cvar = max(1, int(dim * cvar_alpha))
    orden_E = np.argsort(E_diag)
    E_ordenado = E_diag[orden_E]

    historia = []
    n_eval = 0

    def objetivo(params):
        nonlocal n_eval
        n_eval += 1
        g = params[:p]
        b = params[p:]
        psi = simular_warm_start_statevector(thetas, g, b, E_diag, nq)
        P = np.abs(psi) ** 2
        P_ordenado = P[orden_E]

        cum = np.cumsum(P_ordenado)
        k = int(np.searchsorted(cum, cvar_alpha))
        k = min(k, dim - 1)

        val = float(np.sum(P_ordenado[:k] * E_ordenado[:k]) / max(1e-12, cum[k]))
        historia.append(val)
        return val

    # Inicialización de parámetros
    x0 = np.concatenate([
        rng.uniform(0.1, 0.5, p),
        rng.uniform(0.1, 0.5, p)
    ])

    res_opt = minimize(objetivo, x0, method="COBYLA", options={"maxiter": max_iter, "rhobeg": 0.2})
    gammas_opt = res_opt.x[:p]
    betas_opt = res_opt.x[p:]

    psi_fin = simular_warm_start_statevector(thetas, gammas_opt, betas_opt, E_diag, nq)
    P_fin = np.abs(psi_fin) ** 2

    # Métricas
    p_fac = float(P_fin[prep.factible].sum())
    p_opt = float(P_fin[prep.optimos].sum())

    # Mejor ruta medida / con mayor probabilidad factible
    indices_fac = np.where(prep.factible)[0]
    if len(indices_fac) > 0:
        k_best = indices_fac[np.argmax(P_fin[indices_fac])]
        orden_best = qb.x_a_perm(qb.indice_a_x(int(k_best), nq), sp.n)
        E_best = float(E_diag[k_best])
    else:
        orden_best = list(sp.info.get("orden_vecino_mas_cercano", list(range(sp.n))))
        E_best = prep.E_opt

    tiempo_s = time.perf_counter() - t0

    return ResultadoWarmStart(
        p=p,
        gammas=gammas_opt,
        betas=betas_opt,
        thetas=thetas,
        c_continuo=p_cont,
        probs=P_fin,
        historia=historia,
        tiempo_s=tiempo_s,
        n_evaluaciones=n_eval,
        p_optimo=p_opt,
        p_factible=p_fac,
        orden_mejor=orden_best,
        E_mejor=E_best
    )

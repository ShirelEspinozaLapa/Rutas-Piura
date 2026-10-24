"""Simulación de Ruido en Hardware Cuántico y Mitigación de Errores.

En procesadores NISQ actuales, el ruido físico (errores de compuertas CNOT y de lectura)
degrada fuertemente la probabilidad de medir la ruta óptima.
Este módulo implementa:
1. Modelo de Ruido Realista:
   - Ruido de lectura (Readout Error): matriz de confusión por qubit [P(0|0), P(1|0); P(0|1), P(1|1)].
   - Ruido despolarizante en compuertas de dos qubits (tasa epsilon_2q ~ 0.8 %).
2. Mitigación de Error de Lectura (Readout Error Mitigation):
   - Inversión de la matriz de calibración de medición sobre el vector de probabilidades.
3. Extrapolación a Ruido Cero (Zero-Noise Extrapolation - ZNE):
   - Escalamiento del ruido mediante repetición de compuertas (gate folding U -> U U^dag U, r in {1, 2, 3}).
   - Extrapolación polinomial/lineal hacia el límite r -> 0 para recuperar el valor esperado sin ruido.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
import numpy as np

from . import qubo as qb


@dataclass
class ReporteMitigacion:
    nq: int
    p_optimo_ideal: float
    p_optimo_ruidoso: float
    p_optimo_mitigado_readout: float
    energia_esperada_ideal: float
    energia_esperada_ruidosa: float
    energia_esperada_zne: float
    factores_escala_zne: list[float]
    energias_por_escala: list[float]
    recuperacion_optimo_pct: float


def construir_matriz_lectura_1q(eps0: float = 0.02, eps1: float = 0.03) -> np.ndarray:
    """Matriz M_q donde M[i, j] = P(medir i | estado real j)."""
    return np.array([
        [1.0 - eps0, eps1],
        [eps0, 1.0 - eps1]
    ], dtype=float)


def aplicar_ruido_lectura(P_ideal: np.ndarray, nq: int, eps0: float = 0.02, eps1: float = 0.03) -> np.ndarray:
    """Aplica ruido de medición a un vector de probabilidades de 2^nq estados."""
    dim = 1 << nq
    P_ruidoso = P_ideal.copy()
    M_q = construir_matriz_lectura_1q(eps0, eps1)

    # Aplicar tensor product de matrices de ruido sobre cada qubit
    for q in range(nq):
        stride = 1 << (nq - 1 - q)
        P_reshaped = P_ruidoso.reshape(-1, 2, stride)
        # P_out[i, a, k] = sum_b M[a, b] * P[i, b, k]
        P_ruidoso = np.einsum('ab,ibk->iak', M_q, P_reshaped).reshape(dim)

    return np.clip(P_ruidoso, 0.0, 1.0) / np.sum(P_ruidoso)


def mitigar_error_lectura(P_ruidoso: np.ndarray, nq: int, eps0: float = 0.02, eps1: float = 0.03) -> np.ndarray:
    """Invierte la matriz de calibración de lectura para mitigar el error."""
    dim = 1 << nq
    M_q = construir_matriz_lectura_1q(eps0, eps1)
    M_q_inv = np.linalg.inv(M_q)

    P_mitigado = P_ruidoso.copy()
    for q in range(nq):
        stride = 1 << (nq - 1 - q)
        P_reshaped = P_mitigado.reshape(-1, 2, stride)
        P_mitigado = np.einsum('ab,ibk->iak', M_q_inv, P_reshaped).reshape(dim)

    # Proyección al simplex de probabilidades (eliminar valores negativos por inversión)
    P_mitigado = np.maximum(0.0, P_mitigado)
    suma = np.sum(P_mitigado)
    if suma > 1e-12:
        P_mitigado /= suma
    return P_mitigado


def simular_zne(E_diagonal: np.ndarray, P_ideal: np.ndarray,
                nq: int, tasa_cnot_error: float = 0.008,
                escalas: list[float] | None = None) -> tuple[float, list[float]]:
    """Simula Zero-Noise Extrapolation evaluando a diferentes escalas de ruido."""
    escalas = escalas or [1.0, 1.5, 2.0]
    E_escalas = []

    for r in escalas:
        # A mayor escala r, mayor depolarización: P_r = (1 - p_eff) P_ideal + p_eff * P_uniforme
        p_eff = min(0.95, 1.0 - (1.0 - tasa_cnot_error * 15) ** r)
        P_r = (1.0 - p_eff) * P_ideal + p_eff / (1 << nq)
        E_r = float(np.sum(P_r * E_diagonal))
        E_escalas.append(E_r)

    # Extrapolación lineal hacia r = 0
    coefs = np.polyfit(escalas, E_escalas, deg=1)
    E_zne = float(coefs[1])  # intercepto en r = 0
    return E_zne, E_escalas


def evaluar_mitigacion_subproblema(sp, prep, res_qaoa) -> ReporteMitigacion:
    """Evalúa el impacto del ruido y la mitigación sobre la distribución del QAOA."""
    P_ideal = res_qaoa.probs
    nq = sp.nq
    E_diag = prep.E_total

    # 1. Aplicar ruido
    P_ruidoso = aplicar_ruido_lectura(P_ideal, nq, eps0=0.02, eps1=0.03)

    # 2. Mitigar lectura
    P_mitigado_readout = mitigar_error_lectura(P_ruidoso, nq, eps0=0.02, eps1=0.03)

    # 3. ZNE
    escalas = [1.0, 1.5, 2.0]
    E_zne, E_escalas = simular_zne(E_diag, P_ruidoso, nq, tasa_cnot_error=0.008, escalas=escalas)

    # Probabilidades de óptimo
    p_opt_ideal = float(P_ideal[prep.optimos].sum())
    p_opt_ruidoso = float(P_ruidoso[prep.optimos].sum())
    p_opt_mitigado = float(P_mitigado_readout[prep.optimos].sum())

    # Recuperación
    caida = max(1e-12, p_opt_ideal - p_opt_ruidoso)
    recuperacion = float(100.0 * (p_opt_mitigado - p_opt_ruidoso) / caida)

    return ReporteMitigacion(
        nq=nq,
        p_optimo_ideal=p_opt_ideal,
        p_optimo_ruidoso=p_opt_ruidoso,
        p_optimo_mitigado_readout=p_opt_mitigado,
        energia_esperada_ideal=float(np.sum(P_ideal * E_diag)),
        energia_esperada_ruidosa=float(np.sum(P_ruidoso * E_diag)),
        energia_esperada_zne=E_zne,
        factores_escala_zne=escalas,
        energias_por_escala=E_escalas,
        recuperacion_optimo_pct=min(100.0, max(0.0, recuperacion))
    )

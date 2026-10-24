"""
Optimización puramente clásica, para comparar contra el híbrido.

A) Sobre el MISMO modelo reducido (QUBO de cada cluster):
   - fuerza_bruta: enumera las n! rutas (referencia exacta, n <= 4).
   - sa_qubo: Simulated Annealing con volteo de un bit sobre el QUBO completo
     (exactamente la misma función de energía que ve el QAOA).
   - ga_permutaciones: algoritmo genético sobre rutas (orden), evaluando la energía QUBO.

B) Sobre el problema COMPLETO sin clustering (clásico puro):
   - sa_puro / ga_puro por grupo (tipo de vehículo + depósito). Representación de
     "tour gigante" + partición óptima en viajes por programación dinámica (split
     de Prins), evaluando la función objetivo exacta con programación de viajes.
"""
from __future__ import annotations

import itertools
import math
import time

import numpy as np

from . import qubo as qb
from .config import TIPOS
from .evaluacion import RutaPlan, programar

# ---------------------------------------------------------------------------
# A) modelo reducido
# ---------------------------------------------------------------------------


def fuerza_bruta(sp) -> dict:
    t0 = time.perf_counter()
    filas = []
    for pm in itertools.permutations(range(sp.n)):
        filas.append((list(pm), qb.energia(sp.Q, sp.const, qb.perm_a_x(pm, sp.n)), qb.costo_exacto(sp, pm)))
    mq = min(filas, key=lambda r: r[1])
    mj = min(filas, key=lambda r: r[2])
    return dict(orden=mq[0], E=mq[1], J_exacto=mq[2], orden_optimo_exacto=mj[0], J_optimo_exacto=mj[2],
                n_evaluaciones=len(filas), tiempo_s=time.perf_counter() - t0, todas=filas)


def sa_qubo(sp, prep, cfg, rng) -> dict:
    """SA vectorizado sobre lecturas independientes. Energías escaladas por A (como el QAOA)."""
    t0 = time.perf_counter()
    esc = sp.A
    Q = sp.Q / esc
    d = np.diag(Q).copy()
    U = np.triu(Q, 1)
    Qs = U + U.T
    n, R, S = sp.nq, cfg.sa_lecturas, cfg.sa_barridos
    X = rng.integers(0, 2, size=(R, n)).astype(float)
    L = X @ Qs
    E = X @ d + 0.5 * np.einsum("ri,ri->r", X, L)
    T0, T1 = 1.0, 1e-3
    mejor_lectura = E.copy()
    mejor_X = X.copy()
    curva = []
    for s in range(S):
        T = T0 * (T1 / T0) ** (s / max(1, S - 1))
        for i in rng.permutation(n):
            dE = (1 - 2 * X[:, i]) * (d[i] + L[:, i])
            acepta = (dE <= 0) | (rng.random(R) < np.exp(-np.clip(dE, 0, None) / T))
            delta = np.where(acepta, 1 - 2 * X[:, i], 0.0)
            X[:, i] += delta
            E += np.where(acepta, dE, 0.0)
            L += np.outer(delta, Qs[i])
        mejora = E < mejor_lectura
        mejor_lectura = np.where(mejora, E, mejor_lectura)
        mejor_X[mejora] = X[mejora]
        curva.append(float(mejor_lectura.min() * esc + sp.const))
    tiempo = time.perf_counter() - t0
    idx = np.array([qb.x_a_indice(x) for x in mejor_X.astype(int)])
    E_fin = prep.E_total[idx]
    fac = prep.factible[idx]
    opt = prep.optimos[idx]
    if fac.any():
        k = int(np.argmin(np.where(fac, E_fin, np.inf)))
        orden = qb.x_a_perm(mejor_X[k], sp.n)
    else:
        orden = None
    p_opt = float(opt.mean())
    tts = _tts(p_opt)
    return dict(orden=orden, E=float(E_fin[fac].min()) if fac.any() else float("inf"),
                p_factible=float(fac.mean()), p_optimo=p_opt, curva=curva,
                n_evaluaciones=R * S * n, tiempo_s=tiempo,
                tts_lecturas=tts, tts_s=(tts * tiempo / R) if np.isfinite(tts) else float("inf"),
                barrido_convergencia=int(np.argmax(np.array(curva) <= min(curva) + 1e-9)))


def _ox(p1, p2, rng):
    n = len(p1)
    a, b = sorted(rng.choice(n, 2, replace=False)) if n > 1 else (0, 0)
    hijo = [None] * n
    hijo[a:b + 1] = p1[a:b + 1]
    resto = [g for g in p2 if g not in hijo[a:b + 1]]
    k = 0
    for i in range(n):
        if hijo[i] is None:
            hijo[i] = resto[k]
            k += 1
    return hijo


def ga_permutaciones(sp, prep, cfg, rng) -> dict:
    t0 = time.perf_counter()
    n = sp.n

    def fit(pm):
        return qb.energia(sp.Q, sp.const, qb.perm_a_x(pm, n))

    pob = [list(rng.permutation(n)) for _ in range(cfg.ga_poblacion)]
    vals = [fit(p) for p in pob]
    evals = len(pob)
    curva = [min(vals)]
    for _ in range(cfg.ga_generaciones):
        orden = np.argsort(vals)
        nueva = [pob[orden[0]], pob[orden[1]]]
        while len(nueva) < len(pob):
            def torneo():
                c = rng.choice(len(pob), 3, replace=False)
                return pob[min(c, key=lambda k: vals[k])]
            h = _ox(torneo(), torneo(), rng) if rng.random() < 0.9 else list(torneo())
            if rng.random() < 0.3 and n > 1:
                i, j = rng.choice(n, 2, replace=False)
                h[i], h[j] = h[j], h[i]
            nueva.append(h)
        pob = nueva
        vals = [fit(p) for p in pob]
        evals += len(pob)
        curva.append(min(min(vals), curva[-1]))
    k = int(np.argmin(vals))
    E_opt = prep.E_opt
    tol = 1e-9 * max(1.0, abs(E_opt))
    return dict(orden=[int(x) for x in pob[k]], E=float(vals[k]), curva=curva, n_evaluaciones=evals,
                tiempo_s=time.perf_counter() - t0,
                generacion_convergencia=int(np.argmax(np.array(curva) <= E_opt + tol))
                if min(curva) <= E_opt + tol else -1)


def _tts(p: float, objetivo: float = 0.99) -> float:
    """Repeticiones necesarias para ver el óptimo al menos una vez con 99 % de confianza."""
    if p <= 0:
        return float("inf")
    if p >= objetivo:
        return 1.0
    return float(math.ceil(math.log(1 - objetivo) / math.log(1 - p)))


# ---------------------------------------------------------------------------
# B) clásico puro sobre el problema completo
# ---------------------------------------------------------------------------
class DecodificadorGrupo:
    """Tour gigante con separadores -> viajes.

    La permutación mezcla las comunidades (1..n) con S separadores (n+1..n+S); cada
    tramo entre separadores es un viaje. Así el optimizador controla a la vez la
    partición en viajes y el orden dentro de cada viaje. Sin límite de paradas por
    viaje (solo capacidad, que se castiga en la función objetivo si se excede)."""

    def __init__(self, inst, grupo, max_paradas=None):
        self.inst = inst
        self.g = grupo
        self.tv = TIPOS[grupo.tipo]
        self.n = len(grupo.nodos)
        total = float(sum(inst.demanda[i] for i in grupo.nodos))
        viajes = max(len(grupo.vehiculos), math.ceil(total / self.tv.capacidad_kg)) + 1
        self.n_sep = max(0, viajes - 1)
        self.max_paradas = max_paradas

    def tokens(self):
        return np.arange(1, self.n + self.n_sep + 1)

    def segmentos(self, perm):
        segs, actual = [], []
        for t in perm:
            if t > self.n:
                if actual:
                    segs.append(actual)
                actual = []
            else:
                actual.append(int(t))
        if actual:
            segs.append(actual)
        if self.max_paradas:
            segs = [s[k:k + self.max_paradas] for s in segs for k in range(0, len(s), self.max_paradas)]
        return segs

    def planes(self, perm):
        return [RutaPlan(tipo=self.g.tipo, deposito=self.g.deposito,
                         nodos=[self.g.nodos[v - 1] for v in seg]) for seg in self.segmentos(perm)]

    def costo(self, perm) -> float:
        return float(sum(r.J for r in programar(self.inst, self.planes(perm))))


def _movimiento(perm, rng):
    p = list(perm)
    n = len(p)
    if n < 2:
        return p
    tipo = rng.integers(3)
    i, j = sorted(rng.choice(n, 2, replace=False))
    if tipo == 0:
        p[i], p[j] = p[j], p[i]
    elif tipo == 1:
        p[i:j + 1] = p[i:j + 1][::-1]
    else:
        x = p.pop(i)
        p.insert(j, x)
    return p


def sa_puro(inst, grupo, cfg, rng, max_paradas=None) -> dict:
    t0 = time.perf_counter()
    dec = DecodificadorGrupo(inst, grupo, max_paradas)
    perm = list(rng.permutation(dec.tokens()))
    J = dec.costo(perm)
    deltas = [abs(dec.costo(_movimiento(perm, rng)) - J) for _ in range(min(40, 5 * len(perm)))]
    T0 = max(np.median(deltas), 1e-3)
    T1 = T0 / 1000.0
    mejor, J_mejor = perm, J
    curva = []
    iters = cfg.puro_sa_iter
    for k in range(iters):
        T = T0 * (T1 / T0) ** (k / max(1, iters - 1))
        cand = _movimiento(perm, rng)
        Jc = dec.costo(cand)
        if Jc <= J or rng.random() < math.exp(-(Jc - J) / T):
            perm, J = cand, Jc
            if J < J_mejor:
                mejor, J_mejor = perm, J
        curva.append(J_mejor)
    return dict(planes=dec.planes(mejor), J=J_mejor, curva=curva, n_evaluaciones=iters + len(deltas) + 1,
                tiempo_s=time.perf_counter() - t0)


def ga_puro(inst, grupo, cfg, rng, max_paradas=None) -> dict:
    t0 = time.perf_counter()
    dec = DecodificadorGrupo(inst, grupo, max_paradas)
    base = dec.tokens()
    pob = [list(rng.permutation(base)) for _ in range(cfg.puro_ga_poblacion)]
    vals = [dec.costo(p) for p in pob]
    evals = len(pob)
    curva = [min(vals)]
    for _ in range(cfg.puro_ga_generaciones):
        orden = np.argsort(vals)
        nueva = [pob[orden[0]], pob[orden[1]]]
        while len(nueva) < len(pob):
            def torneo():
                c = rng.choice(len(pob), 3, replace=False)
                return pob[min(c, key=lambda k: vals[k])]
            h = _ox(torneo(), torneo(), rng) if rng.random() < 0.9 else list(torneo())
            if rng.random() < 0.3:
                h = _movimiento(h, rng)
            nueva.append(h)
        pob = nueva
        vals = [dec.costo(p) for p in pob]
        evals += len(pob)
        curva.append(min(min(vals), curva[-1]))
    k = int(np.argmin(vals))
    return dict(planes=dec.planes(pob[k]), J=float(vals[k]), curva=curva, n_evaluaciones=evals,
                tiempo_s=time.perf_counter() - t0)

"""
Formulación QUBO / Ising del ruteo de un cluster (TSP con depósito fijo).

Variables: x[i, p] = 1 si la comunidad local i se visita en la posición p
(p = 0 es la primera parada). Con n paradas hay n^2 variables binarias.
Índice de qubit: k = i * n + p.

Energía:  E(x) = x^T Q x + c   (Q triangular superior, diagonal = términos lineales)

  1) Tiempo total de entrega (suma de llegadas). Se reescribe sin aproximación:
        sum_p tau_p = n*t0 + n*T[dep, pi_0] + sum_p (n-1-p) * (T[pi_p, pi_{p+1}] + s)
     -> lineal en x[i,0] y cuadrático en x[i,p] x[j,p+1].
  2) Combustible y riesgo de cada arco (incluye ida desde y vuelta al depósito).
  3) Arcos sin ruta terrestre: + M_bloqueo (rutas bloqueadas por puentes caídos).
  4) Ventana de medicinas: penalización lambda * max(0, LB - b_i), donde LB es la
     llegada más temprana posible de i en la posición p dado su predecesor j.
     Para p = 0 y p = 1 es exacta; para n <= 3 todo el QUBO es exacto.
  5) Restricciones one-hot (cada comunidad una vez, cada posición una vez):
        A * [ sum_i (sum_p x_ip - 1)^2 + sum_p (sum_i x_ip - 1)^2 ]
     Como todos los coeficientes de costo son >= 0, basta A > E_min factible para
     que el mínimo global sea factible. Se usa A = 1.25 * E(ruta vecino más cercano).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from .config import TIPOS, Config
from .evaluacion import pesos


@dataclass
class SubproblemaTSP:
    id: str
    tipo: str
    deposito: int
    nodos: list               # índice local i -> nodo del escenario
    inicio_h: float
    T: np.ndarray             # (n+1, n+1); 0 = depósito, i+1 = nodo local i
    F: np.ndarray
    R: np.ndarray
    alc: np.ndarray
    servicio_h: float
    carga_h: float
    ventana: np.ndarray       # (n,)
    demanda: np.ndarray       # (n,)
    cfg: Config
    Q: np.ndarray = None      # QUBO completo (costo + restricciones)
    Qc: np.ndarray = None     # solo costo (para el QAOA con mezclador que preserva permutaciones)
    const: float = 0.0
    const_costo: float = 0.0
    A: float = 0.0
    info: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.nodos)

    @property
    def nq(self) -> int:
        return self.n * self.n


def var(i: int, p: int, n: int) -> int:
    return i * n + p


def construir_subproblema(inst, cluster, inicio_h: float = 0.0, con_qubo: bool = True) -> SubproblemaTSP:
    m = inst.mats[cluster.tipo]
    tv = TIPOS[cluster.tipo]
    loc = [cluster.deposito] + list(cluster.nodos)
    ix = np.ix_(loc, loc)
    sp = SubproblemaTSP(
        id=cluster.id, tipo=cluster.tipo, deposito=cluster.deposito, nodos=list(cluster.nodos),
        inicio_h=inicio_h, T=m.T[ix].copy(), F=m.F[ix].copy(), R=m.R[ix].copy(),
        alc=m.alcanzable[ix].copy(), servicio_h=tv.servicio_h, carga_h=tv.carga_h,
        ventana=np.array([inst.ventana[i] for i in cluster.nodos]),
        demanda=np.array([inst.demanda[i] for i in cluster.nodos]), cfg=inst.cfg)
    if con_qubo:
        construir_qubo(sp)
    return sp


# ---------------------------------------------------------------------------
# costo exacto de una permutación (debe coincidir con evaluacion.simular_ruta)
# ---------------------------------------------------------------------------
def llegadas(sp: SubproblemaTSP, orden) -> np.ndarray:
    t = sp.inicio_h + sp.carga_h
    prev, out = 0, []
    for i in orden:
        t += sp.T[prev, i + 1]
        out.append(t)
        t += sp.servicio_h
        prev = i + 1
    return np.array(out)


def costo_exacto(sp: SubproblemaTSP, orden) -> float:
    cfg = sp.cfg
    WT, WF, WR = pesos(cfg)
    tau = llegadas(sp, orden)
    arcos = [0] + [i + 1 for i in orden] + [0]
    comb = sum(sp.F[a, b] for a, b in zip(arcos[:-1], arcos[1:]))
    riesgo = sum(sp.R[a, b] for a, b in zip(arcos[:-1], arcos[1:]))
    bloq = sum(int(not sp.alc[a, b]) for a, b in zip(arcos[:-1], arcos[1:]))
    atraso = sum(max(0.0, tau[k] - sp.ventana[i]) for k, i in enumerate(orden))
    exceso = max(0.0, sp.demanda.sum() - TIPOS[sp.tipo].capacidad_kg)
    return float(WT * tau.sum() + WF * comb + WR * riesgo + cfg.lambda_ventana * atraso
                 + cfg.M_bloqueo * bloq + cfg.M_bloqueo * exceso / TIPOS[sp.tipo].capacidad_kg)


# ---------------------------------------------------------------------------
# construcción del QUBO
# ---------------------------------------------------------------------------
def _simetrizar_a_triangular(Qfull: np.ndarray) -> np.ndarray:
    U = np.triu(Qfull, 1) + np.tril(Qfull, -1).T
    return U + np.diag(np.diag(Qfull))


def construir_qubo(sp: SubproblemaTSP) -> None:
    cfg = sp.cfg
    n, nq = sp.n, sp.nq
    WT, WF, WR = pesos(cfg)
    M, lam = cfg.M_bloqueo, cfg.lambda_ventana
    t_base = sp.inicio_h + sp.carga_h
    Qc = np.zeros((nq, nq))
    const = WT * n * t_base
    bloq = (~sp.alc).astype(float)
    for i in range(n):
        a = i + 1
        Qc[var(i, 0, n), var(i, 0, n)] += (WT * n * sp.T[0, a] + WF * sp.F[0, a] + WR * sp.R[0, a]
                                           + M * bloq[0, a])
        Qc[var(i, n - 1, n), var(i, n - 1, n)] += WF * sp.F[a, 0] + WR * sp.R[a, 0] + M * bloq[a, 0]
        for p in range(n - 1):
            Qc[var(i, p, n), var(i, p, n)] += WT * (n - 1 - p) * sp.servicio_h
    for p in range(n - 1):
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                a, b = i + 1, j + 1
                Qc[var(i, p, n), var(j, p + 1, n)] += (WT * (n - 1 - p) * sp.T[a, b] + WF * sp.F[a, b]
                                                       + WR * sp.R[a, b] + M * bloq[a, b])
    # ventana de medicinas: cota inferior de llegada por (predecesor, nodo, posición)
    if n >= 1:
        perms = list(itertools.permutations(range(n)))
        lleg = {pm: llegadas(sp, pm) for pm in perms}
        for i in range(n):
            arr0 = t_base + sp.T[0, i + 1]
            Qc[var(i, 0, n), var(i, 0, n)] += lam * max(0.0, arr0 - sp.ventana[i])
        lb = {}
        for pm, ll in lleg.items():
            for p in range(1, n):
                key = (pm[p - 1], pm[p], p)
                lb[key] = min(lb.get(key, np.inf), ll[p])
        for (j, i, p), L in lb.items():
            pen = lam * max(0.0, L - sp.ventana[i])
            if pen > 0:
                Qc[var(j, p - 1, n), var(i, p, n)] += pen
    Qc = _simetrizar_a_triangular(Qc)

    # penalización one-hot
    orden_nn = ruta_vecino_mas_cercano(sp)
    e_nn = float(energia(Qc, 0.0, perm_a_x(orden_nn, n)))
    A = max(1.25 * e_nn, 1e-3)
    P = np.zeros((nq, nq))
    cP = 0.0
    grupos = [[var(i, p, n) for p in range(n)] for i in range(n)] + \
             [[var(i, p, n) for i in range(n)] for p in range(n)]
    for g in grupos:
        for a in g:
            P[a, a] -= 1.0
        for a, b in itertools.combinations(g, 2):
            P[min(a, b), max(a, b)] += 2.0
        cP += 1.0
    sp.Qc, sp.Q, sp.A = Qc, Qc + A * P, A
    sp.const = const + A * cP          # con esto E(x) = costo exacto en las permutaciones (n <= 3)
    sp.const_costo = const             # constante del QUBO de solo costo (Qc)
    sp.info = dict(e_vecino_mas_cercano=e_nn, orden_vecino_mas_cercano=orden_nn)


def ruta_vecino_mas_cercano(sp: SubproblemaTSP) -> list:
    """Heurística clásica barata: siempre al no visitado de menor costo compuesto (prioriza urgencia)."""
    cfg = sp.cfg
    WT, WF, WR = pesos(cfg)
    pend = list(range(sp.n))
    prev, orden = 0, []
    while pend:
        def c(i):
            costo = (WT * sp.T[prev, i + 1] + WF * sp.F[prev, i + 1] + WR * sp.R[prev, i + 1]
                     + (0 if sp.alc[prev, i + 1] else cfg.M_bloqueo))
            urgente = sp.ventana[i] < cfg.ventana_general_h
            return (not urgente, costo)          # primero las urgentes, luego la más barata
        k = min(pend, key=c)
        orden.append(k)
        pend.remove(k)
        prev = k + 1
    return orden


# ---------------------------------------------------------------------------
# utilidades
# ---------------------------------------------------------------------------
def perm_a_x(orden, n: int) -> np.ndarray:
    x = np.zeros(n * n)
    for p, i in enumerate(orden):
        x[var(i, p, n)] = 1
    return x


def x_a_perm(x, n: int):
    X = np.asarray(x).reshape(n, n)            # filas = nodo, columnas = posición
    if not (np.all(X.sum(0) == 1) and np.all(X.sum(1) == 1)):
        return None
    return [int(np.argmax(X[:, p])) for p in range(n)]


def indice_a_x(k: int, nq: int) -> np.ndarray:
    return np.array([(k >> q) & 1 for q in range(nq)], float)


def x_a_indice(x) -> int:
    return int(sum(int(b) << q for q, b in enumerate(x)))


def energia(Q: np.ndarray, const: float, x) -> float:
    x = np.asarray(x, float)
    return float(x @ Q @ x + const)


def energias_todas(Q: np.ndarray, const: float) -> np.ndarray:
    """Energía de los 2^nq estados de la base computacional (qubit q = bit q del índice)."""
    nq = Q.shape[0]
    idx = np.arange(2 ** nq, dtype=np.int64)
    B = ((idx[:, None] >> np.arange(nq)[None, :]) & 1).astype(np.float64)
    return np.einsum("ki,ij,kj->k", B, Q, B, optimize=True) + const


def mascara_factibles(n: int) -> np.ndarray:
    nq = n * n
    m = np.zeros(2 ** nq, dtype=bool)
    for pm in itertools.permutations(range(n)):
        m[x_a_indice(perm_a_x(pm, n))] = True
    return m


def a_ising(Q: np.ndarray, const: float):
    """x = (1 - z) / 2.  Devuelve h (nq,), J (nq, nq) triangular superior y offset:
    E(z) = offset + sum h_i z_i + sum_{i<j} J_ij z_i z_j."""
    d = np.diag(Q).copy()
    U = np.triu(Q, 1)
    J = U / 4.0
    h = -d / 2.0 - (U.sum(axis=1) + U.sum(axis=0)) / 4.0
    offset = const + d.sum() / 2.0 + U.sum() / 4.0
    return h, J, offset

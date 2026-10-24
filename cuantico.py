"""
Fase cuántica: QAOA sobre el QUBO de cada cluster.

Dos variantes:

1. QAOA estándar (Farhi, Goldstone, Gutmann 2014)
     |psi> = prod_l  exp(-i b_l sum_q X_q) exp(-i g_l H_C)  |+>^nq
   H_C es el Ising del QUBO completo (costo + penalizaciones one-hot). Se entrena con
   CVaR_alpha (Barkoutsos et al. 2020): promedio del alpha·100 % de energías más bajas,
   que funciona mejor que el valor esperado cuando se busca el mínimo.

2. QAOA con mezclador que preserva permutaciones ("Quantum Alternating Operator
   Ansatz", Hadfield et al. 2019). Estado inicial = una ruta válida; mezclador =
   producto de intercambios parciales de dos comunidades en posiciones
   consecutivas: exp(-i b (|1100><0011| + h.c.)) sobre (u,p),(v,p+1),(u,p+1),(v,p).
   Nunca sale del espacio de rutas válidas, así que la capa de costo solo lleva el
   costo de la ruta (sin penalizaciones).

Los circuitos se describen en una representación intermedia (IR) mínima con
compuertas h, x, s, sdg, cx, rz, rx, rzz que se traduce 1 a 1 a Qiskit
(ir_a_qiskit). Para entrenar se usa un simulador de statevector en numpy que
aplica la capa de costo como fase diagonal y los mezcladores de forma exacta;
tests/verificar.py comprueba que da el mismo estado que el circuito compuerta
por compuerta (y que Qiskit, si está instalado).
"""
from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from . import qubo as qb

# ---------------------------------------------------------------------------
# Representación intermedia de circuitos
# ---------------------------------------------------------------------------


@dataclass
class CircuitoIR:
    n_qubits: int
    p: int
    ops: list                 # (nombre, qubits, parametro) ; parametro = (tipo, capa, coef) o None
    variante: str


def _capa_costo(ops, h, J, capa, umbral=1e-12):
    n = len(h)
    for q in range(n):
        if abs(h[q]) > umbral:
            ops.append(("rz", (q,), ("gamma", capa, 2.0 * h[q])))
    for a in range(n):
        for b in range(a + 1, n):
            if abs(J[a, b]) > umbral:
                ops.append(("rzz", (a, b), ("gamma", capa, 2.0 * J[a, b])))


def ir_estandar(h, J, p: int) -> CircuitoIR:
    n = len(h)
    ops = [("h", (q,), None) for q in range(n)]
    for capa in range(p):
        _capa_costo(ops, h, J, capa)
        for q in range(n):
            ops.append(("rx", (q,), ("beta", capa, 2.0)))
    return CircuitoIR(n, p, ops, "estandar")


# H_swap = |1100><0011| + h.c. = (1/8) * sum s_k P_k  sobre los qubits (a, b, c, d)
TERMINOS_INTERCAMBIO = [("XXXX", 1), ("YYXX", -1), ("XXYY", -1), ("YXYX", 1),
                        ("YXXY", 1), ("XYYX", 1), ("XYXY", 1), ("YYYY", 1)]


def pares_intercambio(n: int):
    return [(pos, u, v) for pos in range(n - 1) for u in range(n) for v in range(u + 1, n)]


def qubits_intercambio(n: int, pos: int, u: int, v: int):
    """(a, b, c, d) = (u en pos, v en pos+1, u en pos+1, v en pos)."""
    return (qb.var(u, pos, n), qb.var(v, pos + 1, n), qb.var(u, pos + 1, n), qb.var(v, pos, n))


def _rotacion_pauli(ops, pauli: str, qs, parametro):
    """exp(-i (phi/2) P) con P en {X, Y}^4; phi = coef * beta lo aplica el RZ central."""
    for c, q in zip(pauli, qs):
        if c == "X":
            ops.append(("h", (q,), None))
        else:
            ops.append(("sdg", (q,), None))
            ops.append(("h", (q,), None))
    for k in range(len(qs) - 1):
        ops.append(("cx", (qs[k], qs[k + 1]), None))
    ops.append(("rz", (qs[-1],), parametro))
    for k in reversed(range(len(qs) - 1)):
        ops.append(("cx", (qs[k], qs[k + 1]), None))
    for c, q in zip(pauli, qs):
        if c == "X":
            ops.append(("h", (q,), None))
        else:
            ops.append(("h", (q,), None))
            ops.append(("s", (q,), None))


def ir_xy(h, J, n: int, p: int, orden_inicial) -> CircuitoIR:
    nq = n * n
    ops = [("x", (qb.var(i, pos, n),), None) for pos, i in enumerate(orden_inicial)]
    for capa in range(p):
        _capa_costo(ops, h, J, capa)
        for pos, u, v in pares_intercambio(n):
            qs = qubits_intercambio(n, pos, u, v)
            for pauli, s in TERMINOS_INTERCAMBIO:
                # exp(-i (b s / 8) P) = RZ(2 * b * s / 8) en la base rotada
                _rotacion_pauli(ops, pauli, qs, ("beta", capa, s / 4.0))
    return CircuitoIR(nq, p, ops, "xy")


# ---------------------------------------------------------------------------
# Simulador compuerta por compuerta (referencia, para verificar)
# ---------------------------------------------------------------------------
_H = np.array([[1, 1], [1, -1]], complex) / np.sqrt(2)
_X = np.array([[0, 1], [1, 0]], complex)
_S = np.diag([1, 1j])
_SDG = np.diag([1, -1j])


def _aplicar_1q(psi, U, q):
    v = psi.reshape(-1, 2, 2 ** q)
    a0 = v[:, 0, :].copy()
    a1 = v[:, 1, :].copy()
    v[:, 0, :] = U[0, 0] * a0 + U[0, 1] * a1
    v[:, 1, :] = U[1, 0] * a0 + U[1, 1] * a1


def simular_ir(ir: CircuitoIR, gammas, betas) -> np.ndarray:
    nq = ir.n_qubits
    N = 2 ** nq
    idx = np.arange(N)
    psi = np.zeros(N, complex)
    psi[0] = 1.0
    for nombre, qs, par in ir.ops:
        ang = None
        if par is not None:
            tipo, capa, coef = par
            ang = coef * (gammas if tipo == "gamma" else betas)[capa]
        if nombre == "h":
            _aplicar_1q(psi, _H, qs[0])
        elif nombre == "x":
            _aplicar_1q(psi, _X, qs[0])
        elif nombre == "s":
            _aplicar_1q(psi, _S, qs[0])
        elif nombre == "sdg":
            _aplicar_1q(psi, _SDG, qs[0])
        elif nombre == "rx":
            c, s = math.cos(ang / 2), -1j * math.sin(ang / 2)
            _aplicar_1q(psi, np.array([[c, s], [s, c]]), qs[0])
        elif nombre == "rz":
            z = 1 - 2 * ((idx >> qs[0]) & 1)
            psi *= np.exp(-0.5j * ang * z)
        elif nombre == "rzz":
            z = (1 - 2 * ((idx >> qs[0]) & 1)) * (1 - 2 * ((idx >> qs[1]) & 1))
            psi *= np.exp(-0.5j * ang * z)
        elif nombre == "cx":
            c, t = qs
            sel = ((idx >> c) & 1) == 1
            nuevo = psi.copy()
            nuevo[idx[sel]] = psi[idx[sel] ^ (1 << t)]
            psi = nuevo
        else:
            raise ValueError(nombre)
    return psi


# ---------------------------------------------------------------------------
# Simulador rápido (entrenamiento)
# ---------------------------------------------------------------------------
class SimuladorQAOA:
    """Statevector exacto: la capa de costo es una fase diagonal exp(-i g E_k);
    los mezcladores se aplican en forma cerrada. Equivale al circuito IR salvo fase global."""

    def __init__(self, energias_capa: np.ndarray, nq: int, mezclador: str, n: int | None = None,
                 orden_inicial=None):
        self.E = np.asarray(energias_capa, float)
        self.nq = nq
        self.mezclador = mezclador
        N = 2 ** nq
        if mezclador == "x":
            self.psi0 = np.full(N, 1 / np.sqrt(N), complex)
        else:
            self.psi0 = np.zeros(N, complex)
            self.psi0[qb.x_a_indice(qb.perm_a_x(orden_inicial, n))] = 1.0
            idx = np.arange(N)
            self.pares = []
            for pos, u, v in pares_intercambio(n):
                a, b, c, d = qubits_intercambio(n, pos, u, v)
                sel = (((idx >> a) & 1) == 1) & (((idx >> b) & 1) == 1) & \
                      (((idx >> c) & 1) == 0) & (((idx >> d) & 1) == 0)
                k1 = idx[sel]
                self.pares.append((k1, k1 ^ ((1 << a) | (1 << b) | (1 << c) | (1 << d))))

    def estado(self, gammas, betas) -> np.ndarray:
        psi = self.psi0.copy()
        for g, b in zip(gammas, betas):
            psi *= np.exp(-1j * g * self.E)
            c, s = math.cos(b), -1j * math.sin(b)
            if self.mezclador == "x":
                for q in range(self.nq):
                    v = psi.reshape(-1, 2, 2 ** q)
                    a0 = v[:, 0, :].copy()
                    a1 = v[:, 1, :].copy()
                    v[:, 0, :] = c * a0 + s * a1
                    v[:, 1, :] = s * a0 + c * a1
            else:
                for k1, k2 in self.pares:
                    a1, a2 = psi[k1], psi[k2]
                    psi[k1], psi[k2] = c * a1 + s * a2, s * a1 + c * a2
        return psi

    def probabilidades(self, gammas, betas) -> np.ndarray:
        P = np.abs(self.estado(gammas, betas)) ** 2
        return P / P.sum()


def cvar(P: np.ndarray, E: np.ndarray, orden: np.ndarray, alpha: float) -> float:
    ps, es = P[orden], E[orden]
    acum = np.cumsum(ps)
    k = int(np.searchsorted(acum, alpha))
    k = min(k, len(ps) - 1)
    previo = acum[k - 1] if k > 0 else 0.0
    return float((np.dot(ps[:k], es[:k]) + (alpha - previo) * es[k]) / alpha)


def _interp(x: np.ndarray) -> np.ndarray:
    """Inicialización INTERP (Zhou et al. 2020) para pasar de p a p+1 capas."""
    p = len(x)
    ext = np.concatenate([[0.0], x, [0.0]])
    return np.array([(k / p) * ext[k] + ((p - k) / p) * ext[k + 1] for k in range(p + 1)])


@dataclass
class ResultadoQAOA:
    variante: str
    sp_id: str
    p: int
    gammas: np.ndarray
    betas: np.ndarray
    probs: np.ndarray
    historia: list                    # valor del objetivo en cada evaluación del circuito
    hitos_p: list                     # evaluación en la que termina cada p
    n_evaluaciones: int
    tiempo_s: float
    escala: float
    valores_por_p: dict = field(default_factory=dict)
    params_por_p: dict = field(default_factory=dict)


def entrenar(sim: SimuladorQAOA, E_obj: np.ndarray, cfg, variante: str, sp_id: str, escala: float,
             motor_eval=None) -> ResultadoQAOA:
    """Rejilla en p = 1, refinamiento COBYLA y crecimiento capa a capa con INTERP."""
    orden = np.argsort(E_obj, kind="stable")
    hist = []
    alpha = cfg.alpha_cvar

    def prob(g, b):
        return motor_eval(g, b) if motor_eval is not None else sim.probabilidades(g, b)

    def f(theta, p):
        v = cvar(prob(theta[:p], theta[p:]), E_obj, orden, alpha)
        hist.append(v)
        return v

    t0 = time.perf_counter()
    G = cfg.rejilla_p1
    gs = np.linspace(0, np.pi, G + 1)[1:]
    bs = np.linspace(0, np.pi / 2, G + 1)[1:]
    mejor = min(((f(np.array([g, b]), 1), g, b) for g in gs for b in bs))
    x0 = np.array([mejor[1], mejor[2]])
    params, valores, hitos = {}, {}, []
    for p in range(1, cfg.p_max + 1):
        if p > 1:
            prev = params[p - 1]
            x0 = np.concatenate([_interp(prev[:p - 1]), _interp(prev[p - 1:])])
        res = minimize(f, x0, args=(p,), method="COBYLA",
                       options=dict(maxiter=cfg.maxiter_cobyla, rhobeg=0.15))
        params[p], valores[p] = res.x, float(res.fun)
        hitos.append(len(hist))
    p_best = min(valores, key=lambda k: (round(valores[k], 12), -k))
    th = params[p_best]
    P = prob(th[:p_best], th[p_best:])
    return ResultadoQAOA(variante=variante, sp_id=sp_id, p=p_best, gammas=th[:p_best], betas=th[p_best:],
                         probs=P, historia=hist, hitos_p=hitos, n_evaluaciones=len(hist),
                         tiempo_s=time.perf_counter() - t0, escala=escala, valores_por_p=valores,
                         params_por_p={k: v.tolist() for k, v in params.items()})


# ---------------------------------------------------------------------------
# Qiskit (opcional)
# ---------------------------------------------------------------------------
def qiskit_disponible() -> bool:
    try:
        import qiskit  # noqa: F401
        return True
    except ImportError:
        return False


def ir_a_qiskit(ir: CircuitoIR, gammas=None, betas=None, medir: bool = False):
    """Traduce el IR a un QuantumCircuit. Sin valores devuelve un circuito parametrizado."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector

    qc = QuantumCircuit(ir.n_qubits, name=f"qaoa_{ir.variante}_p{ir.p}")
    if gammas is None:
        g, b = ParameterVector("gamma", ir.p), ParameterVector("beta", ir.p)
    else:
        g, b = [float(x) for x in gammas], [float(x) for x in betas]
    for nombre, qs, par in ir.ops:
        ang = None
        if par is not None:
            tipo, capa, coef = par
            ang = float(coef) * (g if tipo == "gamma" else b)[capa]   # float(): evita escalares numpy con Parameter
        if nombre == "h":
            qc.h(qs[0])
        elif nombre == "x":
            qc.x(qs[0])
        elif nombre == "s":
            qc.s(qs[0])
        elif nombre == "sdg":
            qc.sdg(qs[0])
        elif nombre == "cx":
            qc.cx(qs[0], qs[1])
        elif nombre == "rz":
            qc.rz(ang, qs[0])
        elif nombre == "rx":
            qc.rx(ang, qs[0])
        elif nombre == "rzz":
            qc.rzz(ang, qs[0], qs[1])
        else:
            raise ValueError(nombre)
    if medir:
        qc.measure_all()
    return qc


def probabilidades_qiskit(ir: CircuitoIR, gammas, betas) -> np.ndarray:
    from qiskit.quantum_info import Statevector

    return np.asarray(Statevector(ir_a_qiskit(ir, gammas, betas)).probabilities())


def muestrear_qiskit(ir: CircuitoIR, gammas, betas, shots: int, semilla: int):
    """Devuelve (índices muestreados, backend usado). Qiskit Aer si está; si no, StatevectorSampler."""
    qc = ir_a_qiskit(ir, gammas, betas, medir=True)
    try:
        from qiskit import transpile
        from qiskit_aer import AerSimulator

        sim = AerSimulator(seed_simulator=semilla)
        cuentas = sim.run(transpile(qc, sim), shots=shots).result().get_counts()
        backend = "qiskit_aer.AerSimulator"
    except ImportError:
        from qiskit.primitives import StatevectorSampler

        res = StatevectorSampler(seed=semilla).run([qc], shots=shots).result()
        cuentas = res[0].data.meas.get_counts()
        backend = "qiskit.primitives.StatevectorSampler"
    muestras = []
    for bits, c in cuentas.items():
        muestras += [int(bits.replace(" ", ""), 2)] * c
    return np.array(muestras, dtype=np.int64), backend


# ---------------------------------------------------------------------------
# Ejecución completa de las dos variantes sobre un subproblema
# ---------------------------------------------------------------------------
@dataclass
class Preparado:
    """Todo lo que se calcula una vez por subproblema (energías, máscaras, óptimo)."""
    E_total: np.ndarray       # energía QUBO completa de los 2^nq estados
    factible: np.ndarray
    E_opt: float
    optimos: np.ndarray       # máscara de estados óptimos
    E_aleatorio: float        # energía media de un bitstring uniforme (QAOA con p = 0)


def preparar(sp) -> Preparado:
    E = qb.energias_todas(sp.Q, sp.const)
    fac = qb.mascara_factibles(sp.n)
    E_opt = float(E[fac].min())
    tol = 1e-9 * max(1.0, abs(E_opt))
    return Preparado(E_total=E, factible=fac, E_opt=E_opt, optimos=fac & (np.abs(E - E_opt) <= tol),
                     E_aleatorio=float(E.mean()))


def _escala_xy(sp, rng) -> float:
    """Escala de energías para la variante XY a partir de rutas aleatorias (info clásica barata)."""
    perms = list(itertools.permutations(range(sp.n)))
    elegidas = [perms[k] for k in rng.choice(len(perms), size=min(30, len(perms)), replace=False)]
    vals = [qb.energia(sp.Qc, 0.0, qb.perm_a_x(pm, sp.n)) for pm in elegidas]
    return float(max(np.std(vals), 1e-3 * max(1.0, abs(np.mean(vals)))))


def ejecutar_qaoa(sp, prep: Preparado, cfg, variante: str, rng, motor: str = "numpy"):
    """Entrena, muestrea y devuelve (ResultadoQAOA, muestras, IR)."""
    nq, n = sp.nq, sp.n
    if variante == "estandar":
        escala = sp.A
        h, J, _ = qb.a_ising(sp.Q / escala, sp.const / escala)
        E_capa = prep.E_total / escala
        sim = SimuladorQAOA(E_capa, nq, "x")
        ir_fn = lambda p: ir_estandar(h, J, p)          # noqa: E731
        E_obj = E_capa
    else:
        escala = _escala_xy(sp, rng)
        h, J, _ = qb.a_ising(sp.Qc / escala, sp.const_costo / escala)
        E_capa = qb.energias_todas(sp.Qc / escala, sp.const_costo / escala)
        # ruta inicial al azar (semilla fija): no le regalamos al QAOA una ruta ya buena
        orden0 = [int(v) for v in rng.permutation(n)]
        sim = SimuladorQAOA(E_capa, nq, "swap", n=n, orden_inicial=orden0)
        ir_fn = lambda p: ir_xy(h, J, n, p, orden0)      # noqa: E731
        E_obj = E_capa

    motor_eval = None
    if motor == "qiskit":
        cache = {}

        def motor_eval(g, b):
            p = len(g)
            if p not in cache:
                cache[p] = ir_fn(p)
            return probabilidades_qiskit(cache[p], g, b)

    res = entrenar(sim, E_obj, cfg, variante, sp.id, escala, motor_eval=motor_eval)
    res.orden_inicial = orden0 if variante == "xy" else None
    ir = ir_fn(res.p)
    backend = "numpy (statevector propio)"
    if motor in ("qiskit", "auto") and qiskit_disponible():
        muestras, backend = muestrear_qiskit(ir, res.gammas, res.betas, cfg.shots, int(rng.integers(1 << 30)))
    else:
        muestras = rng.choice(2 ** nq, size=cfg.shots, p=res.probs)
    res.backend_muestreo = backend
    return res, muestras, ir

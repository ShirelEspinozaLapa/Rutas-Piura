"""
Pruebas de consistencia del modelo. Se corren con:
    python tests/verificar.py        (sin dependencias extra)
    python -m pytest tests/          (si tienes pytest)

Lo importante que se verifica:
  - el QUBO reproduce el costo exacto de cada ruta y su mínimo global es una ruta válida
  - el Ising equivale al QUBO
  - el simulador rápido da el mismo estado que el circuito compuerta por compuerta
  - el mezclador XY nunca sale del espacio de rutas válidas
  - (si Qiskit está instalado) el circuito de Qiskit da el mismo statevector
"""
from __future__ import annotations

import itertools
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qrutas import cuantico as qc  # noqa: E402
from qrutas import qubo as qb  # noqa: E402
from qrutas.clustering import clustering_instancia  # noqa: E402
from qrutas.config import Config  # noqa: E402
from qrutas.costos import construir_instancia  # noqa: E402
from qrutas.datos import cargar_escenario  # noqa: E402
from qrutas.evaluacion import RutaPlan, simular_ruta  # noqa: E402

_CACHE = {}


def _instancia():
    if "inst" not in _CACHE:
        cfg = Config()
        with warnings.catch_warnings():
            warnings.simplefilter("error")          # un corredor que cruce el río sería un error de datos
            esc = cargar_escenario(cfg)
            inst = construir_instancia(esc, cfg)
        clusters, _ = clustering_instancia(inst)
        _CACHE["inst"], _CACHE["clusters"] = inst, clusters
        _CACHE["subs"] = [qb.construir_subproblema(inst, c, 0.4 * k) for k, c in enumerate(clusters)]
    return _CACHE["inst"], _CACHE["clusters"], _CACHE["subs"]


def test_datos():
    inst, _, _ = _instancia()
    esc = inst.esc
    assert len(esc.depositos) == 2 and len(esc.comunidades) == 23
    for n in esc.nodos:
        assert -5.6 < n.lat < -4.8 and -80.9 < n.lon < -80.1, n.nombre
        if n.tipo == "comunidad":
            assert n.codigo_inei.startswith("2001"), n.nombre       # provincia de Piura (ubigeo 2001)


def test_red_y_accesibilidad():
    inst, _, _ = _instancia()
    G = inst.red.G
    for n in inst.esc.comunidades:
        assert G.degree(n.idx) >= 1, n.nombre
    assert any(d["bloqueada"] for _, _, d in G.edges(data=True))
    cam = inst.mats["camion"]
    for i, t in inst.tipo_por_nodo.items():
        if t == "camion":
            d = inst.deposito_por_nodo[i]
            assert cam.alcanzable[d, i] and cam.alcanzable[i, d]
    for n in inst.esc.comunidades:                       # comunidad cortada => solo helicóptero
        if n.acceso_terrestre == "cortado":
            assert inst.tipo_por_nodo[n.idx] == "helicoptero"


def test_clusters_respetan_capacidad_y_qubits():
    inst, clusters, _ = _instancia()
    from qrutas.config import TIPOS
    vistos = []
    for c in clusters:
        assert 1 <= len(c.nodos) <= inst.cfg.max_paradas
        assert inst.demanda[c.nodos].sum() <= TIPOS[c.tipo].capacidad_kg + 1e-9
        vistos += c.nodos
    assert sorted(vistos) == sorted(inst.tipo_por_nodo)


def test_qubo_reproduce_costo_exacto():
    _, _, subs = _instancia()
    for sp in subs:
        for pm in itertools.permutations(range(sp.n)):
            e = qb.energia(sp.Q, sp.const, qb.perm_a_x(pm, sp.n))
            j = qb.costo_exacto(sp, pm)
            if sp.n <= 3:
                assert abs(e - j) < 1e-8 * max(1, abs(j)), (sp.id, pm, e, j)
            else:                                        # n = 4: la ventana usa una cota inferior
                assert e <= j + 1e-8 * max(1, abs(j)), (sp.id, pm, e, j)


def test_minimo_global_es_ruta_valida():
    _, _, subs = _instancia()
    for sp in subs:
        E = qb.energias_todas(sp.Q, sp.const)
        x = qb.indice_a_x(int(np.argmin(E)), sp.nq)
        assert qb.x_a_perm(x, sp.n) is not None, sp.id


def test_ising_equivale_a_qubo():
    _, _, subs = _instancia()
    rng = np.random.default_rng(0)
    for sp in subs:
        h, J, off = qb.a_ising(sp.Q, sp.const)
        E = qb.energias_todas(sp.Q, sp.const)
        for k in rng.integers(0, 2 ** sp.nq, 40):
            z = 1 - 2 * qb.indice_a_x(int(k), sp.nq)
            assert abs(off + h @ z + z @ J @ z - E[k]) < 1e-8 * max(1, abs(E[k]))


def test_hamiltoniano_de_intercambio():
    X = np.array([[0, 1], [1, 0]], complex)
    Y = np.array([[0, -1j], [1j, 0]], complex)
    H = sum(s * np.kron(np.kron(np.kron(*[X if c == "X" else Y for c in P[:2]]),
                                *[X if c == "X" else Y for c in P[2:3]]), X if P[3] == "X" else Y)
            for P, s in qc.TERMINOS_INTERCAMBIO) / 8
    objetivo = np.zeros((16, 16), complex)
    objetivo[0b1100, 0b0011] = objetivo[0b0011, 0b1100] = 1
    assert np.allclose(H, objetivo)


def _sim_y_ir(sp, variante, p, orden0):
    if variante == "estandar":
        h, J, _ = qb.a_ising(sp.Q / sp.A, sp.const / sp.A)
        E = qb.energias_todas(sp.Q / sp.A, sp.const / sp.A)
        return qc.SimuladorQAOA(E, sp.nq, "x"), qc.ir_estandar(h, J, p)
    h, J, _ = qb.a_ising(sp.Qc, sp.const_costo)
    E = qb.energias_todas(sp.Qc, sp.const_costo)
    return (qc.SimuladorQAOA(E, sp.nq, "swap", n=sp.n, orden_inicial=orden0),
            qc.ir_xy(h, J, sp.n, p, orden0))


def test_simulador_rapido_igual_a_circuito():
    _, _, subs = _instancia()
    rng = np.random.default_rng(1)
    elegidos = sorted([s for s in subs if s.n >= 3], key=lambda s: s.nq)
    for sp in (elegidos[0], elegidos[-1]):
        orden0 = list(rng.permutation(sp.n))
        for var in ("estandar", "xy"):
            sim, ir = _sim_y_ir(sp, var, 2, orden0)
            g, b = rng.uniform(0, 2, 2), rng.uniform(0, 1.5, 2)
            fid = abs(np.vdot(sim.estado(g, b), qc.simular_ir(ir, g, b))) ** 2
            assert fid > 1 - 1e-10, (sp.id, var, fid)


def test_xy_preserva_factibilidad():
    _, _, subs = _instancia()
    sp = max(subs, key=lambda s: s.nq)
    fac = qb.mascara_factibles(sp.n)
    sim, _ = _sim_y_ir(sp, "xy", 3, list(range(sp.n))[::-1])
    P = sim.probabilidades([0.7, 1.3, 0.2], [0.4, 0.9, 1.2])
    assert abs(P[fac].sum() - 1) < 1e-10


def test_costo_qubo_igual_a_evaluacion():
    inst, clusters, subs = _instancia()
    for c, sp in zip(clusters, subs):
        orden = list(range(sp.n))
        ej = simular_ruta(inst, RutaPlan(c.tipo, c.deposito, [c.nodos[i] for i in orden]), sp.inicio_h)
        assert abs(ej.J - qb.costo_exacto(sp, orden)) < 1e-9 * max(1, ej.J), c.id


def test_qiskit_equivalente():
    if not qc.qiskit_disponible():
        print("   (Qiskit no está instalado: se omite la comparación con Qiskit)")
        return
    _, _, subs = _instancia()
    rng = np.random.default_rng(2)
    sp = sorted([s for s in subs if s.n == 3], key=lambda s: s.id)[0]
    for var in ("estandar", "xy"):
        _, ir = _sim_y_ir(sp, var, 2, list(rng.permutation(sp.n)))
        g, b = rng.uniform(0, 2, 2), rng.uniform(0, 1.5, 2)
        p_np = np.abs(qc.simular_ir(ir, g, b)) ** 2
        p_qk = qc.probabilidades_qiskit(ir, g, b)
        assert np.allclose(p_np, p_qk, atol=1e-9), var


if __name__ == "__main__":
    fallas = 0
    for nombre, f in sorted((k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)):
        try:
            f()
            print(f"OK    {nombre}")
        except AssertionError as e:
            fallas += 1
            print(f"FALLA {nombre}: {e}")
    print("\nTodo en orden." if not fallas else f"\n{fallas} prueba(s) fallaron.")
    sys.exit(1 if fallas else 0)

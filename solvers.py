"""Rutas de emergencia en Piura (El Niño): solver clásico vs QAOA híbrido."""
import time
import networkx as nx
import numpy as np
from scipy.optimize import minimize
from qiskit.circuit.library import QAOAAnsatz
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp, Statevector

# Datos ILUSTRATIVOS: reemplazar por datos reales (INDECI, SENAMHI, MTC).
NODOS = {
    "Piura": (-5.19, -80.63), "Castilla": (-5.20, -80.66),
    "Catacaos": (-5.27, -80.68), "Sullana": (-4.90, -80.69),
    "Paita": (-5.09, -81.11), "Tambogrande": (-4.93, -80.34),
    "Chulucanas": (-5.10, -80.16),
}
# (origen, destino, minutos en condiciones normales, riesgo de inundación 0-1)
TRAMOS = [
    ("Piura", "Castilla", 10, 0.5), ("Piura", "Catacaos", 20, 0.9),
    ("Piura", "Sullana", 45, 0.6), ("Castilla", "Sullana", 40, 0.4),
    ("Sullana", "Paita", 50, 0.3), ("Piura", "Paita", 70, 0.5),
    ("Sullana", "Tambogrande", 35, 0.7), ("Piura", "Tambogrande", 55, 0.6),
    ("Piura", "Chulucanas", 60, 0.8), ("Tambogrande", "Chulucanas", 40, 0.9),
]


def construir_grafo(nivel, umbral=0.8):
    """nivel: intensidad de El Niño (0-1). Tramos con riesgo*nivel >= umbral se bloquean."""
    G, bloqueados = nx.Graph(), []
    G.add_nodes_from(NODOS)
    for a, b, t, r in TRAMOS:
        p = r * nivel
        if p >= umbral:
            bloqueados.append((a, b))
        else:
            G.add_edge(a, b, costo=t * (1 + 2 * p))  # minutos con congestión/lodo
    return G, bloqueados


def dijkstra(G, s, t):
    t0 = time.time()
    ruta = nx.shortest_path(G, s, t, weight="costo")
    return ruta, nx.path_weight(G, ruta, "costo"), time.time() - t0


def _cuadrado(Q, terminos, c, A):
    """Suma A*(sum a_i x_i - c)^2 a la matriz QUBO Q (x binaria)."""
    items = list(terminos.items())
    for i, (u, a) in enumerate(items):
        Q[(u, u)] = Q.get((u, u), 0) + A * (a * a - 2 * c * a)
        for v, b in items[i + 1:]:
            k = tuple(sorted((u, v)))
            Q[k] = Q.get(k, 0) + 2 * A * a * b


def construir_qubo(G, s, t, A=1.5):
    aristas = list(G.edges)
    cmax = max(d["costo"] for *_, d in G.edges(data=True))
    vars_ = [("e", e) for e in aristas] + [("y", v) for v in G.nodes if v not in (s, t)]
    idx = {v: i for i, v in enumerate(vars_)}
    Q = {}
    for e in aristas:  # minimizar costo normalizado
        i = idx[("e", e)]
        Q[(i, i)] = Q.get((i, i), 0) + G.edges[e]["costo"] / cmax
    for v in G.nodes:  # grado 1 en s,t; grado 2*y en intermedios
        term = {idx[("e", e)]: 1 for e in aristas if v in e}
        if v in (s, t):
            _cuadrado(Q, term, 1, A)
        else:
            term[idx[("y", v)]] = -2
            _cuadrado(Q, term, 0, A)
    return Q, vars_, aristas


def qubo_a_ising(Q, n):
    h, J = np.zeros(n), {}
    for (i, j), w in Q.items():
        if i == j:
            h[i] -= w / 2
        else:
            J[(i, j)] = J.get((i, j), 0) + w / 4
            h[i] -= w / 4
            h[j] -= w / 4
    ops = [("Z", [i], h[i]) for i in range(n)] + [("ZZ", [i, j], w) for (i, j), w in J.items()]
    return SparsePauliOp.from_sparse_list(ops, num_qubits=n)


def qaoa(G, s, t, reps=2, maxiter=120, semilla=7):
    t0 = time.time()
    Q, vars_, aristas = construir_qubo(G, s, t)
    n = len(vars_)
    H = qubo_a_ising(Q, n)
    # Descomponer evita exponenciar matrices (mucho más rápido en simulador)
    ansatz = QAOAAnsatz(cost_operator=H, reps=reps).decompose(reps=3)
    est = StatevectorEstimator()
    f = lambda th: float(est.run([(ansatz, H, th)]).result()[0].data.evs)
    th0 = np.random.default_rng(semilla).uniform(0, np.pi, ansatz.num_parameters)
    res = minimize(f, th0, method="COBYLA", options={"maxiter": maxiter})
    probs = Statevector(ansatz.assign_parameters(res.x)).probabilities_dict()
    mejor, prob_ok = None, 0.0
    for bits, p in sorted(probs.items(), key=lambda kv: -kv[1])[:200]:
        bits = bits[::-1]  # Qiskit es little-endian
        sel = [aristas[i] for i in range(len(aristas)) if bits[i] == "1"]
        H_ = nx.Graph()
        H_.add_nodes_from(G.nodes)
        H_.add_edges_from(sel)
        if nx.has_path(H_, s, t):
            ruta = nx.shortest_path(H_, s, t)
            costo = nx.path_weight(G, ruta, "costo")
            prob_ok += p
            if mejor is None or costo < mejor[1]:
                mejor = (ruta, costo)
    return {"ruta": mejor[0] if mejor else None, "costo": mejor[1] if mejor else None,
            "qubits": n, "prob_valida": prob_ok, "evals": res.nfev, "tiempo": time.time() - t0}


def qaoa_seguro(*args, **kwargs):
    """Ejecuta qaoa() en un hilo con pila grande (evita fallos dentro de Streamlit)."""
    import threading
    salida = {}

    def tarea():
        try:
            salida["r"] = qaoa(*args, **kwargs)
        except Exception as e:  # noqa
            salida["e"] = e

    anterior = threading.stack_size(256 * 1024 * 1024)
    try:
        h = threading.Thread(target=tarea)
        h.start(); h.join()
    finally:
        threading.stack_size(anterior)
    if "e" in salida:
        raise salida["e"]
    return salida["r"]

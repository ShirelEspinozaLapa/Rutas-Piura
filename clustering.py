"""
Fase clásica: reducción de dimensionalidad por clustering capacitado.

Cada grupo (tipo de vehículo + depósito) se parte en clusters que cumplen:
  - demanda total <= capacidad del vehículo
  - a lo más `max_paradas` comunidades (presupuesto de qubits: n paradas -> n^2 qubits)

Método: k-medoides capacitado. La asignación de comunidades a medoides se
resuelve exactamente como un problema entero (scipy.optimize.milp / HiGHS) y los
medoides se actualizan hasta que no cambian. La distancia es espacio-temporal:

    d'(i, j) = c(i, j) + peso_urgencia * |u_i - u_j| * c_media

donde c es el costo compuesto de viaje del modo (red vial para camiones, línea
recta para helicópteros) y u_i = 1 si la comunidad necesita medicinas con
ventana. Así las comunidades urgentes tienden a quedar juntas y su ruta sale
primero en la programación.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .config import TIPOS


@dataclass
class Cluster:
    id: str
    tipo: str
    deposito: int
    nodos: list


def _matriz_distancias(inst, grupo):
    comp = inst.mats[grupo.tipo].compuesto(inst.cfg)
    nodos = grupo.nodos
    D = comp[np.ix_(nodos, nodos)]
    D = (D + D.T) / 2
    u = np.array([inst.esc.nodos[i].prioridad_medica for i in nodos], float)
    media = D[np.triu_indices(len(nodos), 1)].mean() if len(nodos) > 1 else 1.0
    return D + inst.cfg.peso_urgencia * np.abs(u[:, None] - u[None, :]) * media, comp


def _asignar(D, medoides, q, Q, S):
    n, k = D.shape[0], len(medoides)
    c = D[:, medoides].ravel()
    filas = []
    A_uno = np.zeros((n, n * k))
    for i in range(n):
        A_uno[i, i * k:(i + 1) * k] = 1
    filas.append(LinearConstraint(A_uno, 1, 1))
    A_cap = np.zeros((k, n * k))
    A_card = np.zeros((k, n * k))
    for j in range(k):
        A_cap[j, j::k] = q
        A_card[j, j::k] = 1
    filas.append(LinearConstraint(A_cap, -np.inf, Q))
    filas.append(LinearConstraint(A_card, -np.inf, S))
    lb = np.zeros(n * k)
    for j, m in enumerate(medoides):       # cada medoide pertenece a su propio cluster
        lb[m * k + j] = 1
    res = milp(c, constraints=filas, integrality=np.ones(n * k), bounds=Bounds(lb, np.ones(n * k)))
    if not res.success:
        return None
    z = res.x.reshape(n, k) > 0.5
    return [list(np.flatnonzero(z[:, j])) for j in range(k)]


def kmedoides_capacitado(inst, grupo, etiqueta: str) -> list:
    cfg = inst.cfg
    tv = TIPOS[grupo.tipo]
    nodos = grupo.nodos
    n = len(nodos)
    if n == 0:
        return []
    q = np.array([inst.demanda[i] for i in nodos])
    S = cfg.max_paradas
    D, comp = _matriz_distancias(inst, grupo)
    dist_dep = comp[grupo.deposito, nodos]
    k = max(math.ceil(n / S), math.ceil(q.sum() / tv.capacidad_kg))
    while k <= n:
        # inicialización determinista: el más lejano al depósito y luego el más lejano a los elegidos
        med = [int(np.argmax(dist_dep))]
        while len(med) < k:
            med.append(int(np.argmax(np.min(D[:, med], axis=1))))
        asign = None
        for _ in range(cfg.iter_kmedoides):
            asign = _asignar(D, med, q, tv.capacidad_kg, S)
            if asign is None:
                break
            nuevos = [int(c[np.argmin(D[np.ix_(c, c)].sum(axis=0))]) for c in asign]
            if nuevos == med:
                break
            med = nuevos
        if asign is not None:
            break
        k += 1
    clusters = []
    for j, c in enumerate(a for a in asign if a):
        clusters.append(Cluster(id=f"{etiqueta}{j + 1}", tipo=grupo.tipo, deposito=grupo.deposito,
                                nodos=[nodos[i] for i in c]))
    return clusters


def _costo_proxy(inst, clusters) -> float:
    """Costo de la solución armando cada ruta con la heurística del vecino más cercano.
    Es barato (no resuelve el TSP de cada cluster: eso queda para la fase cuántica)."""
    from .evaluacion import RutaPlan, evaluar
    from .qubo import construir_subproblema, ruta_vecino_mas_cercano

    planes = []
    for c in clusters:
        sp = construir_subproblema(inst, c, 0.0, con_qubo=False)
        planes.append(RutaPlan(c.tipo, c.deposito, [c.nodos[i] for i in ruta_vecino_mas_cercano(sp)], c.id))
    return evaluar(inst, planes).J


def refinar_clusters(inst, clusters, max_pasadas: int = 50):
    """Búsqueda local sobre la partición: mover o intercambiar comunidades entre clusters
    del mismo grupo (respetando capacidad y máximo de paradas) mientras baje el costo proxy."""
    cfg = inst.cfg
    actual = [Cluster(c.id, c.tipo, c.deposito, list(c.nodos)) for c in clusters]
    J = _costo_proxy(inst, actual)
    J0, movimientos = J, 0
    for _ in range(max_pasadas):
        mejora = False
        for a in range(len(actual)):
            for b in range(len(actual)):
                A, B = actual[a], actual[b]
                if a == b or (A.tipo, A.deposito) != (B.tipo, B.deposito):
                    continue
                cap = TIPOS[A.tipo].capacidad_kg
                cA, cB = inst.demanda[A.nodos].sum(), inst.demanda[B.nodos].sum()
                candidatos = []
                for i in A.nodos:          # mover i de A a B
                    if len(A.nodos) > 1 and len(B.nodos) < cfg.max_paradas and cB + inst.demanda[i] <= cap:
                        candidatos.append(([i], []))
                for i in A.nodos:          # intercambiar i (A) con j (B)
                    for j in B.nodos:
                        if (cA - inst.demanda[i] + inst.demanda[j] <= cap
                                and cB - inst.demanda[j] + inst.demanda[i] <= cap):
                            candidatos.append(([i], [j]))
                for sale_a, sale_b in candidatos:
                    nuevo = [Cluster(c.id, c.tipo, c.deposito, list(c.nodos)) for c in actual]
                    nuevo[a].nodos = [x for x in nuevo[a].nodos if x not in sale_a] + sale_b
                    nuevo[b].nodos = [x for x in nuevo[b].nodos if x not in sale_b] + sale_a
                    Jn = _costo_proxy(inst, nuevo)
                    if Jn < J - 1e-9:
                        actual, J, mejora = nuevo, Jn, True
                        movimientos += 1
                        break
                if mejora:
                    break
            if mejora:
                break
        if not mejora:
            break
    return actual, dict(J_proxy_inicial=J0, J_proxy_final=J, movimientos=movimientos)


def clustering_instancia(inst, refinar: bool = True):
    clusters = []
    for g in inst.grupos:
        # ids que no se confunden con los vehículos (H1, T1...) ni con las comunidades (C01...)
        dep = inst.esc.nodos[g.deposito]
        cod = {"D1": "A", "D2": "U"}.get(dep.id, dep.id)       # A = aeropuerto, U = UDEP
        pref = "Aéreo-" if g.tipo == "helicoptero" else f"Terr-{cod}"
        clusters += kmedoides_capacitado(inst, g, pref)
    info = dict(J_proxy_inicial=None, J_proxy_final=None, movimientos=0)
    if refinar:
        clusters, info = refinar_clusters(inst, clusters)
    return clusters, info


def metricas_reduccion(inst, clusters) -> dict:
    """Cuántos qubits haría falta sin y con la fase clásica (codificación por posiciones)."""
    N = len(inst.comunidades)
    K = len(inst.cfg.flota)
    sin_clustering_tsp = N * N
    sin_clustering_vrp = (N + K - 1) ** 2          # tour gigante con copias del depósito
    con = [len(c.nodos) ** 2 for c in clusters]
    return dict(comunidades=N, vehiculos=K, qubits_tsp_completo=sin_clustering_tsp,
                qubits_vrp_completo=sin_clustering_vrp, n_subproblemas=len(clusters),
                qubits_max_subproblema=max(con) if con else 0, qubits_por_subproblema=con,
                dimension_espacio_completo=f"2^{sin_clustering_vrp}",
                dimension_max_subproblema=f"2^{max(con) if con else 0}")

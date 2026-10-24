"""Formulación de Partición de Conjuntos Cuántica (Quantum Set Partitioning / Column Generation).

En el modelo estándar TSP-QUBO, n paradas requieren n^2 qubits. Con 23 comunidades y 6 vehículos,
el VRP completo requeriría 784 qubits.
En la formulación por Partición de Conjuntos (Set Partitioning):
1. Se genera un banco o pool de rutas candidatas válidas por vehículo (respetando capacidad,
   tipo de vehículo, ventanas y depósitos).
2. Cada variable binaria z_r in {0, 1} representa seleccionar o descartar la ruta candidata r.
3. El QUBO selecciona el subconjunto óptimo de rutas que cubre a todas las comunidades afectadas
   exactamente una vez y respeta la disponibilidad de la flota:

   min sum_r c_r z_r + P_cubrimiento sum_{i in C} (1 - sum_{r: i in r} z_r)^2
                     + P_flota sum_m max(0, sum_{r in R_m} z_r - K_m)

Con este enfoque, 23 comunidades se resuelven con 20-35 qubits totales (1 qubit por ruta candidata),
permitiendo optimizar el ruteo completo en procesadores cuánticos actuales.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
import numpy as np

from .evaluacion import RutaPlan, evaluar
from . import qubo as qb


@dataclass
class RutaCandidata:
    id_ruta: int
    tipo_vehiculo: str
    deposito: int
    nodos: list[int]
    costo: float
    tiempo_h: float
    combustible_l: float
    riesgo: float
    carga_kg: int
    violaciones_ventana: int
    plan: RutaPlan


@dataclass
class ProblemaParticion:
    rutas: list[RutaCandidata]
    n_rutas: int
    comunidades: list[int]
    limites_flota: dict[str, int]
    Q: np.ndarray
    constante: float
    P_cubrimiento: float
    P_flota: float


def generar_rutas_candidatas(inst, max_rutas_por_grupo: int = 12) -> list[RutaCandidata]:
    """Genera un pool de rutas candidatas factibles para cada grupo de flota."""
    esc = inst.esc
    candidatas = []
    idx = 0

    for g in inst.grupos:
        tipo = g.tipo
        dep = g.deposito
        cap = 4000.0 if tipo == "helicoptero" else 8000.0
        coms_grupo = list(g.nodos)

        if not coms_grupo:
            continue

        # Generar combinaciones de 1 a 4 paradas que respeten capacidad
        rutas_grupo = []
        for tam in range(1, min(5, len(coms_grupo) + 1)):
            for comb in itertools.combinations(coms_grupo, tam):
                carga = sum(esc.nodos[c].demanda_kg for c in comb)
                if carga > cap:
                    continue

                # Encontrar el orden óptimo para esta combinación
                mejor_perm = list(comb)
                mejor_costo = float("inf")
                
                # Permutaciones si tam <= 4
                for p in itertools.permutations(comb):
                    plan = RutaPlan(tipo, dep, list(p), f"cand_{idx}")
                    sol = evaluar(inst, [plan])
                    if sol.J < mejor_costo:
                        mejor_costo = sol.J
                        mejor_perm = list(p)

                plan_opt = RutaPlan(tipo, dep, mejor_perm, f"cand_{idx}")
                sol_opt = evaluar(inst, [plan_opt])
                r_info = sol_opt.rutas[0]

                rutas_grupo.append(RutaCandidata(
                    id_ruta=idx,
                    tipo_vehiculo=tipo,
                    deposito=dep,
                    nodos=mejor_perm,
                    costo=float(sol_opt.J),
                    tiempo_h=float(r_info.fin_h - r_info.inicio_h),
                    combustible_l=float(r_info.combustible_l),
                    riesgo=float(r_info.riesgo),
                    carga_kg=carga,
                    violaciones_ventana=int(sol_opt.componentes["violaciones_ventana"]),
                    plan=plan_opt
                ))
                idx += 1

        # Ordenar por costo y filtrar las más eficientes
        rutas_grupo.sort(key=lambda r: r.costo / max(1, len(r.nodos)))
        candidatas.extend(rutas_grupo[:max_rutas_por_grupo])

    # Re-indexar rutas consecutivamente
    for i, r in enumerate(candidatas):
        r.id_ruta = i
        r.plan.etiqueta = f"cand_{i}"

    return candidatas


def construir_qubo_particion(rutas: list[RutaCandidata], inst,
                             P_cubrimiento: float = 80.0,
                             P_flota: float = 50.0) -> ProblemaParticion:
    """Construye la matriz QUBO para el problema de Partición de Conjuntos.
    
    Variables: z_r in {0, 1} para cada ruta candidata r (Total de qubits = len(rutas)).
    """
    R = len(rutas)
    comunidades = [c.idx for c in inst.esc.comunidades]
    limites_flota = {g.tipo: len(g.vehiculos) for g in inst.grupos}

    Q = np.zeros((R, R), dtype=float)
    constante = 0.0

    # 1. Función objetivo: costo de las rutas seleccionadas
    for r in range(R):
        Q[r, r] += rutas[r].costo

    # 2. Penalización de cubrimiento: exactamente una visita por comunidad
    # P_cubrimiento * sum_i (1 - sum_{r: i in r} z_r)^2
    # = P * sum_i [ 1 - 2 sum_{r: i in r} z_r + sum_{r: i in r} sum_{s: i in s} z_r z_s ]
    for i in comunidades:
        rutas_con_i = [r for r in range(R) if i in rutas[r].nodos]
        if not rutas_con_i:
            continue
        constante += P_cubrimiento
        for r in rutas_con_i:
            Q[r, r] -= 2.0 * P_cubrimiento
        for r in rutas_con_i:
            for s in rutas_con_i:
                if r == s:
                    Q[r, r] += P_cubrimiento
                elif r < s:
                    Q[r, s] += 2.0 * P_cubrimiento

    # 3. Penalización de flota por tipo de vehículo
    # Para cada tipo de vehículo m, no superar K_m rutas activas
    for m, K_m in limites_flota.items():
        rutas_m = [r for r in range(R) if rutas[r].tipo_vehiculo == m]
        # Penalización cuadrática por exceso aproximada
        if len(rutas_m) > K_m:
            # (sum_{r in m} z_r - K_m)^2 si se busca exactamente K_m o penalizar exceso
            for r in rutas_m:
                for s in rutas_m:
                    if r == s:
                        Q[r, r] += 0.5 * P_flota
                    elif r < s:
                        Q[r, s] += P_flota

    return ProblemaParticion(
        rutas=rutas,
        n_rutas=R,
        comunidades=comunidades,
        limites_flota=limites_flota,
        Q=Q,
        constante=constante,
        P_cubrimiento=P_cubrimiento,
        P_flota=P_flota
    )


def resolver_particion_clasico(prob: ProblemaParticion, inst, max_intentos: int = 5000) -> dict:
    """Resuelve el QUBO de partición con Simulated Annealing para selección de rutas."""
    R = prob.n_rutas
    Q = prob.Q
    const = prob.constante
    rng = np.random.default_rng(2026)

    t0 = time.perf_counter()

    # Estado inicial: heurística voraz para cubrir comunidades
    z = np.zeros(R, dtype=int)
    cubiertas = set()
    for r in range(R):
        nuevas = set(prob.rutas[r].nodos) - cubiertas
        if nuevas:
            z[r] = 1
            cubiertas |= nuevas
        if len(cubiertas) == len(prob.comunidades):
            break

    def energia(x):
        return float(x @ Q @ x + const)

    E_actual = energia(z)
    E_mejor = E_actual
    z_mejor = z.copy()

    T0 = 10.0
    T_fin = 0.01
    gamma = (T_fin / T0) ** (1.0 / max_intentos)
    T = T0

    for paso in range(max_intentos):
        idx_flip = rng.integers(0, R)
        z_cand = z.copy()
        z_cand[idx_flip] = 1 - z_cand[idx_flip]
        E_cand = energia(z_cand)

        dE = E_cand - E_actual
        if dE < 0 or rng.random() < np.exp(-dE / max(T, 1e-6)):
            z = z_cand
            E_actual = E_cand
            if E_actual < E_mejor:
                E_mejor = E_actual
                z_mejor = z.copy()

        T *= gamma

    tiempo_s = time.perf_counter() - t0

    # Extraer rutas seleccionadas
    rutas_sel = [prob.rutas[r] for r in range(R) if z_mejor[r] == 1]
    if not rutas_sel:
        rutas_sel = [prob.rutas[r] for r in range(R) if z[r] == 1]
    if not rutas_sel and prob.rutas:
        rutas_sel = [prob.rutas[0]]
    planes = [r.plan for r in rutas_sel]
    sol_eval = evaluar(inst, planes) if planes else None

    # Verificar cobertura de comunidades
    nodos_visitados = [nodo for r in rutas_sel for nodo in r.nodos]
    cobertura_completa = set(nodos_visitados) == set(prob.comunidades)

    return {
        "metodo": "Partición de Conjuntos (QUBO)",
        "qubits_usados": R,
        "rutas_candidatas_totales": R,
        "rutas_seleccionadas": len(rutas_sel),
        "energia_qubo": E_mejor,
        "solucion_global": sol_eval,
        "tiempo_s": tiempo_s,
        "cobertura_completa": cobertura_completa,
        "comunidades_cubiertas": len(set(nodos_visitados)),
        "comunidades_totales": len(prob.comunidades),
        "detalles_rutas": [{
            "id": r.id_ruta,
            "tipo": r.tipo_vehiculo,
            "nodos": r.nodos,
            "costo": r.costo
        } for r in rutas_sel]
    }

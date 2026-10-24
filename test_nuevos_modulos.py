"""Pruebas de verificación de los nuevos módulos cuánticos y de arquitectura."""
from __future__ import annotations

import os
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.join(AQUI, "..")
if RAIZ not in sys.path:
    sys.path.insert(0, RAIZ)

import numpy as np
from qrutas.config import Config
from qrutas.datos import cargar_escenario
from qrutas.costos import construir_instancia
from qrutas.clustering import clustering_instancia
from qrutas import qubo as qb
from qrutas import cuantico
from qrutas.cuantico_warmstart import calcular_inicializacion_warm_start, ejecutar_warm_start_qaoa
from qrutas.cuantico_ring_xy import ejecutar_ring_xy, contar_compuertas_ring_vs_completo
from qrutas.dwave_solver import qubo_a_bqm, resolver_quantum_annealing
from qrutas.mitigacion_ruido import evaluar_mitigacion_subproblema
from qrutas.cuantico_particion import generar_rutas_candidatas, construir_qubo_particion, resolver_particion_clasico
from qrutas.solvers import listar_solvers, obtener_solver
from qrutas.paralelo import resolver_subproblemas_en_paralelo
from qrutas.config_tipada import cargar_escenario_json, EscenarioEmergencia


def test_warm_start():
    cfg = Config()
    esc = cargar_escenario(cfg)
    inst = construir_instancia(esc, cfg)
    clusters, _ = clustering_instancia(inst)
    sp = qb.construir_subproblema(inst, clusters[0], 0.0)
    prep = cuantico.preparar(sp)

    p_cont, thetas = calcular_inicializacion_warm_start(sp)
    assert len(p_cont) == sp.nq
    assert len(thetas) == sp.nq

    res_ws = ejecutar_warm_start_qaoa(sp, prep, p=1, max_iter=30)
    assert res_ws.p_factible > 0.0
    print("OK    test_warm_start")


def test_ring_xy():
    cfg = Config()
    esc = cargar_escenario(cfg)
    inst = construir_instancia(esc, cfg)
    clusters, _ = clustering_instancia(inst)
    sp = qb.construir_subproblema(inst, clusters[0], 0.0)
    prep = cuantico.preparar(sp)

    cnot_ring, cnot_all = contar_compuertas_ring_vs_completo(sp.n, 1)
    assert cnot_ring <= cnot_all

    res_ring = ejecutar_ring_xy(sp, prep, p=1, max_iter=25)
    assert res_ring.p_factible == 1.0  # El Ring XY siempre preserva factibilidad
    print("OK    test_ring_xy")


def test_dwave_bqm():
    cfg = Config()
    esc = cargar_escenario(cfg)
    inst = construir_instancia(esc, cfg)
    clusters, _ = clustering_instancia(inst)
    sp = qb.construir_subproblema(inst, clusters[0], 0.0)
    prep = cuantico.preparar(sp)

    bqm = qubo_a_bqm(sp.Q, sp.const)
    assert bqm.num_variables == sp.nq

    res_dw = resolver_quantum_annealing(sp, prep, num_reads=100)
    assert res_dw["qubits_usados"] == sp.nq
    assert len(res_dw["orden_mejor"]) == sp.n
    print("OK    test_dwave_bqm")


def test_mitigacion_ruido():
    cfg = Config()
    esc = cargar_escenario(cfg)
    inst = construir_instancia(esc, cfg)
    clusters, _ = clustering_instancia(inst)
    sp = qb.construir_subproblema(inst, clusters[0], 0.0)
    prep = cuantico.preparar(sp)

    # QAOA base
    res_qaoa, _, _ = cuantico.ejecutar_qaoa(sp, prep, cfg, "estandar", np.random.default_rng(2026), "numpy")
    reporte = evaluar_mitigacion_subproblema(sp, prep, res_qaoa)
    assert reporte.nq == sp.nq
    assert len(reporte.energias_por_escala) == 3
    print("OK    test_mitigacion_ruido")


def test_set_partitioning():
    cfg = Config()
    esc = cargar_escenario(cfg)
    inst = construir_instancia(esc, cfg)

    rutas_cand = generar_rutas_candidatas(inst, max_rutas_por_grupo=6)
    assert len(rutas_cand) > 0

    prob = construir_qubo_particion(rutas_cand, inst)
    assert prob.n_rutas == len(rutas_cand)

    res = resolver_particion_clasico(prob, inst, max_intentos=500)
    assert res["rutas_candidatas_totales"] == prob.n_rutas
    assert res["solucion_global"] is not None
    print("OK    test_set_partitioning")


def test_solvers_registry():
    solvers = listar_solvers()
    assert len(solvers) >= 5
    s = obtener_solver("qaoa_estandar")
    assert s.es_cuantico is True
    print("OK    test_solvers_registry")


def test_config_tipada():
    ruta_json = os.path.join(RAIZ, "escenarios", "fen_catastrofico_2023.json")
    esc_val = cargar_escenario_json(ruta_json)
    assert len(esc_val.puentes) == 5
    assert esc_val.severidad_rio_factor == 1.5
    print("OK    test_config_tipada")


def test_paralelo():
    cfg = Config()
    esc = cargar_escenario(cfg)
    inst = construir_instancia(esc, cfg)
    clusters, _ = clustering_instancia(inst)
    sps = [qb.construir_subproblema(inst, c, 0.0) for c in clusters]

    def resolver_dummy(sp):
        return sp.id, {"n": sp.n, "nq": sp.nq}

    res_par, t_par = resolver_subproblemas_en_paralelo(sps, resolver_dummy, max_workers=2)
    assert len(res_par) == len(sps)
    print("OK    test_paralelo")


if __name__ == "__main__":
    test_warm_start()
    test_ring_xy()
    test_dwave_bqm()
    test_mitigacion_ruido()
    test_set_partitioning()
    test_solvers_registry()
    test_config_tipada()
    test_paralelo()
    print("\nTodos los módulos nuevos pasaron las pruebas satisfactoriamente.")

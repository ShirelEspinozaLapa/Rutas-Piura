"""Orquestación del flujo híbrido y de la comparación con métodos clásicos."""
from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass, field

import numpy as np

from . import clasicos, cuantico, hardware
from . import qubo as qb
from .clustering import clustering_instancia, metricas_reduccion
from .config import DIR_RESULTADOS, Config
from .costos import construir_instancia
from .datos import cargar_escenario
from .evaluacion import RutaPlan, evaluar, programar


def _log(msg, verbose=True):
    if verbose:
        print(msg, flush=True)


@dataclass
class Resultados:
    cfg: Config
    inst: object
    clusters: list
    subproblemas: list
    reduccion: dict
    sub: dict = field(default_factory=dict)          # id -> métricas por método
    soluciones: dict = field(default_factory=dict)   # nombre -> Solucion
    tiempos: dict = field(default_factory=dict)      # nombre -> segundos de cómputo
    extra: dict = field(default_factory=dict)


def _gap(E, E_opt):
    return 100.0 * (E - E_opt) / max(abs(E_opt), 1e-12)


def _primera_convergencia(hist, rel=1e-3):
    h = np.asarray(hist)
    objetivo = h.min() + rel * max(abs(h.min()), 1e-12)
    return int(np.argmax(h <= objetivo)) + 1


def _ruta_desde_muestras(sp, prep, muestras):
    """Mejor ruta factible entre los bitstrings medidos (post-selección)."""
    unicos, cuentas = np.unique(muestras, return_counts=True)
    fac = prep.factible[unicos]
    if fac.any():
        cand = unicos[fac]
        k = cand[np.argmin(prep.E_total[cand])]
        return qb.x_a_perm(qb.indice_a_x(int(k), sp.nq), sp.n), False
    # sin muestras factibles: reparación clásica (ruta vecino más cercano)
    return list(sp.info["orden_vecino_mas_cercano"]), True


def _metricas_qaoa(sp, prep, res, muestras, cfg):
    P = res.probs
    p_fac = float(P[prep.factible].sum())
    p_opt = float(P[prep.optimos].sum())
    orden, reparado = _ruta_desde_muestras(sp, prep, muestras)
    E_best = qb.energia(sp.Q, sp.const, qb.perm_a_x(orden, sp.n))
    rec = hardware.recursos_logicos(res.ir)
    # Se transpila solo el QAOA estándar: el XY tiene miles de compuertas y basta la estimación analítica.
    trans = (hardware.transpilar_ibm(res.ir, res.gammas, res.betas)
             if cfg.motor_cuantico != "numpy" and res.variante == "estandar" else None)
    qpu = hardware.estimar_qpu(rec, cfg, res.n_evaluaciones, cfg.shots, trans)
    return dict(
        orden=orden, reparado=reparado, E=E_best, gap_pct=_gap(E_best, prep.E_opt),
        p_factible=p_fac, p_optimo=p_opt,
        p_factible_muestras=float(prep.factible[muestras].mean()),
        p_optimo_muestras=float(prep.optimos[muestras].mean()),
        E_esperada=float(P @ prep.E_total), E_aleatoria=prep.E_aleatorio,
        cvar=float(res.valores_por_p[res.p] * res.escala),
        tts_disparos=clasicos._tts(p_opt), p=res.p, gammas=res.gammas.tolist(), betas=res.betas.tolist(),
        n_evaluaciones=res.n_evaluaciones, eval_convergencia=_primera_convergencia(res.historia),
        tiempo_s=res.tiempo_s, historia=[float(v * res.escala) for v in res.historia],
        hitos_p=res.hitos_p, backend_muestreo=res.backend_muestreo, recursos=rec, transpilado=trans,
        escala=res.escala, orden_inicial=getattr(res, "orden_inicial", None),
        qpu=qpu, valores_por_p={int(k): float(v * res.escala) for k, v in res.valores_por_p.items()},
    )


def ejecutar(cfg: Config | None = None, verbose: bool = True) -> Resultados:
    cfg = cfg or Config()
    rng = np.random.default_rng(cfg.semilla)
    motor = cfg.motor_cuantico
    if motor == "qiskit" and not cuantico.qiskit_disponible():
        raise RuntimeError("motor_cuantico='qiskit' pero Qiskit no está instalado (pip install qiskit qiskit-aer)")

    # ------------------------------------------------------------------ fase clásica
    t0 = time.perf_counter()
    esc = cargar_escenario(cfg)
    inst = construir_instancia(esc, cfg)
    clusters, info_refinado = clustering_instancia(inst)
    reduccion = metricas_reduccion(inst, clusters) | {"refinamiento": info_refinado}
    planes_nn = []
    for c in clusters:
        sp0 = qb.construir_subproblema(inst, c, 0.0)
        planes_nn.append(RutaPlan(c.tipo, c.deposito, [c.nodos[i] for i in sp0.info["orden_vecino_mas_cercano"]],
                                  c.id))
    salida = {r.plan.etiqueta: r.inicio_h for r in programar(inst, planes_nn)}
    subproblemas = [qb.construir_subproblema(inst, c, salida[c.id]) for c in clusters]
    t_clasica = time.perf_counter() - t0
    _log(f"[fase clásica] {len(esc.comunidades)} comunidades -> {len(clusters)} clusters "
         f"(máx. {reduccion['qubits_max_subproblema']} qubits; sin clustering: "
         f"{reduccion['qubits_vrp_completo']}) en {t_clasica:.2f} s", verbose)

    R = Resultados(cfg=cfg, inst=inst, clusters=clusters, subproblemas=subproblemas, reduccion=reduccion)
    R.extra["t_fase_clasica_s"] = t_clasica
    R.extra["salida_preliminar_h"] = salida
    R.extra["qiskit"] = cuantico.qiskit_disponible()

    # ------------------------------------------------------------------ subproblemas
    for sp in subproblemas:
        etiqueta = f"  {sp.id:6s} ({sp.tipo}, {sp.n} paradas, {sp.nq} qubits)"
        prep = cuantico.preparar(sp)
        fb = clasicos.fuerza_bruta(sp)
        d = dict(id=sp.id, tipo=sp.tipo, n=sp.n, nq=sp.nq, nodos=sp.nodos, inicio_h=sp.inicio_h, A=sp.A,
                 E_opt=prep.E_opt, J_opt_exacto=fb["J_optimo_exacto"],
                 exacto=dict(orden=fb["orden"], E=fb["E"], tiempo_s=fb["tiempo_s"],
                             n_evaluaciones=fb["n_evaluaciones"]))
        if sp.n == 1:
            for m in ("sa", "ga", "qaoa_estandar", "qaoa_xy"):
                d[m] = dict(orden=[0], E=fb["E"], gap_pct=0.0, trivial=True, tiempo_s=0.0, n_evaluaciones=0)
            R.sub[sp.id] = d
            _log(etiqueta + " -> trivial (una sola parada)", verbose)
            continue
        sas = [clasicos.sa_qubo(sp, prep, cfg, rng) for _ in range(cfg.repeticiones)]
        gas = [clasicos.ga_permutaciones(sp, prep, cfg, rng) for _ in range(cfg.repeticiones)]
        d["sa"] = dict(sas[0], gap_pct=_gap(sas[0]["E"], prep.E_opt),
                       p_optimo_medio=float(np.mean([s["p_optimo"] for s in sas])),
                       p_factible_medio=float(np.mean([s["p_factible"] for s in sas])),
                       tiempo_medio_s=float(np.mean([s["tiempo_s"] for s in sas])),
                       exito_corridas=float(np.mean([s["E"] <= prep.E_opt + 1e-9 for s in sas])))
        d["ga"] = dict(gas[0], gap_pct=_gap(gas[0]["E"], prep.E_opt),
                       exito_corridas=float(np.mean([g["E"] <= prep.E_opt + 1e-9 for g in gas])),
                       tiempo_medio_s=float(np.mean([g["tiempo_s"] for g in gas])))
        for var in ("estandar", "xy"):
            res, muestras, ir = cuantico.ejecutar_qaoa(sp, prep, cfg, var, rng, motor)
            res.ir = ir
            d[f"qaoa_{var}"] = _metricas_qaoa(sp, prep, res, muestras, cfg)
            d[f"qaoa_{var}"]["probs_top"] = _top_estados(sp, prep, res.probs)
        R.sub[sp.id] = d
        _log(etiqueta + f" -> exacto E={prep.E_opt:.3f} | SA gap {d['sa']['gap_pct']:.2f}% "
             f"P(opt) {d['sa']['p_optimo_medio']:.2f} | QAOA p={d['qaoa_estandar']['p']} "
             f"P(fact) {d['qaoa_estandar']['p_factible']:.3f} P(opt) {d['qaoa_estandar']['p_optimo']:.4f} "
             f"gap {d['qaoa_estandar']['gap_pct']:.2f}% | QAOA-XY P(opt) {d['qaoa_xy']['p_optimo']:.3f} "
             f"gap {d['qaoa_xy']['gap_pct']:.2f}% ({d['qaoa_estandar']['tiempo_s'] + d['qaoa_xy']['tiempo_s']:.1f} s)",
             verbose)

    # ------------------------------------------------------------------ soluciones completas
    def ensamblar(metodo):
        planes = []
        for sp in subproblemas:
            orden = R.sub[sp.id][metodo]["orden"]
            planes.append(RutaPlan(sp.tipo, sp.deposito, [sp.nodos[i] for i in orden], sp.id))
        return evaluar(inst, planes)

    nombres = {"exacto": "Clustering + exacto (fuerza bruta)", "sa": "Clustering + SA sobre QUBO",
               "ga": "Clustering + algoritmo genético", "qaoa_estandar": "Híbrido: clustering + QAOA",
               "qaoa_xy": "Híbrido: clustering + QAOA-XY"}
    for m, nombre in nombres.items():
        R.soluciones[nombre] = ensamblar(m)
        R.tiempos[nombre] = t_clasica + sum(R.sub[s.id][m].get("tiempo_s", 0.0) for s in subproblemas)

    for metodo, fn in (("SA", clasicos.sa_puro), ("GA", clasicos.ga_puro)):
        nombre = f"Clásico puro: {metodo} sin clustering"
        mejores, tiempos, Js = [], 0.0, []
        curvas = {}
        for rep in range(cfg.repeticiones):
            planes, t_rep = [], 0.0
            for g in inst.grupos:
                r = fn(inst, g, cfg, rng)
                planes += r["planes"]
                t_rep += r["tiempo_s"]
                if rep == 0:
                    curvas[f"{g.tipo}-{inst.esc.nodos[g.deposito].id}"] = r["curva"]
            sol = evaluar(inst, planes)
            Js.append(sol.J)
            mejores.append(sol)
            tiempos += t_rep
        k = int(np.argmin(Js))
        R.soluciones[nombre] = mejores[k]
        R.tiempos[nombre] = tiempos / cfg.repeticiones
        R.extra[f"puro_{metodo}"] = dict(J_mejor=float(min(Js)), J_medio=float(np.mean(Js)),
                                        J_peor=float(max(Js)), curvas=curvas)
        _log(f"[clásico puro {metodo}] J mejor={min(Js):.3f} medio={np.mean(Js):.3f} "
             f"({R.tiempos[nombre]:.1f} s por corrida)", verbose)

    for nombre, sol in R.soluciones.items():
        _log(f"  {nombre:40s} J={sol.J:9.3f}  entrega={sol.componentes['tiempo_total_entrega_h']:6.1f} h  "
             f"comb={sol.componentes['combustible_l']:7.0f} L  riesgo={sol.componentes['riesgo']:.3f}  "
             f"atrasos={sol.componentes['violaciones_ventana']}", verbose)
    return R


def _top_estados(sp, prep, P, k=24):
    orden = np.argsort(P)[::-1][:k]
    return [dict(indice=int(i), prob=float(P[i]), E=float(prep.E_total[i]), factible=bool(prep.factible[i]),
                 optimo=bool(prep.optimos[i])) for i in orden]


# ---------------------------------------------------------------------------
# guardado
# ---------------------------------------------------------------------------
def _serializable(o):
    if isinstance(o, dict):
        return {str(k): _serializable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_serializable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, np.ndarray):
        return _serializable(o.tolist())
    return o


def guardar(R: Resultados, dir_salida: str = DIR_RESULTADOS) -> dict:
    os.makedirs(dir_salida, exist_ok=True)
    esc = R.inst.esc
    rutas_out = {}

    # rutas detalladas
    with open(os.path.join(dir_salida, "rutas.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metodo", "vehiculo", "tipo", "deposito", "salida_h", "regreso_h", "parada", "comunidad",
                    "llegada_h", "ventana_h", "atraso_h", "carga_ruta_kg", "combustible_ruta_l", "riesgo_ruta"])
        for nombre, sol in R.soluciones.items():
            for r in sol.rutas:
                for k, i in enumerate(r.plan.nodos):
                    ven = R.inst.ventana[i]
                    w.writerow([nombre, r.vehiculo, r.plan.tipo, esc.nodos[r.plan.deposito].nombre,
                                f"{r.inicio_h:.3f}", f"{r.fin_h:.3f}", k + 1, esc.nodos[i].nombre,
                                f"{r.llegadas[i]:.3f}", f"{ven:.1f}", f"{max(0, r.llegadas[i] - ven):.3f}",
                                f"{r.carga_kg:.0f}", f"{r.combustible_l:.1f}", f"{r.riesgo:.4f}"])
            rutas_out[nombre] = [dict(vehiculo=r.vehiculo, tipo=r.plan.tipo, cluster=r.plan.etiqueta,
                                      deposito=esc.nodos[r.plan.deposito].id,
                                      paradas=[esc.nodos[i].id for i in r.plan.nodos],
                                      salida_h=r.inicio_h, regreso_h=r.fin_h) for r in sol.rutas]

    # métricas globales
    filas_g = []
    for nombre, sol in R.soluciones.items():
        c = sol.componentes
        filas_g.append(dict(metodo=nombre, J=c["J"], tiempo_total_entrega_h=c["tiempo_total_entrega_h"],
                            combustible_l=c["combustible_l"], riesgo=c["riesgo"], atraso_h=c["atraso_h"],
                            violaciones_ventana=c["violaciones_ventana"], makespan_h=c["makespan_h"],
                            n_rutas=c["n_rutas"], tiempo_computo_s=R.tiempos[nombre]))
    with open(os.path.join(dir_salida, "metricas_globales.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas_g[0].keys()))
        w.writeheader()
        w.writerows(filas_g)

    # métricas por subproblema
    filas_s = []
    for sid, d in R.sub.items():
        for m in ("exacto", "sa", "ga", "qaoa_estandar", "qaoa_xy"):
            x = d[m]
            filas_s.append(dict(
                subproblema=sid, tipo=d["tipo"], paradas=d["n"], qubits=d["nq"], metodo=m,
                E=x.get("E"), E_opt=d["E_opt"], gap_pct=x.get("gap_pct", 0.0),
                p_factible=x.get("p_factible", x.get("p_factible_medio", "")),
                p_optimo=x.get("p_optimo", x.get("p_optimo_medio", "")),
                tts=x.get("tts_disparos", x.get("tts_lecturas", "")),
                n_evaluaciones=x.get("n_evaluaciones"), tiempo_s=x.get("tiempo_s"),
                p_capas=x.get("p", ""),
                qpu_entrenamiento_s=(x.get("qpu") or {}).get("tiempo_entrenamiento_qpu_s", ""),
                prob_disparo_sin_error=(x.get("qpu") or {}).get("prob_disparo_sin_error", "")))
    with open(os.path.join(dir_salida, "metricas_subproblemas.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas_s[0].keys()))
        w.writeheader()
        w.writerows(filas_s)

    resumen = dict(
        configuracion={k: v for k, v in R.cfg.__dict__.items() if k != "flota"} | {"flota": R.cfg.flota},
        reduccion=R.reduccion, qiskit_disponible=R.extra.get("qiskit"),
        triage_modal={esc.nodos[i].id: dict(tipo=t, deposito=esc.nodos[R.inst.deposito_por_nodo[i]].id,
                                            motivo=R.inst.motivo_modal.get(i, ""))
                      for i, t in R.inst.tipo_por_nodo.items()},
        clusters={c.id: [esc.nodos[i].id for i in c.nodos] for c in R.clusters},
        subproblemas={sid: {k: v for k, v in d.items() if k not in ()} for sid, d in R.sub.items()},
        soluciones={n: s.componentes for n, s in R.soluciones.items()},
        tiempos_computo_s=R.tiempos, rutas=rutas_out,
        puro={k: {kk: vv for kk, vv in v.items() if kk != "curvas"} for k, v in R.extra.items()
              if k.startswith("puro_")},
    )
    with open(os.path.join(dir_salida, "resultados.json"), "w", encoding="utf-8") as f:
        json.dump(_serializable(resumen), f, ensure_ascii=False, indent=1)
    return resumen

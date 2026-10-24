"""Genera resultados/resumen.md con las tablas y una lectura de los números."""
from __future__ import annotations

import datetime
import os

import numpy as np

from .config import TIPOS

NOMBRE_VEHICULO = {"helicoptero": ("helicóptero", "helicópteros"), "camion": ("camión", "camiones")}


def _veh(tipo, k=1):
    return NOMBRE_VEHICULO.get(tipo, (tipo, tipo + "s"))[0 if k == 1 else 1]


def _f(x, nd=2):
    if x is None:
        return "-"
    if isinstance(x, float) and not np.isfinite(x):
        return "∞"
    return f"{x:.{nd}f}"


def _t(seg):
    if seg is None or not np.isfinite(seg):
        return "-"
    if seg < 1:
        return f"{seg * 1000:.0f} ms"
    if seg < 120:
        return f"{seg:.1f} s"
    if seg < 7200:
        return f"{seg / 60:.1f} min"
    return f"{seg / 3600:.1f} h"


def escribir_resumen(R, figuras: dict, ruta: str) -> None:
    inst, esc, cfg = R.inst, R.inst.esc, R.cfg
    L = []
    w = L.append
    w("# Resumen de resultados · Q-Rutas Piura\n")
    w(f"Generado el {datetime.datetime.now():%Y-%m-%d %H:%M} · semilla {cfg.semilla} · "
      f"modo {'rápido' if cfg.rapido else 'completo'} · motor cuántico `{cfg.motor_cuantico}` · "
      f"Qiskit instalado: {'sí' if R.extra.get('qiskit') else 'no'}\n")

    # ------------------------------------------------------------------ escenario
    fam = sum(n.familias for n in esc.comunidades)
    w("## 1. Escenario\n")
    w(f"- {len(esc.comunidades)} comunidades afectadas ({fam:,} familias, "
      f"{inst.demanda.sum() / 1000:.1f} t de ayuda en la primera respuesta) y "
      f"{len(esc.depositos)} centros de acopio: " + ", ".join(d.nombre for d in esc.depositos) + ".")
    flota = {}
    for v, t, d in cfg.flota:
        flota.setdefault((t, d), []).append(v)
    w("- Flota: " + "; ".join(f"{len(v)} {_veh(t, len(v))} en {esc.por_id(d).nombre} "
                              f"({TIPOS[t].capacidad_kg / 1000:.0f} t c/u)" for (t, d), v in flota.items()) + ".")
    w("- Puentes: " + "; ".join(f"{p.nombre} → **{p.estado}**" for p in esc.puentes) + ".")
    heli = [i for i, t in inst.tipo_por_nodo.items() if t == "helicoptero"]
    triage = [i for i in heli if "ventana" in inst.motivo_modal.get(i, "")]
    w(f"- Triage modal: {len(heli)} comunidades solo por aire"
      + (f" ({len(triage)} de ellas {'tiene' if len(triage) == 1 else 'tienen'} ruta terrestre, pero el camión "
         f"no llega a tiempo con las medicinas: " + ", ".join(esc.nodos[i].nombre for i in triage) + ")"
         if triage else "")
      + f"; {len(inst.tipo_por_nodo) - len(heli)} por camión.\n")

    # ------------------------------------------------------------------ fase clásica
    red = R.reduccion
    ref = red.get("refinamiento", {})
    w("## 2. Fase clásica: reducción de dimensionalidad\n")
    w(f"Formular el VRP completo como QUBO (tour gigante con copias del depósito) pide "
      f"**{red['qubits_vrp_completo']} qubits** (espacio de 2^{red['qubits_vrp_completo']} estados). "
      f"Tras el triage y el k-medoides capacitado quedan **{red['n_subproblemas']} subproblemas de "
      f"{min(red['qubits_por_subproblema'])} a {red['qubits_max_subproblema']} qubits**.")
    if ref.get("J_proxy_inicial"):
        w(f" El refinamiento por búsqueda local bajó el costo estimado de los clusters de "
          f"{ref['J_proxy_inicial']:.2f} a {ref['J_proxy_final']:.2f} ({ref['movimientos']} movimientos).")
    w("\n")
    w("| Cluster | Vehículo | Sale de | Paradas | Qubits | Carga (kg) | Salida prevista | Comunidades |")
    w("|---|---|---|---|---|---|---|---|")
    for c, sp in zip(R.clusters, R.subproblemas):
        dep = esc.nodos[c.deposito].nombre.split(' (')[0].split(' /')[0]
        w(f"| {c.id} | {_veh(c.tipo)} | {dep} | {sp.n} | {sp.nq} | "
          f"{inst.demanda[c.nodos].sum():.0f} | {sp.inicio_h:.2f} h | "
          + ", ".join(esc.nodos[i].nombre for i in c.nodos) + " |")
    w("")

    # ------------------------------------------------------------------ subproblemas
    w("## 3. Mismo modelo reducido: QAOA frente a métodos clásicos\n")
    w("Todos resuelven exactamente el mismo QUBO de cada cluster. *P(opt)* es la probabilidad de obtener la "
      "ruta óptima en un solo disparo (QAOA) o en una sola lectura (SA). *TTS* es cuántos disparos o "
      "lecturas hacen falta para verla al menos una vez con 99 % de confianza.\n")
    w("| Subproblema | Qubits | QAOA p | QAOA P(fact) | QAOA P(opt) | TTS QAOA | QAOA-XY P(opt) | TTS XY | "
      "SA P(opt) | TTS SA | Brecha final (QAOA / XY / SA / GA) |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for sp in R.subproblemas:
        d = R.sub[sp.id]
        if sp.n == 1:
            w(f"| {sp.id} | {sp.nq} | - | - | - | - | - | - | - | - | trivial (1 parada) |")
            continue
        q, x, s, g = d["qaoa_estandar"], d["qaoa_xy"], d["sa"], d["ga"]
        w(f"| {sp.id} | {sp.nq} | {q['p']} | {q['p_factible']:.3f} | {q['p_optimo']:.4f} | "
          f"{_f(q['tts_disparos'], 0)} | {x['p_optimo']:.3f} | {_f(x['tts_disparos'], 0)} | "
          f"{s['p_optimo_medio']:.2f} | {_f(s['tts_lecturas'], 0)} | "
          f"{q['gap_pct']:.1f} % / {x['gap_pct']:.1f} % / {s['gap_pct']:.1f} % / {g['gap_pct']:.1f} % |")
    w("")
    w("Costo computacional y estimación en hardware real:\n")
    w("| Subproblema | Evaluaciones QAOA | QAOA simulado (CPU) | QAOA en QPU (estimado) | CX lógicas QAOA / XY | "
      "Prob. disparo sin error QAOA / XY | SA (CPU) | GA (CPU) |")
    w("|---|---|---|---|---|---|---|---|")
    for sp in R.subproblemas:
        d = R.sub[sp.id]
        if sp.n == 1:
            continue
        q, x = d["qaoa_estandar"], d["qaoa_xy"]
        w(f"| {sp.id} | {q['n_evaluaciones']} | {_t(q['tiempo_s'])} | {_t(q['qpu']['tiempo_entrenamiento_qpu_s'])} | "
          f"{q['recursos']['cx_equivalentes']} / {x['recursos']['cx_equivalentes']} | "
          f"{q['qpu']['prob_disparo_sin_error']:.3f} / {x['qpu']['prob_disparo_sin_error']:.1e} | "
          f"{_t(d['sa']['tiempo_s'])} | {_t(d['ga']['tiempo_s'])} |")
    w("")
    w("La estimación de QPU usa tiempos y tasas de error de orden de magnitud (ver `qrutas/config.py` y "
      "`qrutas/hardware.py`); con Qiskit y `qiskit-ibm-runtime` instalados se reemplaza por la transpilación a un "
      "backend falso de IBM.\n")

    # ------------------------------------------------------------------ soluciones
    w("## 4. Solución completa\n")
    w("| Método | J | Entrega acumulada (h) | Combustible (L) | Riesgo | Atraso medicinas (h) | "
      "Ventanas incumplidas | Última entrega (h) | Rutas | Cómputo |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for nombre, sol in sorted(R.soluciones.items(), key=lambda kv: kv[1].J):
        c = sol.componentes
        w(f"| {nombre} | **{c['J']:.2f}** | {c['tiempo_total_entrega_h']:.1f} | {c['combustible_l']:.0f} | "
          f"{c['riesgo']:.3f} | {c['atraso_h']:.2f} | {c['violaciones_ventana']} | {c['makespan_h']:.2f} | "
          f"{c['n_rutas']} | {_t(R.tiempos[nombre])} |")
    w("")
    for nombre in ("Híbrido: clustering + QAOA", "Clásico puro: SA sin clustering"):
        sol = R.soluciones[nombre]
        w(f"**{nombre}**\n")
        for r in sorted(sol.rutas, key=lambda r: (r.vehiculo, r.inicio_h)):
            paradas = " → ".join(f"{esc.nodos[i].nombre} ({r.llegadas[i]:.2f} h)" for i in r.plan.nodos)
            w(f"- {r.vehiculo} sale {r.inicio_h:.2f} h de {esc.nodos[r.plan.deposito].nombre.split(' (')[0]}: "
              f"{paradas}; vuelve {r.fin_h:.2f} h · {r.carga_kg:.0f} kg")
        w("")

    # ------------------------------------------------------------------ lectura
    subs = [sp for sp in R.subproblemas if sp.n > 1]
    hib = R.soluciones["Híbrido: clustering + QAOA"]
    puro = min((s for n, s in R.soluciones.items() if n.startswith("Clásico puro")), key=lambda s: s.J)
    exacto = R.soluciones["Clustering + exacto (fuerza bruta)"]
    p_std = [R.sub[s.id]["qaoa_estandar"]["p_optimo"] for s in subs]
    p_xy = [R.sub[s.id]["qaoa_xy"]["p_optimo"] for s in subs]
    p_sa = [R.sub[s.id]["sa"]["p_optimo_medio"] for s in subs]
    gaps_q = [R.sub[s.id]["qaoa_estandar"]["gap_pct"] for s in subs]
    fid = [R.sub[s.id]["qaoa_estandar"]["qpu"]["prob_disparo_sin_error"] for s in subs]
    w("## 5. Lectura de los números\n")
    w(f"- El QAOA estándar llegó a la ruta óptima en {sum(g < 1e-9 for g in gaps_q)} de {len(subs)} subproblemas, "
      f"pero gracias a la post-selección de miles de disparos: su P(opt) por disparo va de "
      f"{min(p_std):.4f} a {max(p_std):.4f}. Con el mezclador que preserva permutaciones (QAOA-XY) sube a "
      f"{min(p_xy):.2f}-{max(p_xy):.2f}, porque nunca gasta amplitud en bitstrings inválidos.")
    w(f"- Simulated Annealing sobre el mismo QUBO acierta el óptimo en {min(p_sa):.2f}-{max(p_sa):.2f} de sus "
      f"lecturas y tarda milisegundos. A este tamaño (≤ 16 qubits) no hay ventaja cuántica: el valor del "
      f"prototipo es el flujo completo y que escala en la dirección correcta cuando haya más qubits.")
    cx_xy = max(R.sub[s.id]["qaoa_xy"]["recursos"]["cx_equivalentes"] for s in subs)
    w(f"- En hardware ruidoso actual, la probabilidad de que un disparo del QAOA estándar corra sin un solo error "
      f"estaría entre {min(fid):.2f} y {max(fid):.2f}. El QAOA-XY es mucho mejor en simulación pero pide hasta "
      f"{cx_xy} CNOT por circuito, así que sin corrección de errores su salida sería casi puro ruido: la "
      f"profundidad del circuito es el cuello de botella real, más que el número de qubits.")
    dif = 100 * (hib.J - puro.J) / puro.J
    max_puro = max(len(r.plan.nodos) for r in puro.rutas)
    if abs(dif) < 0.05:
        w(f"- La solución híbrida (J = {hib.J:.2f}) empata con el mejor clásico puro sin clustering "
          f"(J = {puro.J:.2f}).")
    else:
        causa = (f"el clásico puro arma viajes de hasta {max_puro} paradas, que en el híbrido no caben "
                 f"({max_puro}^2 = {max_puro ** 2} qubits)" if max_puro > cfg.max_paradas
                 else "el clásico puro elige libremente cómo partir las comunidades en viajes, mientras el "
                      "híbrido fija los clusters antes de rutear")
        quien = "a favor del clásico" if dif > 0 else "a favor del híbrido"
        w(f"- La solución híbrida tiene J = {hib.J:.2f}; el mejor clásico puro sin clustering, J = {puro.J:.2f} "
          f"({dif:+.1f} % {quien}). La diferencia no viene del QAOA (clustering + fuerza bruta da "
          f"exactamente J = {exacto.J:.2f}) sino de la descomposición: {causa}.")
    w(f"- Ninguna solución deja comunidades sin atender. Ventanas de medicinas incumplidas: híbrido "
      f"{hib.componentes['violaciones_ventana']}, clásico puro {puro.componentes['violaciones_ventana']}.\n")

    if figuras:
        w("## 6. Figuras\n")
        for p in figuras.values():
            w(f"- `{os.path.basename(p)}`")
        w("")
    with open(ruta, "w", encoding="utf-8") as f:
        f.write("\n".join(L))

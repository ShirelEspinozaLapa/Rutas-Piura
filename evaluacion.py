"""
Evaluación exacta de soluciones (la misma para todos los métodos).

Función objetivo (en unidades adimensionales "J"):

    J = wT * sum_i tau_i / T_ref            (tiempo total de entrega: suma de llegadas)
      + wF * combustible / F_ref
      + wR * riesgo / R_ref
      + lambda * sum_i max(0, tau_i - b_i)  (atraso respecto a la ventana de medicinas)
      + M * (#arcos sin ruta terrestre)     (puente caído / vía cortada)
      + M * exceso_de_carga / Q

La programación asigna rutas a vehículos del mismo tipo y depósito: si hay más
rutas que vehículos, un vehículo hace varios viajes (salida = fin del anterior +
recarga). Las rutas con medicinas más urgentes salen primero.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import TIPOS


@dataclass
class RutaPlan:
    tipo: str
    deposito: int
    nodos: list
    etiqueta: str = ""


@dataclass
class RutaEjecutada:
    plan: RutaPlan
    vehiculo: str
    inicio_h: float
    fin_h: float
    llegadas: dict
    combustible_l: float
    riesgo: float
    carga_kg: float
    arcos_bloqueados: int
    atraso_h: float
    J: float


@dataclass
class Solucion:
    rutas: list
    J: float
    componentes: dict = field(default_factory=dict)


def pesos(cfg):
    return cfg.w_tiempo / cfg.T_ref_h, cfg.w_combustible / cfg.F_ref_l, cfg.w_riesgo / cfg.R_ref


def simular_ruta(inst, plan: RutaPlan, inicio_h: float, vehiculo: str = "") -> RutaEjecutada:
    cfg = inst.cfg
    m = inst.mats[plan.tipo]
    tv = TIPOS[plan.tipo]
    WT, WF, WR = pesos(cfg)
    t = inicio_h + tv.carga_h
    prev = plan.deposito
    llegadas, comb, riesgo, bloq, atraso = {}, 0.0, 0.0, 0, 0.0
    for i in plan.nodos:
        bloq += int(not m.alcanzable[prev, i])
        t += m.T[prev, i]
        comb += m.F[prev, i]
        riesgo += m.R[prev, i]
        llegadas[i] = t
        atraso += max(0.0, t - inst.ventana[i])
        t += tv.servicio_h
        prev = i
    bloq += int(not m.alcanzable[prev, plan.deposito])
    t += m.T[prev, plan.deposito]
    comb += m.F[prev, plan.deposito]
    riesgo += m.R[prev, plan.deposito]
    carga = float(sum(inst.demanda[i] for i in plan.nodos))
    exceso = max(0.0, carga - tv.capacidad_kg)
    J = (WT * sum(llegadas.values()) + WF * comb + WR * riesgo + cfg.lambda_ventana * atraso
         + cfg.M_bloqueo * bloq + cfg.M_bloqueo * exceso / tv.capacidad_kg)
    return RutaEjecutada(plan=plan, vehiculo=vehiculo, inicio_h=inicio_h, fin_h=t, llegadas=llegadas,
                         combustible_l=comb, riesgo=riesgo, carga_kg=carga, arcos_bloqueados=bloq,
                         atraso_h=atraso, J=J)


def _clave_urgencia(inst, plan: RutaPlan):
    if not plan.nodos:
        return (np.inf, 0)
    return (min(inst.ventana[i] for i in plan.nodos), -len(plan.nodos))


def programar(inst, planes: list) -> list:
    """Asigna rutas a vehículos (varios viajes si hace falta) y simula cada una."""
    cfg = inst.cfg
    idx_dep = {d.id: d.idx for d in inst.esc.depositos}
    ejecutadas = []
    claves = sorted({(p.tipo, p.deposito) for p in planes})
    for tipo, dep in claves:
        vehs = [v for v, t, d in cfg.flota if t == tipo and idx_dep[d] == dep]
        if not vehs:
            raise ValueError(f"No hay vehículos {tipo} en el depósito {inst.esc.nodos[dep].id}")
        libre = {v: 0.0 for v in vehs}
        grupo = sorted([p for p in planes if p.tipo == tipo and p.deposito == dep],
                       key=lambda p: _clave_urgencia(inst, p))
        for plan in grupo:
            if not plan.nodos:
                continue
            v = min(vehs, key=lambda x: (libre[x], x))
            ej = simular_ruta(inst, plan, libre[v], v)
            libre[v] = ej.fin_h + TIPOS[tipo].recarga_h
            ejecutadas.append(ej)
    return ejecutadas


def evaluar(inst, planes: list) -> Solucion:
    rutas = programar(inst, planes)
    cfg = inst.cfg
    WT, WF, WR = pesos(cfg)
    atendidos = [i for r in rutas for i in r.plan.nodos]
    faltan = set(inst.comunidades) - set(atendidos)
    duplicados = len(atendidos) - len(set(atendidos))
    lat = sum(sum(r.llegadas.values()) for r in rutas)
    comb = sum(r.combustible_l for r in rutas)
    riesgo = sum(r.riesgo for r in rutas)
    atraso = sum(r.atraso_h for r in rutas)
    viol = sum(1 for r in rutas for i, t in r.llegadas.items() if t > inst.ventana[i] + 1e-9)
    bloq = sum(r.arcos_bloqueados for r in rutas)
    J = sum(r.J for r in rutas) + cfg.M_bloqueo * (len(faltan) + duplicados)
    comp = dict(
        J=J,
        termino_tiempo=WT * lat, termino_combustible=WF * comb, termino_riesgo=WR * riesgo,
        termino_atraso=cfg.lambda_ventana * atraso,
        tiempo_total_entrega_h=lat, combustible_l=comb, riesgo=riesgo, atraso_h=atraso,
        violaciones_ventana=viol, arcos_bloqueados=bloq, comunidades_sin_atender=len(faltan),
        makespan_h=max((max(r.llegadas.values()) for r in rutas if r.llegadas), default=0.0),
        n_rutas=len(rutas),
        familias_atendidas=int(sum(inst.esc.nodos[i].familias for i in set(atendidos))),
    )
    return Solucion(rutas=rutas, J=J, componentes=comp)

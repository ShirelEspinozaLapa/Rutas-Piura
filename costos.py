"""
Matrices de costo por tipo de vehículo, accesibilidad y armado de la instancia.

Para cada par de nodos (i, j) y cada tipo de vehículo se calcula:
    T[i, j]  tiempo de viaje (h)
    F[i, j]  combustible (L)
    R[i, j]  índice de riesgo (prob. aproximada de incidente, aditiva)
    alcanzable[i, j]  False si no existe ruta (puente caído / vía cortada)

Camión: camino de menor costo compuesto en la red vial transitable (Dijkstra).
Helicóptero: línea recta (gran círculo); no le afectan puentes ni anegamientos.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from . import geo
from .config import TIPOS, Config, TipoVehiculo
from .datos import Escenario
from .red import RedVial, construir_red_vial, subgrafo_transitable


@dataclass
class MatricesModo:
    tipo: TipoVehiculo
    T: np.ndarray
    F: np.ndarray
    R: np.ndarray
    alcanzable: np.ndarray
    caminos: dict = field(default_factory=dict)   # (i, j) -> lista de puntos del grafo (solo camión)

    def compuesto(self, cfg: Config) -> np.ndarray:
        c = (cfg.w_tiempo * self.T / cfg.T_ref_h + cfg.w_combustible * self.F / cfg.F_ref_l
             + cfg.w_riesgo * self.R / cfg.R_ref)
        return np.where(self.alcanzable, c, c + cfg.M_bloqueo)


@dataclass
class Grupo:
    """Comunidades atendidas por un mismo tipo de vehículo desde un mismo depósito."""
    tipo: str
    deposito: int
    nodos: list
    vehiculos: list


@dataclass
class Instancia:
    cfg: Config
    esc: Escenario
    red: RedVial
    mats: dict                  # nombre de tipo -> MatricesModo
    tipo_por_nodo: dict         # idx comunidad -> nombre de tipo
    deposito_por_nodo: dict     # idx comunidad -> idx depósito
    ventana: np.ndarray         # fin de ventana de medicinas (h) por nodo
    demanda: np.ndarray         # kg por nodo
    grupos: list
    motivo_modal: dict = field(default_factory=dict)

    @property
    def comunidades(self):
        return sorted(self.tipo_por_nodo)


def matrices_helicoptero(esc: Escenario, tv: TipoVehiculo) -> MatricesModo:
    d = geo.matriz_haversine(esc.lat, esc.lon)
    T = d / tv.velocidad_kmh
    F = T * tv.consumo_l_h
    es_com = np.array([n.tipo == "comunidad" for n in esc.nodos])
    R = T * tv.riesgo_h + np.where(es_com[None, :], tv.riesgo_parada, 0.0)
    np.fill_diagonal(R, 0.0)
    return MatricesModo(tipo=tv, T=T, F=F, R=R, alcanzable=np.ones_like(T, dtype=bool))


def matrices_camion(esc: Escenario, red: RedVial, tv: TipoVehiculo) -> MatricesModo:
    n = red.n_escenario
    H = subgrafo_transitable(red)
    T = np.zeros((n, n))
    F = np.zeros((n, n))
    R = np.zeros((n, n))
    alc = np.zeros((n, n), dtype=bool)
    caminos = {}
    es_com = np.array([nd.tipo == "comunidad" for nd in esc.nodos])
    recta = geo.matriz_haversine(esc.lat, esc.lon)
    for i in range(n):
        dist, rutas = nx.single_source_dijkstra(H, i, weight="peso")
        for j in range(n):
            if i == j:
                alc[i, j] = True
                continue
            if j in rutas:
                cam = rutas[j]
                aristas = [H.edges[a, b] for a, b in zip(cam[:-1], cam[1:])]
                T[i, j] = sum(e["t_h"] for e in aristas)
                F[i, j] = sum(e["comb_l"] for e in aristas)
                R[i, j] = sum(e["riesgo"] for e in aristas) + (tv.riesgo_parada if es_com[j] else 0.0)
                alc[i, j] = True
                caminos[(i, j)] = cam
            else:
                # Sin ruta: valores finitos de referencia (el castigo real lo pone M_bloqueo)
                km = recta[i, j] * 1.3
                T[i, j] = km / (0.5 * tv.velocidad_kmh)
                F[i, j] = km * tv.consumo_l_km
                R[i, j] = km * tv.riesgo_km * 5
    return MatricesModo(tipo=tv, T=T, F=F, R=R, alcanzable=alc, caminos=caminos)


def construir_instancia(esc: Escenario, cfg: Config) -> Instancia:
    red = construir_red_vial(esc, cfg)
    mats = {"helicoptero": matrices_helicoptero(esc, TIPOS["helicoptero"]),
            "camion": matrices_camion(esc, red, TIPOS["camion"])}

    idx_dep = {d.id: d.idx for d in esc.depositos}
    deps_camion = sorted({idx_dep[d] for _, t, d in cfg.flota if t == "camion"})
    dep_aereo = idx_dep[cfg.deposito_aereo]
    comp_cam = mats["camion"].compuesto(cfg)

    # Triage modal (fase clásica):
    #  - sin ruta terrestre (puente caído, vía anegada) -> helicóptero
    #  - con prioridad médica y el camión no llega dentro de la ventana ni yendo directo -> helicóptero
    #  - resto -> camión desde el depósito con menor costo de ida y vuelta
    tipo_por_nodo, dep_por_nodo, motivo_modal = {}, {}, {}
    cam = mats["camion"]
    for c in esc.comunidades:
        i = c.idx
        opciones = [(comp_cam[d, i] + comp_cam[i, d], d) for d in deps_camion
                    if cam.alcanzable[d, i] and cam.alcanzable[i, d]]
        if not opciones:
            tipo_por_nodo[i], dep_por_nodo[i] = "helicoptero", dep_aereo
            motivo_modal[i] = "sin ruta terrestre"
            continue
        d_best = min(opciones)[1]
        llegada_min = cam.tipo.carga_h + cam.T[d_best, i]
        if c.prioridad_medica and llegada_min > cfg.ventana_medicina_h["terrestre"]:
            tipo_por_nodo[i], dep_por_nodo[i] = "helicoptero", dep_aereo
            motivo_modal[i] = f"camión llegaría en {llegada_min:.1f} h > ventana"
        else:
            tipo_por_nodo[i], dep_por_nodo[i] = "camion", d_best
            motivo_modal[i] = "ruta terrestre disponible"

    N = len(esc.nodos)
    ventana = np.full(N, np.inf)
    demanda = np.zeros(N)
    for c in esc.comunidades:
        modo = TIPOS[tipo_por_nodo[c.idx]].modo
        ventana[c.idx] = cfg.ventana_medicina_h[modo] if c.prioridad_medica else cfg.ventana_general_h
        demanda[c.idx] = c.demanda_kg

    grupos = []
    for t in ("helicoptero", "camion"):
        for d in sorted({dep_por_nodo[i] for i in tipo_por_nodo if tipo_por_nodo[i] == t}):
            nodos = sorted(i for i in tipo_por_nodo if tipo_por_nodo[i] == t and dep_por_nodo[i] == d)
            veh = [v for v, tt, dd in cfg.flota if tt == t and idx_dep[dd] == d]
            if not veh:
                raise ValueError(f"No hay vehículos tipo {t} en el depósito {esc.nodos[d].id}")
            grupos.append(Grupo(tipo=t, deposito=d, nodos=nodos, vehiculos=veh))

    return Instancia(cfg=cfg, esc=esc, red=red, mats=mats, tipo_por_nodo=tipo_por_nodo,
                     deposito_por_nodo=dep_por_nodo, ventana=ventana, demanda=demanda, grupos=grupos,
                     motivo_modal=motivo_modal)

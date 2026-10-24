"""
Red vial simplificada del Bajo Piura.

Dentro del entorno de la hackatón no hay acceso a OpenStreetMap, así que la red
se arma con corredores explícitos (datos/corredores.csv) entre nodos reales,
puntos auxiliares y los extremos de cada puente. Los trazos son aproximados
(rectos entre puntos) y la longitud se corrige con un factor de sinuosidad.

Reglas físicas que sí se respetan:
1. Ningún corredor cruza el río salvo por un puente (se verifica al construir).
2. Los puentes son aristas con estado (operativo, restringido, cerrado, colapsado).
3. Los tramos que llegan a una comunidad con acceso terrestre "cortado" quedan
   bloqueados (vía anegada o cortada por quebrada activa).

Para un despliegue real se reemplaza este módulo por una red de OSM
(osmnx.graph_from_place) conservando los mismos atributos de arista.
"""
from __future__ import annotations

import csv
import os
import warnings
from dataclasses import dataclass

import networkx as nx
import numpy as np

from . import geo
from .config import CAMION, DIR_DATOS, Config
from .datos import Escenario

# tipo de vía -> (factor de velocidad, factor de riesgo)
TIPOS_VIA = {
    "carretera": (1.00, 1.0),
    "urbana": (0.60, 1.0),
    "afirmado": (0.75, 1.5),
    "trocha": (0.45, 2.5),
    "puente": (1.00, 1.0),
}


@dataclass
class RedVial:
    G: nx.Graph
    xy: np.ndarray            # (m, 2) km de todos los puntos del grafo
    lat: np.ndarray
    lon: np.ndarray
    etiquetas: list           # nombre de cada punto
    claves: list              # id corto de cada punto (C01, A1, P5-MD, ...)
    n_escenario: int          # los primeros n puntos son los nodos del escenario

    def aristas(self, bloqueadas: bool | None = None):
        for u, v, d in self.G.edges(data=True):
            if bloqueadas is None or d["bloqueada"] == bloqueadas:
                yield u, v, d


def _extremos_puente(rio_xy: np.ndarray, p: np.ndarray, offset_km: float):
    """Extremos del puente a ambos lados del cauce, perpendiculares al río.
    Devuelve (extremo margen izquierda, extremo margen derecha)."""
    dist = [geo.dist_punto_segmento(p, rio_xy[k], rio_xy[k + 1]) for k in range(len(rio_xy) - 1)]
    k = int(np.argmin(dist))
    d = rio_xy[k + 1] - rio_xy[k]
    d = d / np.linalg.norm(d)
    normal = np.array([-d[1], d[0]])      # izquierda del sentido aguas abajo
    return p + offset_km * normal, p - offset_km * normal


def construir_red_vial(esc: Escenario, cfg: Config, dir_datos: str = DIR_DATOS) -> RedVial:
    n = len(esc.nodos)
    pts = [nd.xy for nd in esc.nodos]
    etiquetas = [nd.nombre for nd in esc.nodos]
    claves = [nd.id for nd in esc.nodos]

    with open(os.path.join(dir_datos, "puntos_auxiliares.csv"), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pts.append(geo.a_km(float(r["lat"]), float(r["lon"])))
            etiquetas.append(r["nombre"])
            claves.append(r["id"])

    offset_km = cfg.offset_puente_grados * 111.0
    puentes_idx = {}
    for k, pu in enumerate(esc.puentes):
        mi, md = _extremos_puente(esc.rio_xy, pu.xy, offset_km)
        pts += [mi, md]
        etiquetas += [f"{pu.nombre} (MI)", f"{pu.nombre} (MD)"]
        claves += [f"{pu.id}-MI", f"{pu.id}-MD"]
        puentes_idx[k] = (len(pts) - 2, len(pts) - 1)

    xy = np.array(pts)
    lat, lon = geo.a_latlon(xy)
    idx = {c: i for i, c in enumerate(claves)}

    G = nx.Graph()
    for i in range(len(xy)):
        G.add_node(i, xy=xy[i], etiqueta=etiquetas[i], clave=claves[i], escenario=i < n)

    with open(os.path.join(dir_datos, "corredores.csv"), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            u, v = idx[r["desde"]], idx[r["hasta"]]
            if geo.cruza_polilinea(xy[u], xy[v], esc.rio_xy):
                warnings.warn(f"El corredor {r['desde']}-{r['hasta']} cruza el río sin puente; se omite.", stacklevel=2)
                continue
            G.add_edge(u, v, puente=None, tipo=r["tipo"], nombre=r["nombre"])

    for k, (u, v) in puentes_idx.items():
        G.add_edge(u, v, puente=k, tipo="puente", nombre=esc.puentes[k].nombre)

    sin_aristas = [etiquetas[i] for i in range(n) if G.degree(i) == 0]
    if sin_aristas:
        warnings.warn(f"Nodos sin ningún corredor: {sin_aristas}", stacklevel=2)

    cortados = {nd.idx for nd in esc.nodos if nd.acceso_terrestre == "cortado"}
    for u, v, d in G.edges(data=True):
        recta = float(np.linalg.norm(xy[u] - xy[v]))
        f_vel, f_riesgo = TIPOS_VIA[d["tipo"]]
        if d["puente"] is not None:
            largo, expo, estado = recta, 1.0, esc.puentes[d["puente"]].estado
        else:
            largo = recta * cfg.sinuosidad
            d_rio = geo.dist_segmento_polilinea(xy[u], xy[v], esc.rio_xy)
            expo = float(np.exp(-d_rio / cfg.escala_exposicion_km))
            estado = "operativo"
        motivo = ""
        if estado in ("cerrado", "colapsado"):
            motivo = f"{esc.puentes[d['puente']].nombre}: {estado}"
        elif u in cortados or v in cortados:
            k = u if u in cortados else v
            motivo = f"acceso cortado a {esc.nodos[k].nombre}"
        v_kmh = CAMION.velocidad_kmh * f_vel * (1 - cfg.reduccion_velocidad * expo)
        t_h = largo / v_kmh
        riesgo = largo * CAMION.riesgo_km * f_riesgo * (1 + cfg.factor_riesgo_exposicion * expo)
        if estado == "restringido":
            t_h *= cfg.puente_restringido_factor_t
            riesgo += cfg.puente_restringido_riesgo
        comb = largo * CAMION.consumo_l_km
        d.update(largo_km=largo, exposicion=expo, estado=estado, bloqueada=bool(motivo),
                 motivo=motivo, t_h=t_h, comb_l=comb, riesgo=riesgo,
                 peso=cfg.w_tiempo * t_h / cfg.T_ref_h + cfg.w_combustible * comb / cfg.F_ref_l
                 + cfg.w_riesgo * riesgo / cfg.R_ref)
    return RedVial(G=G, xy=xy, lat=lat, lon=lon, etiquetas=etiquetas, claves=claves, n_escenario=n)


def subgrafo_transitable(red: RedVial) -> nx.Graph:
    H = nx.Graph()
    H.add_nodes_from(red.G.nodes(data=True))
    H.add_edges_from((u, v, d) for u, v, d in red.G.edges(data=True) if not d["bloqueada"])
    return H
